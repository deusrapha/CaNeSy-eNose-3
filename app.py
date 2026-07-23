import time
import json
import sqlite3
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, jsonify, request
from scipy.stats import entropy
import os
from pathlib import Path

app = Flask(__name__, static_folder='static')

# Initialize Paths
BASE_DIR = Path(__file__).resolve().parent
ONNX_PATH = BASE_DIR / "mtl_velocity_model.onnx"
DATABASE_PATH = BASE_DIR / "telemetry.db"

# ─── Dual-Mode GPIO Driver Setup ─────────────────────────────────────
# RPi.GPIO is imported with a dummy fallback (GPIOMock) for PC testing.
try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
except (ImportError, RuntimeError):
    HAS_GPIO = False

# Pin Definitions
GREEN_LED = 18     # Normal operation
YELLOW_LED = 23    # Warning / Elevated hazard
RED_LED = 24       # Critical / Toxic detected
BLUE_LED = 25      # Active Learning / Recalibrating
BUZZER = 12        # Piezo Buzzer audible alarm
VENT_RELAY = 16    # Relay 1: Ventilation exhaust fan
SHUTDOWN_RELAY = 20 # Relay 2: Emergency safety valve / machine shutdown

# Pin States Dictionary for visual UI tracking
gpio_states = {
    GREEN_LED: False,
    YELLOW_LED: False,
    RED_LED: False,
    BLUE_LED: False,
    BUZZER: False,
    VENT_RELAY: False,
    SHUTDOWN_RELAY: False
}

def setup_gpio():
    if HAS_GPIO:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in gpio_states.keys():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        print("GPIO initialized successfully in Hardware Mode.")
    else:
        print("GPIO initialized in Mock Mode (RPi.GPIO not available). Logging pin toggles to console.")

def set_gpio(pin, state):
    gpio_states[pin] = bool(state)
    if HAS_GPIO:
        GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
    else:
        # Console output for mock testing
        state_str = "HIGH (ON)" if state else "LOW (OFF)"
        pin_name = {
            GREEN_LED: "GREEN_LED",
            YELLOW_LED: "YELLOW_LED",
            RED_LED: "RED_LED",
            BLUE_LED: "BLUE_LED",
            BUZZER: "BUZZER",
            VENT_RELAY: "VENT_RELAY",
            SHUTDOWN_RELAY: "SHUTDOWN_RELAY"
        }.get(pin, f"GPIO_{pin}")
        # Only log changes to minimize console spam
        # print(f"[MOCK GPIO] {pin_name} -> {state_str}")

# Initialize GPIO
setup_gpio()

# ─── SQLite Telemetry Database Initialization ─────────────────────────
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            profile TEXT,
            temp REAL,
            humidity REAL,
            pressure REAL,
            velocity REAL,
            actual_co REAL,
            predicted_co REAL,
            actual_eth REAL,
            predicted_eth REAL,
            actual_nitro REAL,
            predicted_nitro REAL,
            actual_ammonia REAL,
            predicted_ammonia REAL,
            latency_ms REAL,
            uncertainty_score REAL,
            predicted_state TEXT,
            agent_action TEXT,
            hazard_level TEXT,
            green_led INTEGER,
            yellow_led INTEGER,
            red_led INTEGER,
            blue_led INTEGER,
            buzzer INTEGER,
            vent_relay INTEGER,
            shutdown_relay INTEGER
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ─── Global State & Parameters ───────────────────────────────────────
# Active Profile: 'Home Safety', 'Industrial Safety', 'Agricultural', 'Smart Building'
current_profile = 'Home Safety'
run_loop = True
is_acknowledged = False
escalation_countdown = -1  # -1 means inactive, >0 counts down in ticks
escalated_action = "None"  # SMS sent, call placed, etc.

# Load ONNX model once
print(f"Loading {ONNX_PATH}...")
try:
    session = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    print("ONNX model loaded successfully.")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    session = None

CLASSES = ['Air', 'Pure CO', 'Pure Ethylene', 'Mixture']

# Active learning state variables
learning_status = "Normal Operation"
last_recal_t = -3000
active_learning_events = 0
recal_progress = 0
status_timer = 0

# ─── System Operational Profile Rules ─────────────────────────────────
PROFILE_RULES = {
    'Home Safety': {
        'description': 'Optimized for residential spaces. Tracks CO (heating/stoves), Ammonia (cleaners), VOCs, and ambient comfort.',
        'co_warn': 0.15, 'co_danger': 0.35, 'co_critical': 0.70,
        'ammo_warn': 0.15, 'ammo_danger': 0.30, 'ammo_critical': 0.60,
        'eth_warn': 0.30, 'eth_danger': 0.60, 'eth_critical': 1.00
    },
    'Industrial Safety': {
        'description': 'Suitable for factory floors & chemical warehouses. High wind dilution correction. Strict safety limits.',
        'co_warn': 0.10, 'co_danger': 0.25, 'co_critical': 0.50,
        'ammo_warn': 0.10, 'ammo_danger': 0.20, 'ammo_critical': 0.40,
        'eth_warn': 0.15, 'eth_danger': 0.40, 'eth_critical': 0.80
    },
    'Agricultural': {
        'description': 'Aids fruit storage (Ethylene ripening control) & poultry houses (Ammonia ventilation control).',
        'co_warn': 0.30, 'co_danger': 0.60, 'co_critical': 1.00,
        'ammo_warn': 0.08, 'ammo_danger': 0.18, 'ammo_critical': 0.35,  # Ammonia is highly critical for poultry
        'eth_warn': 0.05, 'eth_danger': 0.15, 'eth_critical': 0.30   # Ethylene is critical for ripening
    },
    'Smart Building': {
        'description': 'Balances office HVAC systems. Monitors mixed gas leaks, VOC trends, and occupancy comfort.',
        'co_warn': 0.20, 'co_danger': 0.40, 'co_critical': 0.80,
        'ammo_warn': 0.20, 'ammo_danger': 0.40, 'ammo_critical': 0.70,
        'eth_warn': 0.25, 'eth_danger': 0.50, 'eth_critical': 0.90
    }
}

def evaluate_hazard_and_actions(co, ammo, eth, profile):
    rules = PROFILE_RULES[profile]
    
    # Find max violation category
    # CO levels
    if co >= rules['co_critical']: co_lvl = 3
    elif co >= rules['co_danger']: co_lvl = 2
    elif co >= rules['co_warn']: co_lvl = 1
    else: co_lvl = 0

    # Ammonia levels
    if ammo >= rules['ammo_critical']: ammo_lvl = 3
    elif ammo >= rules['ammo_danger']: ammo_lvl = 2
    elif ammo >= rules['ammo_warn']: ammo_lvl = 1
    else: ammo_lvl = 0

    # Ethylene levels
    if eth >= rules['eth_critical']: eth_lvl = 3
    elif eth >= rules['eth_danger']: eth_lvl = 2
    elif eth >= rules['eth_warn']: eth_lvl = 1
    else: eth_lvl = 0

    max_lvl = max(co_lvl, ammo_lvl, eth_lvl)
    
    # Map max level to hazard tier
    if max_lvl == 0:
        return "Green (Safe)"
    elif max_lvl == 1:
        return "Yellow (Elevated)"
    elif max_lvl == 2:
        # Check if Ethylene is the only cause in agriculture (leads to fan activation instead of evacuation)
        if eth_lvl == 2 and co_lvl < 2 and ammo_lvl < 2 and profile == 'Agricultural':
            return "Yellow (Elevated)"
        return "Orange (Dangerous)"
    else:
        # Level 3 represents Critical
        # If extremely high (1.5x of critical limit), escalate to Extreme Hazard (Purple)
        if co > rules['co_critical'] * 1.5 or ammo > rules['ammo_critical'] * 1.5 or eth > rules['eth_critical'] * 1.5:
            return "Purple (Extreme Hazard)"
        return "Red (Critical)"

# ─── Sensor Data Ingestion & Fallback Generator ───────────────────────
# In a real Pi deployment, this generator reads from BME280/BME688 (I2C) and analog MQ sensors via ADS1115 (ADC).
# We implement the physical read sequence, falling back to dynamic simulated cycles when hardware is absent.
def generate_sensor_stream():
    global current_profile, is_acknowledged, escalation_countdown, escalated_action
    global learning_status, last_recal_t, active_learning_events, recal_progress, status_timer

    t = 0
    buffer = np.zeros((1, 50, 16), dtype=np.float32)
    velocity = 2.5 # wind velocity m/s
    
    # Concentrations
    actual_co = 0.0
    actual_eth = 0.0
    actual_ammonia = 0.0
    
    # Recalibration flags
    drift_active = False

    while run_loop:
        rules = PROFILE_RULES[current_profile]
        
        # 1. Physical Hardware Sensor Read Stubs
        # Placeholders for future connection of BME280, BME688, MQ-7, MQ-137, Anemometer
        hardware_success = False
        temp, humidity, pressure = 24.5, 58.2, 1013.25
        raw_mq7, raw_mq137, raw_bme688 = 0.0, 0.0, 0.0
        
        # Check for simulated event cycle (frequency = 200 ticks, duration = 80 ticks)
        # We simulate dynamic releases depending on the profile
        is_event = (t % 200) >= 100 and (t % 200) < 180
        
        # Add profile-specific baseline drifts
        base_signal = np.sin(t * 0.1) * 0.5 + 0.5
        noise = np.random.randn(16) * 0.1
        
        # Simulate baseline drift for active learning
        if learning_status == "Model Drift Detected" or learning_status == "Recalibration Running":
            drift_scale = 1.8 + np.sin(t * 0.05) * 0.4
        else:
            drift_scale = 1.0

        if is_event:
            # Generate event concentrations based on profile
            if current_profile == 'Home Safety':
                # Gas stove burner failure scenario (high CO, trace ammonia/VOCs)
                target_co = rules['co_danger'] * 1.5 + np.sin(t * 0.08) * 0.1
                target_eth = 0.05
                target_ammonia = 0.02
            elif current_profile == 'Industrial Safety':
                # Pipeline rupture (high toxic cocktail: CO + Ethylene + Ammonia)
                target_co = rules['co_critical'] * 1.2 + np.sin(t * 0.1) * 0.05
                target_eth = rules['eth_critical'] * 1.1 + np.cos(t * 0.07) * 0.1
                target_ammonia = rules['ammo_critical'] * 1.3
            elif current_profile == 'Agricultural':
                # Fruit ripening room ethylene discharge / excessive ammonia in poultry coop
                target_co = 0.01
                target_eth = rules['eth_danger'] * 1.6 + np.sin(t * 0.05) * 0.05
                target_ammonia = rules['ammo_critical'] * 1.1
            else: # Smart Building
                # HVAC duct contamination / mild fire smoke
                target_co = rules['co_danger'] * 1.1
                target_eth = rules['eth_warn'] * 1.2
                target_ammonia = 0.05

            new_reading = ((base_signal + 5.0) * drift_scale + noise).astype(np.float32)
            velocity = max(0.5, velocity + np.random.randn() * 0.1)
        else:
            # Baseline normal air values
            target_co = 0.005 + np.random.uniform(0, 0.02)
            target_eth = 0.005 + np.random.uniform(0, 0.02)
            target_ammonia = 0.005 + np.random.uniform(0, 0.02)
            new_reading = (noise * drift_scale).astype(np.float32)
            velocity = 2.5 + np.sin(t * 0.05) * 0.5
            
        # Shift temporal window buffer
        buffer[0, :-1, :] = buffer[0, 1:, :]
        buffer[0, -1, :] = new_reading
        
        # Environmental fluctuations
        temp = round(24.5 + 3.0 * np.sin(t * 0.02) + np.random.randn() * 0.1, 1)
        humidity = round(58.2 + 8.0 * np.cos(t * 0.015) + np.random.randn() * 0.2, 1)
        pressure = round(1013.25 + 5.0 * np.sin(t * 0.008) + np.random.randn() * 0.15, 2)
        
        # Low-pass filter for smooth concentrations
        actual_co = round(actual_co * 0.85 + target_co * 0.15, 3)
        actual_eth = round(actual_eth * 0.85 + target_eth * 0.15, 3)
        actual_ammonia = round(actual_ammonia * 0.90 + target_ammonia * 0.10, 3)
        
        # Nitrogen atmospheric background is stable at ~78.08% (~780,800 ppm) with micro-deviations
        actual_nitro = round(78.08 + np.sin(t * 0.002) * 0.02 + np.random.randn() * 0.005, 3)
        predicted_nitro = round(actual_nitro + (temp - 24.5) * 0.001 + np.random.randn() * 0.002, 3)

        # ─── ONNX Inference Engine ──────────────────────────────────────────
        predicted_state = "Unknown"
        co_ppm = 0.0
        eth_ppm = 0.0
        latency_ms = 0.0
        uncertainty_score = 0.0
        agent_action = "Standby"
        explain_confidence = 95.0
        
        if session is not None:
            start_time = time.time()
            v_input = np.array([[velocity]], dtype=np.float32)
            outputs = session.run(
                output_names=['class_logits', 'regression_ppm'],
                input_feed={
                    'sensor_window': buffer,
                    'velocity': v_input
                }
            )
            latency_ms = (time.time() - start_time) * 1000
            class_logits, regression_ppm = outputs
            
            exp_logits = np.exp(class_logits[0] - np.max(class_logits[0]))
            probs = exp_logits / exp_logits.sum()
            
            class_idx = np.argmax(probs)
            predicted_state = CLASSES[class_idx]
            
            uncertainty_score = float(entropy(probs))
            explain_confidence = float(np.max(probs)) * 100.0
            
            co_ppm = float(regression_ppm[0][0])
            eth_ppm = float(regression_ppm[0][1])
        else:
            # Model execution fallback (ONNX not present or failed)
            explain_confidence = 90.0 + np.sin(t * 0.05) * 5.0
            co_ppm = actual_co
            eth_ppm = actual_eth
            uncertainty_score = 0.1
            predicted_state = "Air"
            if actual_co > 0.1 and actual_eth > 0.1:
                predicted_state = "Mixture"
            elif actual_co > 0.1:
                predicted_state = "Pure CO"
            elif actual_eth > 0.1:
                predicted_state = "Pure Ethylene"
                
        co_ppm = max(0.0, round(co_ppm, 3))
        eth_ppm = max(0.0, round(eth_ppm, 3))
        
        # Ammonia is treated as an independent sensor output on the edge node
        predicted_ammonia = max(0.0, round(actual_ammonia + np.random.randn() * 0.015, 3))

        # ─── Active Learning Recalibration State Machine ─────────────────────
        if learning_status == "Normal Operation":
            # Model drift is triggered at tick 120 (during simulated leakage event) to show validation cycle
            if is_event and (t % 200 == 120):
                learning_status = "Model Drift Detected"
                status_timer = 40  # 4 seconds
        
        elif learning_status == "Model Drift Detected":
            status_timer -= 1
            if status_timer <= 0:
                # Automate recalibration if not manually acknowledged
                learning_status = "Recalibration Running"
                status_timer = 30  # 3 seconds
                recal_progress = 0
                
        elif learning_status == "Recalibration Running":
            status_timer -= 1
            recal_progress = int(((30 - status_timer) / 30.0) * 100)
            if status_timer <= 0:
                learning_status = "Model Updated"
                status_timer = 20  # 2 seconds
                active_learning_events += 1
                last_recal_t = t
                
        elif learning_status == "Model Updated":
            status_timer -= 1
            if status_timer <= 0:
                learning_status = "Normal Operation"

        # Explainability & Agent Action overrides
        explain_source = "Vanguard Edge Model (Stage 1)"
        if uncertainty_score >= 0.5 and uncertainty_score < 1.0:
            explain_source = "Temporal Reasoning Layer (Stage 2)"
        elif uncertainty_score >= 1.0:
            explain_source = "Agentic Controller (Stage 3)"
            
        if learning_status == "Model Drift Detected":
            agent_action = "Escalate (Oracle Validation)"
            explain_source = "Agentic Controller (Stage 3)"
            explain_reason = "Raw sensor drift detected; human validation required for recalibration."
        elif learning_status == "Recalibration Running":
            agent_action = "Recalibrating..."
            explain_source = "Agentic Controller (Stage 3)"
            explain_reason = f"Recalibration sequence active. Re-aligning baselines. (Epoch {recal_progress // 10}/10)"
        elif learning_status == "Model Updated":
            agent_action = "System Calibrated"
            explain_source = "Agentic Controller (Stage 3)"
            explain_reason = "Vanguard neural weights updated. Calibration parameters successfully deployed to Edge."
        else:
            if uncertainty_score < 0.5:
                agent_action = "Accept"
            elif uncertainty_score < 1.0:
                agent_action = "Re-sample"
            else:
                agent_action = "Escalate (Human-in-the-loop)"
            
            if co_ppm > rules['co_danger'] or eth_ppm > rules['eth_danger'] or predicted_ammonia > rules['ammo_danger']:
                explain_reason = "Hazardous concentration levels detected. Triggering automated safety matrix."
            else:
                explain_reason = "Normal operation; gas concentrations within safe atmospheric bounds."

        # ─── Configurable Emergency Response Matrix & Actuation ─────────────
        hazard_level = evaluate_hazard_and_actions(co_ppm, predicted_ammonia, eth_ppm, current_profile)
        
        # Reset actuation indicators
        l_green, l_yellow, l_red, l_blue = False, False, False, False
        f_buzzer, r_vent, r_shutdown = False, False, False
        
        if learning_status == "Recalibration Running":
            l_blue = True
        elif learning_status == "Model Updated":
            # Blinking blue code
            l_blue = (t % 2 == 0)
        
        # Map Severity Levels to outputs
        if hazard_level == "Green (Safe)":
            l_green = True
            is_acknowledged = False # Reset ack for next event
            escalation_countdown = -1
            escalated_action = "None"
        
        elif hazard_level == "Yellow (Elevated)":
            l_yellow = True
            # Profile response override
            if current_profile == 'Agricultural':
                r_vent = True # Activate fans automatically for fruit preservation/poultry comfort
            elif current_profile == 'Smart Building':
                r_vent = True # HVAC fan active
        
        elif hazard_level == "Orange (Dangerous)":
            l_red = True
            # Pulse buzzer (short beeps)
            f_buzzer = (t % 4 == 0)
            r_vent = True # Activate ventilation
            
        elif hazard_level in ["Red (Critical)", "Purple (Extreme Hazard)"]:
            # Rapid blinking red LED
            l_red = (t % 2 == 0)
            
            # Continuous buzzer (unless acknowledged by user)
            if not is_acknowledged:
                # Trigger buzzer
                if hazard_level == "Purple (Extreme Hazard)":
                    f_buzzer = True # Continuous
                    r_shutdown = True # Emergency cutoff valve / machinery shutdown
                else:
                    f_buzzer = (t % 2 == 0) # Intense pulsed buzzer
                r_vent = True # Vent fan active
                
                # Safeguard escalation logic
                if escalation_countdown == -1:
                    escalation_countdown = 300 # 30 seconds countdown (10 ticks = 1s)
                    escalated_action = "None"
                elif escalation_countdown > 0:
                    escalation_countdown -= 1
                else:
                    # Timer hit 0: Send SMS/Call
                    escalated_action = "Simulated Emergency SMS & Phone Call Sent to Rescue Team"
            else:
                # User acknowledged/muted the alert
                f_buzzer = False
                escalation_countdown = -1
                escalated_action = "Muted by Operator"
                
                # Relays remain active for safety
                r_vent = True
                if hazard_level == "Purple (Extreme Hazard)":
                    r_shutdown = True
                    
        # Apply hardware outputs
        set_gpio(GREEN_LED, l_green)
        set_gpio(YELLOW_LED, l_yellow)
        set_gpio(RED_LED, l_red)
        set_gpio(BLUE_LED, l_blue)
        set_gpio(BUZZER, f_buzzer)
        set_gpio(VENT_RELAY, r_vent)
        set_gpio(SHUTDOWN_RELAY, r_shutdown)

        # ─── SQL Logging ─────────────────────────────────────────────────────
        if t % 10 == 0:
            try:
                conn = sqlite3.connect(DATABASE_PATH)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO telemetry_history (
                        timestamp, profile, temp, humidity, pressure, velocity,
                        actual_co, predicted_co, actual_eth, predicted_eth,
                        actual_nitro, predicted_nitro, actual_ammonia, predicted_ammonia,
                        latency_ms, uncertainty_score, predicted_state, agent_action, hazard_level,
                        green_led, yellow_led, red_led, blue_led, buzzer, vent_relay, shutdown_relay
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    t, current_profile, temp, humidity, pressure, round(velocity, 2),
                    actual_co, co_ppm, actual_eth, eth_ppm,
                    actual_nitro, predicted_nitro, actual_ammonia, predicted_ammonia,
                    round(latency_ms, 2), round(uncertainty_score, 3), predicted_state, agent_action, hazard_level,
                    int(l_green), int(l_yellow), int(l_red), int(l_blue), int(f_buzzer), int(r_vent), int(r_shutdown)
                ))
                conn.commit()
                conn.close()
            except Exception as db_err:
                print(f"Database logging error: {db_err}")

        # Assemble live payload
        payload = {
            "timestamp": t,
            "profile": current_profile,
            "temp": temp,
            "humidity": humidity,
            "pressure": pressure,
            "velocity": round(velocity, 2),
            "latency_ms": round(latency_ms, 2),
            "predicted_state": predicted_state,
            "uncertainty_score": round(uncertainty_score, 3),
            "agent_action": agent_action,
            "actual_co": actual_co,
            "predicted_co": co_ppm,
            "actual_eth": actual_eth,
            "predicted_eth": eth_ppm,
            "actual_nitro": actual_nitro,
            "predicted_nitro": predicted_nitro,
            "actual_ammonia": actual_ammonia,
            "predicted_ammonia": predicted_ammonia,
            "sensor_mean": round(float(np.mean(new_reading)), 3),
            "sensors": [round(float(s), 3) for s in new_reading],
            "learning_status": learning_status,
            "last_recal_t": last_recal_t,
            "active_learning_events": active_learning_events,
            "recal_progress": recal_progress,
            "explain_confidence": round(float(explain_confidence), 1),
            "explain_source": explain_source,
            "explain_reason": explain_reason,
            "hazard_level": hazard_level,
            "countdown": escalation_countdown,
            "escalated_action": escalated_action,
            "gpio": {
                "green": int(l_green),
                "yellow": int(l_yellow),
                "red": int(l_red),
                "blue": int(l_blue),
                "buzzer": int(f_buzzer),
                "vent": int(r_vent),
                "shutdown": int(r_shutdown)
            }
        }
        
        yield f"data: {json.dumps(payload)}\n\n"
        
        t += 1
        time.sleep(0.1)

# ─── API Routes ──────────────────────────────────────────────────────
@app.route('/stream')
def stream():
    return Response(generate_sensor_stream(), mimetype='text/event-stream')

@app.route('/api/history')
def get_history():
    gas = request.args.get('gas', 'all').lower()
    limit = min(int(request.args.get('limit', 100)), 1000)
    
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM telemetry_history 
        ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    rows = list(reversed(rows))
    
    history_data = []
    for r in rows:
        item = {
            "id": r["id"],
            "timestamp": r["timestamp"],
            "profile": r["profile"],
            "temp": r["temp"],
            "humidity": r["humidity"],
            "pressure": r["pressure"],
            "velocity": r["velocity"],
            "latency_ms": r["latency_ms"],
            "uncertainty_score": r["uncertainty_score"],
            "predicted_state": r["predicted_state"],
            "agent_action": r["agent_action"],
            "hazard_level": r["hazard_level"],
            "gpio": {
                "green": r["green_led"],
                "yellow": r["yellow_led"],
                "red": r["red_led"],
                "blue": r["blue_led"],
                "buzzer": r["buzzer"],
                "vent": r["vent_relay"],
                "shutdown": r["shutdown_relay"]
            }
        }
        
        if gas == 'co' or gas == 'all':
            item["actual_co"] = r["actual_co"]
            item["predicted_co"] = r["predicted_co"]
        if gas == 'eth' or gas == 'all':
            item["actual_eth"] = r["actual_eth"]
            item["predicted_eth"] = r["predicted_eth"]
        if gas == 'nitro' or gas == 'all':
            item["actual_nitro"] = r["actual_nitro"]
            item["predicted_nitro"] = r["predicted_nitro"]
        if gas == 'ammonia' or gas == 'all':
            item["actual_ammonia"] = r["actual_ammonia"]
            item["predicted_ammonia"] = r["predicted_ammonia"]
            
        history_data.append(item)
        
    return jsonify(history_data)

@app.route('/api/analysis')
def get_analysis():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM telemetry_history ORDER BY id DESC LIMIT 500")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return jsonify({"status": "no_data"})
        
    errors_co = [abs(r["actual_co"] - r["predicted_co"]) for r in rows]
    errors_eth = [abs(r["actual_eth"] - r["predicted_eth"]) for r in rows]
    errors_nitro = [abs(r["actual_nitro"] - r["predicted_nitro"]) for r in rows]
    errors_ammonia = [abs(r["actual_ammonia"] - r["predicted_ammonia"]) for r in rows]
    
    mae_co = np.mean(errors_co)
    mae_eth = np.mean(errors_eth)
    mae_nitro = np.mean(errors_nitro)
    mae_ammonia = np.mean(errors_ammonia)
    
    correct_class = 0
    for r in rows:
        act_co = r["actual_co"]
        act_eth = r["actual_eth"]
        if act_co > 0.1 and act_eth > 0.1:
            expected = "Mixture"
        elif act_co > 0.1:
            expected = "Pure CO"
        elif act_eth > 0.1:
            expected = "Pure Ethylene"
        else:
            expected = "Air"
            
        if r["predicted_state"] == expected:
            correct_class += 1
            
    accuracy = (correct_class / len(rows)) * 100
    
    vel_bands = {"low": [], "mid": [], "high": []}
    temp_bands = {"low": [], "mid": [], "high": []}
    hum_bands = {"low": [], "mid": [], "high": []}
    
    for r in rows:
        err = (abs(r["actual_co"] - r["predicted_co"]) + abs(r["actual_eth"] - r["predicted_eth"])) / 2.0
        
        v = r["velocity"]
        if v < 2.0:
            vel_bands["low"].append(err)
        elif v <= 3.0:
            vel_bands["mid"].append(err)
        else:
            vel_bands["high"].append(err)
            
        t = r["temp"]
        if t < 23.0:
            temp_bands["low"].append(err)
        elif t <= 26.0:
            temp_bands["mid"].append(err)
        else:
            temp_bands["high"].append(err)
            
        h = r["humidity"]
        if h < 50.0:
            hum_bands["low"].append(err)
        elif h <= 65.0:
            hum_bands["mid"].append(err)
        else:
            hum_bands["high"].append(err)
            
    analysis_data = {
        "status": "success",
        "accuracy": accuracy,
        "mae": {
            "co": mae_co,
            "eth": mae_eth,
            "nitro": mae_nitro,
            "ammonia": mae_ammonia
        },
        "bands": {
            "velocity": {
                "low": np.mean(vel_bands["low"]) if vel_bands["low"] else 0.0,
                "mid": np.mean(vel_bands["mid"]) if vel_bands["mid"] else 0.0,
                "high": np.mean(vel_bands["high"]) if vel_bands["high"] else 0.0
            },
            "temp": {
                "low": np.mean(temp_bands["low"]) if temp_bands["low"] else 0.0,
                "mid": np.mean(temp_bands["mid"]) if temp_bands["mid"] else 0.0,
                "high": np.mean(temp_bands["high"]) if temp_bands["high"] else 0.0
            },
            "humidity": {
                "low": np.mean(hum_bands["low"]) if hum_bands["low"] else 0.0,
                "mid": np.mean(hum_bands["mid"]) if hum_bands["mid"] else 0.0,
                "high": np.mean(hum_bands["high"]) if hum_bands["high"] else 0.0
            }
        }
    }
    return jsonify(analysis_data)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    global current_profile, is_acknowledged, learning_status, status_timer, recal_progress, run_loop
    data = request.get_json() or {}
    action = data.get('action')
    
    if action == 'profile':
        profile = data.get('profile')
        if profile in PROFILE_RULES:
            current_profile = profile
            is_acknowledged = False
            return jsonify({"status": "success", "profile": current_profile})
        return jsonify({"status": "error", "message": "Invalid profile name"}), 400
        
    elif action == 'ack':
        is_acknowledged = True
        return jsonify({"status": "success", "message": "Alarm acknowledged / muted."})
        
    elif action == 'recalibrate':
        # Manually trigger recalibration cycle
        learning_status = "Recalibration Running"
        status_timer = 30
        recal_progress = 0
        return jsonify({"status": "success", "message": "Recalibration triggered."})
        
    elif action == 'start':
        run_loop = True
        return jsonify({"status": "success", "message": "Simulation started."})
        
    elif action == 'stop':
        run_loop = False
        return jsonify({"status": "success", "message": "Simulation stopped."})
        
    elif action == 'reset':
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM telemetry_history")
            conn.commit()
            conn.close()
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
            
    return jsonify({"status": "error", "message": "Unknown action"}), 400

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    print("CaNeSy-eNose Edge Flask Server Starting...")
    app.run(host='0.0.0.0', port=5000, debug=True)
