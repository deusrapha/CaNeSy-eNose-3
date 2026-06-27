import time
import json
import sqlite3
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, jsonify, request
from scipy.stats import entropy

app = Flask(__name__, static_folder='static')

# Load the ONNX Model once
ONNX_PATH = "mtl_velocity_model.onnx"
print(f"Loading {ONNX_PATH}...")
try:
    session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    session = None

CLASSES = ['Air', 'Pure CO', 'Pure Ethylene', 'Mixture']
DATABASE_PATH = "history.db"

def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            temp REAL,
            humidity REAL,
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
            agent_action TEXT
        )
    """)
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

def generate_sensor_stream():
    """Simulates real-time sensor polling and runs inference at a defined rate."""
    # Simulation state
    t = 0
    buffer = np.zeros((1, 50, 16), dtype=np.float32)
    velocity = 2.5 # initial wind velocity m/s
    
    # State variables for actual concentrations
    actual_co = 0.0
    actual_eth = 0.0
    actual_nitro = 52.0
    actual_ammonia = 0.0
    
    # Active learning state variables
    learning_status = "Normal Operation"
    last_recal_t = -3000  # simulated as 5 minutes ago initially
    active_learning_events = 0
    recal_progress = 0
    status_timer = 0
    
    while True:
        is_event = (t % 200) > 100
        
        base_signal = np.sin(t * 0.1) * 0.5 + 0.5
        noise = np.random.randn(16) * 0.1
        
        if is_event:
            new_reading = (base_signal + 5.0 + noise).astype(np.float32)
            velocity = max(0.5, velocity + np.random.randn() * 0.1)
        else:
            new_reading = (noise).astype(np.float32)
            velocity = 2.5 + np.sin(t * 0.05) * 0.5
            
        buffer[0, :-1, :] = buffer[0, 1:, :]
        buffer[0, -1, :] = new_reading
        
        # Environmental variables
        temp = round(24.5 + 3.0 * np.sin(t * 0.02) + np.random.randn() * 0.1, 1)
        humidity = round(58.2 + 8.0 * np.cos(t * 0.015) + np.random.randn() * 0.2, 1)
        
        # Ground truth simulation matching model's scale (0.0 to 1.5 ppm)
        if is_event:
            target_co = 0.65 + np.sin(t * 0.05) * 0.15
            target_eth = 1.1 + np.cos(t * 0.07) * 0.2
            target_ammonia = 0.4 + np.sin(t * 0.03) * 0.08
        else:
            target_co = 0.0
            target_eth = 0.0
            target_ammonia = 0.0
            
        actual_co = round(actual_co * 0.9 + target_co * 0.1, 3)
        actual_eth = round(actual_eth * 0.9 + target_eth * 0.1, 3)
        actual_ammonia = round(actual_ammonia * 0.95 + target_ammonia * 0.05, 3)
        actual_nitro = round(50.0 + 5.0 * np.sin(t * 0.02) + np.random.randn() * 0.3, 2)
        
        # Model predictions
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
            
            uncertainty_score = entropy(probs)
            explain_confidence = float(np.max(probs)) * 100.0
            
            if uncertainty_score < 0.5:
                agent_action = "Accept"
            elif uncertainty_score < 1.0:
                agent_action = "Re-sample"
            else:
                agent_action = "Escalate (Human-in-the-loop)"
            
            co_ppm = float(regression_ppm[0][0])
            eth_ppm = float(regression_ppm[0][1])
        else:
            explain_confidence = 92.5 + np.sin(t * 0.05) * 2.5 + np.random.randn() * 0.5
            
        co_ppm = max(0.0, round(co_ppm, 3))
        eth_ppm = max(0.0, round(eth_ppm, 3))
        
        # State machine logic for active learning & recalibration
        if learning_status == "Normal Operation":
            # Trigger drift detection at a specific point in the toxic event cycle
            if t > 0 and t % 200 == 120:
                learning_status = "Model Drift Detected"
                status_timer = 40  # 4 seconds
        
        elif learning_status == "Model Drift Detected":
            status_timer -= 1
            if status_timer <= 0:
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

        # Apply state overlays on Agentic Control and Explainability Source/Reason
        explain_source = "Vanguard Edge Model (Stage 1)"
        if uncertainty_score >= 0.5 and uncertainty_score < 1.0:
            explain_source = "Temporal Reasoning Layer (Stage 2)"
        elif uncertainty_score >= 1.0:
            explain_source = "Agentic Controller (Stage 3)"
            
        if actual_co > 0.5 and actual_eth > 0.5:
            explain_reason = "Elevated gas response observed across multiple channels."
        elif actual_co > 0.5:
            explain_reason = "Elevated CO response observed across multiple channels."
        elif actual_eth > 0.5:
            explain_reason = "Ethylene concentration rising under favorable wind conditions."
        else:
            explain_reason = "Normal operation; trace levels within acceptable safety margins"

        if learning_status == "Model Drift Detected":
            agent_action = "Escalate (Awaiting Human Oracle)"
            explain_source = "Agentic Controller (Stage 3)"
            explain_reason = "Sensor drift detected; human validation required."
        elif learning_status == "Recalibration Running":
            agent_action = "Recalibrating Model..."
            explain_source = "Agentic Controller (Stage 3)"
            explain_reason = f"Uncertainty threshold exceeded; initiating recalibration. (Epoch {recal_progress // 10}/10)"
        elif learning_status == "Model Updated":
            agent_action = "Model Updated & Re-deployed"
            explain_source = "Agentic Controller (Stage 3)"
            explain_reason = "Baseline calibration optimized for dynamic environmental conditions."

        # Predictions for background/trace helper gases
        temp_drift = (temp - 24.5) * 0.02
        hum_drift = (humidity - 58.2) * 0.005
        vel_drift = (velocity - 2.5) * 0.03
        
        predicted_nitro = round(max(0.0, actual_nitro + np.random.randn() * 0.4 + temp_drift * 10 + vel_drift * 4), 2)
        predicted_ammonia = round(max(0.0, actual_ammonia + np.random.randn() * 0.04 + hum_drift * 2 + vel_drift), 3)
        
        payload = {
            "timestamp": t,
            "temp": temp,
            "humidity": humidity,
            "velocity": round(velocity, 2),
            "latency_ms": round(latency_ms, 2),
            "predicted_state": predicted_state,
            "uncertainty_score": round(float(uncertainty_score), 3),
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
            "explain_reason": explain_reason
        }
        
        # Save to SQLite once every 10 ticks (~1s)
        if t % 10 == 0:
            try:
                conn = sqlite3.connect(DATABASE_PATH)
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO telemetry_history (
                        timestamp, temp, humidity, velocity,
                        actual_co, predicted_co, actual_eth, predicted_eth,
                        actual_nitro, predicted_nitro, actual_ammonia, predicted_ammonia,
                        latency_ms, uncertainty_score, predicted_state, agent_action
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    t, temp, humidity, round(velocity, 2),
                    actual_co, co_ppm, actual_eth, eth_ppm,
                    actual_nitro, predicted_nitro, actual_ammonia, predicted_ammonia,
                    round(latency_ms, 2), round(float(uncertainty_score), 3),
                    predicted_state, agent_action
                ))
                conn.commit()
                conn.close()
            except Exception as db_err:
                print(f"Database logging error: {db_err}")
                
        yield f"data: {json.dumps(payload)}\n\n"
        
        t += 1
        time.sleep(0.1)

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
            "temp": r["temp"],
            "humidity": r["humidity"],
            "velocity": r["velocity"],
            "latency_ms": r["latency_ms"],
            "uncertainty_score": r["uncertainty_score"],
            "predicted_state": r["predicted_state"],
            "agent_action": r["agent_action"],
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
        "total_records": len(rows),
        "mae": {
            "co": round(float(mae_co), 4),
            "eth": round(float(mae_eth), 4),
            "nitro": round(float(mae_nitro), 4),
            "ammonia": round(float(mae_ammonia), 4)
        },
        "accuracy": round(float(accuracy), 2),
        "bands": {
            "velocity": {
                "low": round(float(np.mean(vel_bands["low"])), 4) if vel_bands["low"] else 0,
                "mid": round(float(np.mean(vel_bands["mid"])), 4) if vel_bands["mid"] else 0,
                "high": round(float(np.mean(vel_bands["high"])), 4) if vel_bands["high"] else 0,
                "counts": {"low": len(vel_bands["low"]), "mid": len(vel_bands["mid"]), "high": len(vel_bands["high"])}
            },
            "temp": {
                "low": round(float(np.mean(temp_bands["low"])), 4) if temp_bands["low"] else 0,
                "mid": round(float(np.mean(temp_bands["mid"])), 4) if temp_bands["mid"] else 0,
                "high": round(float(np.mean(temp_bands["high"])), 4) if temp_bands["high"] else 0,
                "counts": {"low": len(temp_bands["low"]), "mid": len(temp_bands["mid"]), "high": len(temp_bands["high"])}
            },
            "humidity": {
                "low": round(float(np.mean(hum_bands["low"])), 4) if hum_bands["low"] else 0,
                "mid": round(float(np.mean(hum_bands["mid"])), 4) if hum_bands["mid"] else 0,
                "high": round(float(np.mean(hum_bands["high"])), 4) if hum_bands["high"] else 0,
                "counts": {"low": len(hum_bands["low"]), "mid": len(hum_bands["mid"]), "high": len(hum_bands["high"])}
            }
        }
    }
    
    return jsonify(analysis_data)

@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.get_json() or {}
    action = data.get('action')
    
    if action == 'reset':
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS telemetry_history")
            conn.commit()
            conn.close()
            init_db()
            return jsonify({"status": "success", "message": "Database reset completed."})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
            
    return jsonify({"status": "error", "message": "Unknown action."}), 400

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    print("Starting PC Simulation Server on http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, threaded=True)
