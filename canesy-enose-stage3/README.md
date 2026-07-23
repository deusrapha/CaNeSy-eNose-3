# CaNeSy-eNose: Standalone Edge AI Deployment Node

**CaNeSy-eNose** is an autonomous edge-intelligent environmental safety platform capable of real-time gas perception, explainable temporal reasoning, adaptive learning, and configurable emergency response for residential, industrial, agricultural, and smart-building environments.

This directory contains the production-ready Edge AI Deployment Package configured for a **Raspberry Pi 4 (4GB RAM)**. It loads a pre-trained multi-task learning (MTL) Vanguard Transformer model to predict gas classification state and regress concentration levels, while executing hardware control overrides through its local GPIO pins.

---

## 1. Hardware Architecture & Sensing Stack

The system supports a rich environmental sensing array to correct for wind dilution and capture diverse trace gases:

| Sensor Module | Monitored Parameter | Target Interface | Academic Role in Olfaction |
| :--- | :--- | :--- | :--- |
| **MQ-7** | Carbon Monoxide (CO) | Analog (via ADS1115 ADC) | Primary toxic combustion gas |
| **MQ-137** | Ammonia ($NH_3$) | Analog (via ADS1115 ADC) | Primary agricultural decay gas |
| **BME688** | VOCs / Ethylene trends | I2C (Address `0x76` or `0x77`) | AI-enabled gas sensor for organic trends |
| **BME280** | Temperature, Humidity, Pressure | I2C (Address `0x76`) | Reference parameters for relative humidity drift |
| **Anemometer** | Wind Speed (Velocity) | Pulse Counter / Analog Input | High-frequency physical wind vector correction |
| **Nitrogen ($N_2$)** | Background Atmosphere | Reference Gas (Software-Modelled) | Stable baseline reference (78.08% of air) |

### System Hardware Connection Diagram
```
                     Wind Sensor (Anemometer)
                                │
   Temp/Humidity/Pressure       │ (Analog / Pulse Count)
                                │
            BME280              │
               │                │
               ▼                ▼
         ┌────────────────────────────┐
         │       Raspberry Pi 4       │
         │  (Vanguard Edge AI Node)   │
         └──────────────┬─────────────┘
                        │ (GPIO Pins)
         ┌──────────────┴─────────────┐
         │                            │
      LED Panel                 Buzzer & Relays
   (Normal / Warn /              (Audible Siren,
    Toxic / Learn)              Exhaust, Valve)
```

---

## 2. GPIO Wiring & Pinout Mapping

Actuator controls are connected to the Raspberry Pi General Purpose Input/Output (GPIO) pins using standard BCM numbering:

| Actuator Device | GPIO BCM Pin | Pi Physical Pin | Signal Action | Active State |
| :--- | :---: | :---: | :--- | :---: |
| **Green LED** | `GPIO 18` | Pin 12 | Normal System Operation | HIGH |
| **Yellow LED** | `GPIO 23` | Pin 16 | Elevated Gas Levels (Warning) | HIGH |
| **Red LED** | `GPIO 24` | Pin 18 | Critical Toxic Alert (Danger) | Pulsing (0.5s) |
| **Blue LED** | `GPIO 25` | Pin 22 | Model Calibration Active (Learning) | Pulsing (0.8s) |
| **Piezo Buzzer** | `GPIO 12` | Pin 32 | Audible Alarm Output | Pulse/Continuous |
| **Exhaust Fan Relay** | `GPIO 16` | Pin 36 | Relay 1: Turn on Ventilation | HIGH |
| **Shutoff Valve Relay** | `GPIO 20` | Pin 38 | Relay 2: Trigger Fuel/Gas Line Cutoff | HIGH |

---

## 3. Configurable Environmental Safety Profiles

To maximize flexibility, CaNeSy-eNose implements four deployment profiles:

### 🏡 Home Safety Mode
Designed for residential environments. Tracks CO (gas leakages, combustion) and Ammonia (cleaning agents).
- **Warning:** Yellow LED, dashboard notification.
- **Danger:** Red LED, Piezo buzzer warning beeps.
- **Critical/Extreme:** Red LED flashing, continuous alarm, emergency SMS notification, recommand evacuation.

### 🏭 Industrial Safety Mode
Optimized for factory floors, labs, and warehouses. Applies strict threshold parameters with high wind-speed aerodynamic correction.
- **Stage 1 (Warning):** Yellow LED, supervisor notified.
- **Stage 2 (Danger):** Red LED, audible alarms active.
- **Stage 3 (Critical/Extreme):** Flashing red, continuous buzzer, shut down machinery, close gas lines, notify control center.

### 🌾 Agricultural Mode
Designed for crop storage (Ethylene trends for ripening control) and poultry houses (Ammonia ventilation control).
- **Actions:** Dynamically turns on Ventilation Fans (Relay 1) to regulate poultry ambient quality or vents ripening rooms, alerts farm manager if thresholds exceed 90% confidence.

### 🏢 Smart Building Mode
Maintains office ventilation and building HVAC efficiency. Tracks mixed VOCs and CO accumulation.
- **Actions:** Turns on maximum building HVAC ventilation (Relay 1), signals building controls, displays exit paths if hazard is critical.

---

## 4. Autonomous Severity Response Matrix

The Edge decision engine classifies risks into five tiers and actuates hardware:

| Hazard Level | LED Pin Active | Buzzer State | Relay 1 (Exhaust) | Relay 2 (Shutdown) | Remote Notification |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Green (Safe)** | Green | OFF | OFF | OFF | None |
| **Yellow (Elevated)** | Yellow | OFF | Optional | OFF | None |
| **Orange (Dangerous)** | Red | Beeping (Slow) | ON | OFF | SMS Alert |
| **Red (Critical)** | Flashing Red | Beeping (Fast) | ON | OFF | SMS + Call (30s) |
| **Purple (Extreme)** | Flashing Red | Continuous | ON | ON | SMS + Call (Immediate) |

### Safety Countdown Safeguard
To prevent false alarms in safety-critical actions, if Hazard Level is **Red (Critical)** or **Purple (Extreme Hazard)**:
1. Activate visual alarm indicators and buzzer immediately.
2. Start a **30-second countdown** on the dashboard.
3. If no operator acknowledges/mutes the alarm, automatically initiate the remote emergency SMS and dial out.

---

## 5. Software Installation & Startup Guide

Follow these steps to run the CaNeSy-eNose server on the Raspberry Pi:

### Step 1: Clone and Navigate
Ensure the code directory is placed on your Raspberry Pi storage.
```bash
cd DEPLOY
```

### Step 2: Establish Virtual Environment
Create a clean environment to isolate edge package dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Core Dependencies
Install packages listed in the requirements directory, and install RPi.GPIO:
```bash
pip install -r requirements.txt
pip install RPi.GPIO
```

### Step 4: Run Flask Server
Execute the server script:
```bash
python3 app.py
```
The console will host the dashboard at `http://localhost:5000` or `http://<your-pi-ip>:5000` for LAN access.

---

## 6. Scientific Justification: Local Edge vs. Cloud Inference

If asked during viva or paper defense, the scientific selection of edge-only deployment centers on:
1. **Low-Latency Safety Realism:** Gas leaks spread within seconds. Round-trip cloud network times (50-200ms) introduce critical delay. Edge inference latency (under 1ms on Pi) ensures sub-second emergency response.
2. **Connectivity Independence:** Safety platforms must function during power grid failures or loss of Internet. Local SQLite databases and local ONNX runtime guarantee continuous monitoring without cellular or Wi-Fi signals.
3. **Bandwidth Efficiency:** Streaming 16 channels at 10Hz to a remote server exhausts LAN traffic and introduces network jitter. Local processing keeps transmission local and keeps bandwidth requirements near zero.
