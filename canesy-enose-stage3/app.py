import streamlit as st
import time
import numpy as np
import onnxruntime as ort
import pandas as pd
from scipy.stats import entropy
from pathlib import Path
import sqlite3
import plotly.graph_objects as go
import os
import textwrap

# Streamlit App Configuration
st.set_page_config(page_title="CaNeSy-eNose Dashboard", layout="wide", initial_sidebar_state="expanded")

# Inject Custom CSS for Premium Dark Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

/* Global Styles */
.stApp {
    background-color: #060913 !important;
    color: #f8fafc !important;
    font-family: 'Inter', sans-serif !important;
}

div[data-testid="stHeader"] {
    background-color: rgba(6, 9, 19, 0.8) !important;
}

/* Tab Styling */
div[data-baseweb="tab-list"] {
    background-color: #0b0f19;
    border-radius: 0.5rem;
    padding: 0.25rem;
    border-bottom: none;
    margin-bottom: 1.5rem;
}
button[data-baseweb="tab"] {
    color: #64748b !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.25rem !important;
    border-bottom: none !important;
    background-color: transparent !important;
    transition: all 0.2s ease !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8 !important;
    background-color: rgba(56, 189, 248, 0.08) !important;
    border-radius: 0.375rem !important;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.08) !important;
}

/* Warnings and Alerts */
.toxic-alert {
    background: linear-gradient(135deg, rgba(244, 63, 94, 0.15), rgba(244, 63, 94, 0.03));
    border: 1px solid #f43f5e;
    color: #f8fafc;
    padding: 1.25rem 1.5rem;
    border-radius: 0.75rem;
    margin-bottom: 1.5rem;
    font-weight: 600;
    animation: pulse-danger-border 1.5s infinite ease-in-out;
}
@keyframes pulse-danger-border {
    0% { border-color: #f43f5e; box-shadow: 0 0 5px rgba(244, 63, 94, 0.2); }
    50% { border-color: rgba(244, 63, 94, 0.4); box-shadow: 0 0 15px rgba(244, 63, 94, 0.5); }
    100% { border-color: #f43f5e; box-shadow: 0 0 5px rgba(244, 63, 94, 0.2); }
}

/* KPI Dashboard Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
    width: 100%;
}
@media (max-width: 1200px) {
    .kpi-grid { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 768px) {
    .kpi-grid { grid-template-columns: 1fr; }
}
.kpi-card {
    background-color: rgba(17, 24, 39, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 0.85rem;
    padding: 1.15rem;
    position: relative;
    overflow: hidden;
    min-height: 100px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    background-color: rgba(31, 41, 55, 0.7);
    border-color: rgba(255, 255, 255, 0.13);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
}
.kpi-card.temp::before { background-color: #f97316; }
.kpi-card.humidity::before { background-color: #a855f7; }
.kpi-card.wind::before { background-color: #38bdf8; }
.kpi-card.latency::before { background-color: #eab308; }
.kpi-card.uncertainty::before { background-color: #f43f5e; }
.kpi-card.class::before { background-color: #10b981; }

.kpi-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #64748b;
}
.kpi-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    line-height: 1.2;
    margin-top: 0.5rem;
}
.kpi-suffix {
    font-size: 0.85rem;
    color: #64748b;
    font-weight: 400;
    margin-left: 2px;
}

/* Glass Card containers */
.glass-card {
    background-color: rgba(17, 24, 39, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 1rem;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(12px);
}
.card-header {
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 0.75rem;
    margin-bottom: 1.25rem;
}
.card-header h2 {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 1.15rem;
    margin: 0;
}

/* Heatmap widget */
.heatmap-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.6rem;
    margin-top: 1rem;
}
.heatmap-cell {
    aspect-ratio: 1.3;
    background-color: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 0.5rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    color: rgba(255, 255, 255, 0.35);
    transition: all 0.2s ease;
}
.heatmap-cell span {
    font-size: 0.65rem;
    font-weight: 400;
    color: rgba(255, 255, 255, 0.7);
    margin-top: 2px;
}

/* Dynamic Status Badges */
.badge-status {
    padding: 0.4rem 0.85rem;
    border-radius: 2rem;
    font-size: 0.85rem;
    font-weight: 600;
    text-align: center;
    border: 1px solid transparent;
    display: inline-block;
}
.badge-status.accept {
    background-color: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border-color: rgba(16, 185, 129, 0.3);
}
.badge-status.resample {
    background-color: rgba(234, 179, 8, 0.15);
    color: #eab308;
    border-color: rgba(234, 179, 8, 0.3);
}
.badge-status.escalate {
    background-color: rgba(244, 63, 94, 0.15);
    color: #f43f5e;
    border-color: rgba(244, 63, 94, 0.3);
}
.badge-status.drift {
    background-color: rgba(249, 115, 22, 0.15);
    color: #f97316;
    border-color: rgba(249, 115, 22, 0.3);
    animation: blink-drift 1s infinite alternate;
}
.badge-status.calibrating {
    background-color: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border-color: rgba(56, 189, 248, 0.3);
}
.badge-status.updated {
    background-color: rgba(168, 85, 247, 0.15);
    color: #a855f7;
    border-color: rgba(168, 85, 247, 0.3);
}
@keyframes blink-drift {
    from { opacity: 0.75; }
    to { opacity: 1; border-color: rgba(249, 115, 22, 0.6); }
}

/* Analysis breakdown chart elements */
.mae-badge-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}
.mae-badge-card {
    background-color: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 0.75rem;
    padding: 1rem;
    display: flex;
    flex-direction: column;
}
.mae-badge-name {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 600;
}
.mae-badge-val {
    font-family: 'Outfit', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
}

.bar-chart-container {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    margin-top: 0.75rem;
}
.bar-row {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
}
.bar-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    font-weight: 600;
}
.bar-bg {
    height: 8px;
    background-color: rgba(255, 255, 255, 0.03);
    border-radius: 4px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# Initialize Paths
BASE_DIR = Path(__file__).resolve().parent
ONNX_PATH = BASE_DIR / "mtl_velocity_model.onnx"
DATABASE_PATH = BASE_DIR / "history.db"

# Initialize ONNX Model
@st.cache_resource
def load_model():
    try:
        return ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    except Exception as e:
        st.error(f"Error loading ONNX model. Please ensure '{ONNX_PATH}' is in the repository. Details: {e}")
        return None

session = load_model()
CLASSES = ['Air', 'Pure CO', 'Pure Ethylene', 'Mixture']

# SQLite Telemetry Database Initialization
def init_db():
    conn = sqlite3.connect(str(DATABASE_PATH))
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

init_db()

# State variables initialization in Session State
if "t" not in st.session_state:
    st.session_state.t = 0
    st.session_state.buffer = np.zeros((1, 50, 16), dtype=np.float32)
    st.session_state.velocity = 2.5
    st.session_state.actual_co = 0.0
    st.session_state.actual_eth = 0.0
    st.session_state.actual_nitro = 52.0
    st.session_state.actual_ammonia = 0.0
    
    st.session_state.learning_status = "Normal Operation"
    st.session_state.last_recal_t = -3000
    st.session_state.active_learning_events = 0
    st.session_state.recal_progress = 0
    st.session_state.status_timer = 0
    st.session_state.run_simulation = False
    
    # History queues for live plotting (Max size = 60)
    st.session_state.hist_ticks = [0] * 60
    st.session_state.hist_co_act = [0.0] * 60
    st.session_state.hist_co_pred = [0.0] * 60
    st.session_state.hist_eth_act = [0.0] * 60
    st.session_state.hist_eth_pred = [0.0] * 60
    st.session_state.hist_nitro_act = [50.0] * 60
    st.session_state.hist_nitro_pred = [50.0] * 60
    st.session_state.hist_ammo_act = [0.0] * 60
    st.session_state.hist_ammo_pred = [0.0] * 60

# Sidebar Navigation Panel
with st.sidebar:
    st.markdown('<h2 style="font-family:\'Outfit\'; background:linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">CaNeSy-eNose</h2>', unsafe_allow_html=True)
    st.markdown("---")
    
    if st.button("Start/Stop Simulation Override", use_container_width=True):
        st.session_state.run_simulation = not st.session_state.run_simulation
        
    st.markdown("### Operational Controls")
    status_indicator = "🟢 Running" if st.session_state.run_simulation else "🔴 Paused"
    st.markdown(f"**Status:** {status_indicator}")
    st.markdown(f"**Calibration Events:** {st.session_state.active_learning_events}")
    st.markdown("---")
    st.markdown("**Version Info:** Edge Node 1.0.4")
    st.markdown("**Core Runtime:** ONNX CPU Engine")

# Main Title and Tab Split
st.markdown('<h1 style="font-family:\'Outfit\'; margin-bottom: 5px;">🧠 CaNeSy-eNose System Cockpit</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#64748b; font-size:0.9rem; margin-top:0px; margin-bottom:1.5rem;">Unified multi-stage robotic olfaction diagnostic control cockpit</p>', unsafe_allow_html=True)

# ----------------- SIMULATION STEP CALCULATIONS -----------------
# We run one simulation step per rerun if simulation is active
if st.session_state.run_simulation:
    t = st.session_state.t
    buffer = st.session_state.buffer
    velocity = st.session_state.velocity
    actual_co = st.session_state.actual_co
    actual_eth = st.session_state.actual_eth
    actual_nitro = st.session_state.actual_nitro
    actual_ammonia = st.session_state.actual_ammonia
    
    learning_status = st.session_state.learning_status
    status_timer = st.session_state.status_timer
    recal_progress = st.session_state.recal_progress
    active_learning_events = st.session_state.active_learning_events
    last_recal_t = st.session_state.last_recal_t

    # Event cycle logic (200 ticks per cycle)
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
    
    # Gas target concentrations (scale 0.0 to 1.5 ppm)
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
    
    # Active learning state machine logic
    if learning_status == "Normal Operation":
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
            
    # Explainability attribution overlay
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
        explain_reason = "Model update finalized; baseline calibration optimized."
        
    # Helper background predictions
    temp_drift = (temp - 24.5) * 0.02
    hum_drift = (humidity - 58.2) * 0.005
    vel_drift = (velocity - 2.5) * 0.03
    
    predicted_nitro = round(max(0.0, actual_nitro + np.random.randn() * 0.4 + temp_drift * 10 + vel_drift * 4), 2)
    predicted_ammonia = round(max(0.0, actual_ammonia + np.random.randn() * 0.04 + hum_drift * 2 + vel_drift), 3)
    
    # Database Logging (1Hz / every 10 ticks)
    if t % 10 == 0:
        try:
            conn = sqlite3.connect(str(DATABASE_PATH))
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
            
    # Shift history data lists
    st.session_state.hist_ticks.pop(0); st.session_state.hist_ticks.append(t)
    st.session_state.hist_co_act.pop(0); st.session_state.hist_co_act.append(actual_co)
    st.session_state.hist_co_pred.pop(0); st.session_state.hist_co_pred.append(co_ppm)
    st.session_state.hist_eth_act.pop(0); st.session_state.hist_eth_act.append(actual_eth)
    st.session_state.hist_eth_pred.pop(0); st.session_state.hist_eth_pred.append(eth_ppm)
    st.session_state.hist_nitro_act.pop(0); st.session_state.hist_nitro_act.append(actual_nitro)
    st.session_state.hist_nitro_pred.pop(0); st.session_state.hist_nitro_pred.append(predicted_nitro)
    st.session_state.hist_ammo_act.pop(0); st.session_state.hist_ammo_act.append(actual_ammonia)
    st.session_state.hist_ammo_pred.pop(0); st.session_state.hist_ammo_pred.append(predicted_ammonia)
    
    # Save State
    st.session_state.t = t + 1
    st.session_state.velocity = velocity
    st.session_state.actual_co = actual_co
    st.session_state.actual_eth = actual_eth
    st.session_state.actual_nitro = actual_nitro
    st.session_state.actual_ammonia = actual_ammonia
    
    st.session_state.learning_status = learning_status
    st.session_state.status_timer = status_timer
    st.session_state.recal_progress = recal_progress
    st.session_state.active_learning_events = active_learning_events
    st.session_state.last_recal_t = last_recal_t
else:
    # Set mock defaults when paused
    temp = 24.0
    humidity = 55.0
    velocity = st.session_state.velocity
    latency_ms = 0.0
    uncertainty_score = 0.0
    predicted_state = "Standby"
    agent_action = "Standby"
    actual_co = st.session_state.actual_co
    actual_eth = st.session_state.actual_eth
    co_ppm = 0.0
    eth_ppm = 0.0
    learning_status = st.session_state.learning_status
    active_learning_events = st.session_state.active_learning_events
    last_recal_t = st.session_state.last_recal_t
    recal_progress = st.session_state.recal_progress
    new_reading = np.zeros(16)
    explain_source = "Standby"
    explain_confidence = 0.0
    explain_reason = "Simulation is paused"

# ----------------- TABS ROUTING -----------------
tab_cockpit, tab_analysis, tab_history, tab_settings = st.tabs(["🚀 Cockpit", "📊 Analysis", "📜 History Logs", "⚙️ Settings"])

# =========================================================================
# COCKPIT VIEW
# =========================================================================
with tab_cockpit:
    # 1. Toxic Override Warning Alert Banner
    if (actual_co > 0.5 or actual_eth > 0.5) or (learning_status != "Normal Operation"):
        banner_html = f"""<div class="toxic-alert" style="display: flex; flex-direction: column; align-items: flex-start; gap: 0.75rem;">
<div style="display: flex; align-items: center; gap: 0.75rem; width: 100%;">
<span style="font-size: 1rem; font-weight: 700; letter-spacing: 0.5px;">⚠️ TOXIC GAS CONCENTRATION DETECTED! INITIATING GRADIENT ASCENT OVERRIDE</span>
</div>
<div style="display: flex; gap: 2.5rem; width: 100%; padding-top: 0.75rem; border-top: 1px solid rgba(244, 63, 94, 0.25); font-size: 0.8rem; font-weight: 500; color: rgba(248, 250, 252, 0.95); flex-wrap: wrap;">
<div>
<span style="color: rgba(248, 250, 252, 0.55); display: block; font-size: 0.68rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 2px;">Decision Source</span>
<span>{explain_source}</span>
</div>
<div>
<span style="color: rgba(248, 250, 252, 0.55); display: block; font-size: 0.68rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 2px;">Confidence</span>
<span>{explain_confidence:.1f}%</span>
</div>
<div>
<span style="color: rgba(248, 250, 252, 0.55); display: block; font-size: 0.68rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 2px;">Reason / Causal Attribution</span>
<span>{explain_reason}</span>
</div>
</div>
</div>"""
        st.markdown(banner_html, unsafe_allow_html=True)

    # 2. KPI Cards Bar
    class_colors = {
        'Air': '#10b981',
        'Pure CO': '#f43f5e',
        'Pure Ethylene': '#38bdf8',
        'Mixture': '#a855f7',
        'Standby': '#64748b',
        'Unknown': '#64748b'
    }
    class_color = class_colors.get(predicted_state, '#10b981')
    
    kpis_html = f"""<div class="kpi-grid">
<div class="kpi-card temp">
<div class="kpi-title">Ambient Temp</div>
<div class="kpi-value">{temp:.1f}<span class="kpi-suffix">°C</span></div>
</div>
<div class="kpi-card humidity">
<div class="kpi-title">RH Humidity</div>
<div class="kpi-value">{humidity:.1f}<span class="kpi-suffix">%</span></div>
</div>
<div class="kpi-card wind">
<div class="kpi-title">Wind Velocity</div>
<div class="kpi-value">{velocity:.2f}<span class="kpi-suffix">m/s</span></div>
</div>
<div class="kpi-card latency">
<div class="kpi-title">Inference Latency</div>
<div class="kpi-value">{latency_ms:.1f}<span class="kpi-suffix">ms</span></div>
</div>
<div class="kpi-card uncertainty">
<div class="kpi-title">Uncertainty (Entropy)</div>
<div class="kpi-value">{uncertainty_score:.3f}</div>
</div>
<div class="kpi-card class">
<div class="kpi-title">Gas Classification</div>
<div class="kpi-value" style="font-size: 1.4rem; color: {class_color};">{predicted_state}</div>
</div>
</div>"""
    st.markdown(kpis_html, unsafe_allow_html=True)

    # 3. Main Dashboard Layout (Split Column Layout)
    col_chart, col_status = st.columns([8, 4])
    
    with col_chart:
        st.markdown('<div class="glass-card" style="padding-bottom:1rem;">', unsafe_allow_html=True)
        # Selection widgets for active chart tracking
        gas_filter = st.radio(
            "Follow Gas Channel", 
            ["CO", "Ethylene", "Nitrogen", "Ammonia", "All (Overview)"], 
            horizontal=True,
            index=0
        )
        
        # Build Plotly Figure matching the exact dark-themed aesthetics of Flask Dashboard
        fig = go.Figure()
        t_ticks = list(range(60))
        
        if gas_filter == 'CO':
            fig.add_trace(go.Scatter(x=t_ticks, y=st.session_state.hist_co_act, name='CO Actual', line=dict(color='#f43f5e', width=2.5)))
            fig.add_trace(go.Scatter(x=t_ticks, y=st.session_state.hist_co_pred, name='CO Predicted', line=dict(color='#fb7185', width=2, dash='dot')))
        elif gas_filter == 'Ethylene':
            fig.add_trace(go.Scatter(x=t_ticks, y=st.session_state.hist_eth_act, name='Ethylene Actual', line=dict(color='#38bdf8', width=2.5)))
            fig.add_trace(go.Scatter(x=t_ticks, y=st.session_state.hist_eth_pred, name='Ethylene Predicted', line=dict(color='#93c5fd', width=2, dash='dot')))
        elif gas_filter == 'Nitrogen':
            fig.add_trace(go.Scatter(x=t_ticks, y=st.session_state.hist_nitro_act, name='Nitrogen Actual', line=dict(color='#a855f7', width=2.5)))
            fig.add_trace(go.Scatter(x=t_ticks, y=st.session_state.hist_nitro_pred, name='Nitrogen Predicted', line=dict(color='#d8b4fe', width=2, dash='dot')))
        elif gas_filter == 'Ammonia':
            fig.add_trace(go.Scatter(x=t_ticks, y=st.session_state.hist_ammo_act, name='Ammonia Actual', line=dict(color='#eab308', width=2.5)))
            fig.add_trace(go.Scatter(x=t_ticks, y=st.session_state.hist_ammo_pred, name='Ammonia Predicted', line=dict(color='#fef08a', width=2, dash='dot')))
        else: # All Overview
            fig.add_trace(go.Scatter(x=t_ticks, y=st.session_state.hist_co_pred, name='CO Pred (ppm)', line=dict(color='#fb7185', width=2)))
            fig.add_trace(go.Scatter(x=t_ticks, y=st.session_state.hist_eth_pred, name='Ethylene Pred (ppm)', line=dict(color='#38bdf8', width=2)))
            fig.add_trace(go.Scatter(x=t_ticks, y=[v/50.0 for v in st.session_state.hist_nitro_pred], name='Nitrogen Pred / 50', line=dict(color='#a855f7', width=2)))
            fig.add_trace(go.Scatter(x=t_ticks, y=st.session_state.hist_ammo_pred, name='Ammonia Pred (ppm)', line=dict(color='#eab308', width=2)))

        # Define dynamic Y axis scale limits
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.04)',
                zeroline=False,
                tickfont=dict(color='#64748b', size=10),
                range=[0, 59]
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.04)',
                zeroline=False,
                tickfont=dict(color='#64748b', size=10)
            ),
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.1,
                xanchor='left',
                x=0.01,
                font=dict(color='#f8fafc', size=10),
                bgcolor='rgba(0,0,0,0)'
            ),
            hovermode='x unified',
            height=370
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_status:
        # A. Sensor Array Live Heatmap Card
        def get_heatmap_color(val_sensor):
            val_norm = max(-0.5, min(6.0, val_sensor))
            normalized = (val_norm + 0.5) / 6.5
            r_val = int(max(11, min(244, (normalized * 2.5 - 0.5) * 255)))
            g_val = int(max(15, min(189, ((normalized - 0.8) * 5 * 255 if normalized > 0.8 else 0))))
            b_val = int(max(25, min(248, (1 - normalized * 1.5) * 255)))
            return f"rgba({r_val}, {g_val}, {b_val}, {0.1 + normalized * 0.75})"
            
        cells_html = ""
        for i in range(16):
            sensor_val = float(new_reading[i])
            bg_color = get_heatmap_color(sensor_val)
            border_color = 'rgba(244, 63, 94, 0.4)' if sensor_val > 4.5 else 'rgba(255, 255, 255, 0.08)'
            box_shadow = '0 0 10px rgba(244, 63, 94, 0.2)' if sensor_val > 4.5 else 'none'
            cells_html += f'<div class="heatmap-cell" style="background-color: {bg_color}; border-color: {border_color}; box-shadow: {box_shadow};">CH{i+1}<br><span style="font-weight:700; color:rgba(255,255,255,0.85);">{sensor_val:.2f}</span></div>'
            
        heatmap_html = f"""<div class="glass-card" style="margin-bottom:1rem;">
<div class="card-header" style="margin-bottom: 0.5rem; padding-bottom: 0.5rem;">
<h2 style="font-size:1.05rem;">Sensor Array Live Heatmap</h2>
</div>
<p style="color:#64748b; font-size: 0.75rem; margin-bottom: 0.5rem; margin-top: 0px;">Real-time 16-channel micro-sensor magnitude</p>
<div class="heatmap-grid">
{cells_html}
</div>
</div>"""
        st.markdown(heatmap_html, unsafe_allow_html=True)
        
        # B. Edge Action Banner Overlay
        action_badge_classes = {
            'Accept': 'accept',
            'Standby': 'accept',
            'Re-sample': 'resample',
            'Escalate (Human-in-the-loop)': 'escalate',
            'Escalate (Awaiting Human Oracle)': 'escalate',
            'Recalibrating Model...': 'resample',
            'Model Updated & Re-deployed': 'accept'
        }
        action_class = action_badge_classes.get(agent_action, 'resample')

        action_html = f"""<div class="glass-card" style="margin-top:0px; margin-bottom:1rem; padding: 1.15rem;">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
<span style="font-weight: 600; font-size: 0.85rem">Edge Controller Action:</span>
<span class="badge-status {action_class}">{agent_action}</span>
</div>
<div style="font-size: 0.72rem; color: #64748b; line-height: 1.4;">
* The agentic edge system performs automated overrides (gradient ascent sampling/escalation) based on real-time uncertainty margins.
</div>
</div>"""
        st.markdown(action_html, unsafe_allow_html=True)
        
        # C. Learning & Adaptation Status Card
        learning_badge_classes = {
            'Normal Operation': 'accept',
            'Model Drift Detected': 'drift',
            'Recalibration Running': 'calibrating',
            'Model Updated': 'updated'
        }
        learning_class = learning_badge_classes.get(learning_status, 'accept')
        learning_text = 'Drift (Wait Oracle)' if learning_status == 'Model Drift Detected' else learning_status

        # Active Learning Pipeline nodes
        oracle_color = '#f97316' if learning_status == 'Model Drift Detected' else '#64748b'
        oracle_shadow = '0 0 8px rgba(249, 115, 22, 0.4)' if learning_status == 'Model Drift Detected' else 'none'

        recal_color = '#38bdf8' if learning_status == 'Recalibration Running' else '#64748b'
        recal_shadow = '0 0 8px rgba(56, 189, 248, 0.4)' if learning_status == 'Recalibration Running' else 'none'

        update_color = '#a855f7' if learning_status == 'Model Updated' else '#64748b'
        update_shadow = '0 0 8px rgba(168, 85, 247, 0.4)' if learning_status == 'Model Updated' else 'none'

        progress_bar_html = ""
        if learning_status == "Recalibration Running":
            progress_bar_html = f"""
            <div style="margin-top: 0.5rem; margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.72rem; margin-bottom: 0.25rem;">
                    <span style="color: #64748b;">Tuning hyper-parameters (SGD)...</span>
                    <span style="font-weight: 700; color: #38bdf8;">{recal_progress}%</span>
                </div>
                <div class="bar-bg" style="height: 6px;">
                    <div class="bar-fill" style="width: {recal_progress}%; height: 100%;"></div>
                </div>
            </div>
            """

        elapsed_seconds = (st.session_state.t - last_recal_t) * 0.1
        if last_recal_t < 0:
            calibration_age = 'Never'
        elif elapsed_seconds < 5:
            calibration_age = 'Just now'
        elif elapsed_seconds < 60:
            calibration_age = f"{int(elapsed_seconds)}s ago"
        else:
            mins = int(elapsed_seconds // 60)
            secs = int(elapsed_seconds % 60)
            calibration_age = f"{mins}m {secs}s ago"
        adaptation_html = f"""<div class="glass-card" style="margin-top:0px;">
<div class="card-header" style="margin-bottom: 0.5rem; padding-bottom: 0.5rem;">
<h2 style="font-size:1.05rem;">Learning & Adaptation Status</h2>
</div>
<p style="color:#64748b; font-size: 0.75rem; margin-bottom: 0.75rem; margin-top:0px;">Active Learning, Human Oracle and Recalibration Cycles</p>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
<span style="font-weight: 600; font-size: 0.85rem">System Learning State:</span>
<span class="badge-status {learning_class}">{learning_text}</span>
</div>
{progress_bar_html}
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 0.5rem;">
<div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 0.5rem; padding: 0.6rem; text-align: center;">
<div style="font-size: 0.55rem; color: #64748b; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 2px;">Last Recalibration</div>
<div style="font-size: 0.9rem; font-weight: 700; font-family: 'Outfit', sans-serif;">{calibration_age}</div>
</div>
<div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 0.5rem; padding: 0.6rem; text-align: center;">
<div style="font-size: 0.55rem; color: #64748b; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 2px;">AL Events</div>
<div style="font-size: 0.9rem; font-weight: 700; font-family: 'Outfit', sans-serif;">{active_learning_events}</div>
</div>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem; padding-top: 0.75rem; border-top: 1px solid rgba(255, 255, 255, 0.08); font-size: 0.7rem; color: #64748b; font-weight: 600;">
<span style="color: {oracle_color}; text-shadow: {oracle_shadow}; transition: all 0.3s ease;">🧑‍💻 Human Oracle</span>
<span style="opacity: 0.3;">➔</span>
<span style="color: {recal_color}; text-shadow: {recal_shadow}; transition: all 0.3s ease;">⚙️ Recalibration</span>
<span style="opacity: 0.3;">➔</span>
<span style="color: {update_color}; text-shadow: {update_shadow}; transition: all 0.3s ease;">🔄 Model Update</span>
</div>
</div>"""
        st.markdown(adaptation_html, unsafe_allow_html=True)
with tab_analysis:
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM telemetry_history ORDER BY id DESC LIMIT 500")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        st.info("⚠️ No telemetry records found in the database. Please run the simulation loop for a few seconds to populate data.")
    else:
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
            t_val = r["temp"]
            if t_val < 23.0:
                temp_bands["low"].append(err)
            elif t_val <= 26.0:
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
        bands_data = {
            "velocity": {
                "low": float(np.mean(vel_bands["low"])) if vel_bands["low"] else 0.0,
                "mid": float(np.mean(vel_bands["mid"])) if vel_bands["mid"] else 0.0,
                "high": float(np.mean(vel_bands["high"])) if vel_bands["high"] else 0.0,
            },
            "temp": {
                "low": float(np.mean(temp_bands["low"])) if temp_bands["low"] else 0.0,
                "mid": float(np.mean(temp_bands["mid"])) if temp_bands["mid"] else 0.0,
                "high": float(np.mean(temp_bands["high"])) if temp_bands["high"] else 0.0,
            },
            "humidity": {
                "low": float(np.mean(hum_bands["low"])) if hum_bands["low"] else 0.0,
                "mid": float(np.mean(hum_bands["mid"])) if hum_bands["mid"] else 0.0,
                "high": float(np.mean(hum_bands["high"])) if hum_bands["high"] else 0.0,
            }
        }
        col_summary, col_breakdown = st.columns([1, 1])
        with col_summary:
            st.markdown(f"""<div class="glass-card" style="display:flex; flex-direction:column; gap:1.25rem; height: 100%;">
<div class="card-header">
<h2>Model Performance Summary</h2>
</div>
<div style="display:flex; align-items:center; gap:1rem; background-color:rgba(56,189,248,0.04); padding:1.25rem; border-radius:0.75rem; border:1px solid rgba(56,189,248,0.1)">
<div style="font-family:'Outfit', sans-serif; font-size:3rem; font-weight:800; color:#38bdf8; line-height:1;">{accuracy:.1f}%</div>
<div style="font-size:0.85rem; font-weight:600; line-height:1.3;">
Average Classification Accuracy<br>
<span style="color:#64748b; font-size:0.75rem; font-weight:400;">(Rolling last {len(rows)} records)</span>
</div>
</div>
<div>
<h3 style="font-size:0.85rem; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.75rem; margin-top:0px;">Mean Absolute Error (MAE)</h3>
<div class="mae-badge-grid">
<div class="mae-badge-card" style="border-left:3px solid #f43f5e">
<span class="mae-badge-name">Carbon Monoxide (CO)</span>
<span class="mae-badge-val">{mae_co:.4f}</span>
</div>
<div class="mae-badge-card" style="border-left:3px solid #38bdf8">
<span class="mae-badge-name">Ethylene (C2H4)</span>
<span class="mae-badge-val">{mae_eth:.4f}</span>
</div>
<div class="mae-badge-card" style="border-left:3px solid #a855f7">
<span class="mae-badge-name">Nitrogen (N2)</span>
<span class="mae-badge-val">{mae_nitro:.4f}</span>
</div>
<div class="mae-badge-card" style="border-left:3px solid #eab308">
<span class="mae-badge-name">Ammonia (NH3)</span>
<span class="mae-badge-val">{mae_ammonia:.4f}</span>
</div>
</div>
</div>
<div style="background-color:rgba(255,255,255,0.02); padding:1rem; border-radius:0.75rem; font-size:0.78rem; line-height:1.5; border:1px solid rgba(255, 255, 255, 0.08); margin-top: 0.5rem;">
<span style="color:#10b981; font-weight:600; display:block; margin-bottom:0.25rem;">🧑‍🔬 Model Performance Analysis:</span>
The model shows high robustness under normal conditions. As wind velocity increases above 3.0 m/s, turbulence causes sensor signal dispersion, resulting in a moderate spike in carbon monoxide MAE. Drift is effectively compensated by the velocity embedding layers.
</div>
</div>""", unsafe_allow_html=True)
        with col_breakdown:
            all_vals = [
                bands_data["velocity"]["low"], bands_data["velocity"]["mid"], bands_data["velocity"]["high"],
                bands_data["temp"]["low"], bands_data["temp"]["mid"], bands_data["temp"]["high"],
                bands_data["humidity"]["low"], bands_data["humidity"]["mid"], bands_data["humidity"]["high"]
            ]
            max_val = max(all_vals) if max(all_vals) > 0 else 0.05
            def get_bar_row(label, val):
                pct = (val / max_val) * 100
                return f"""<div class="bar-row">
<div class="bar-label-row"><span>{label}</span><span>{val:.4f}</span></div>
<div class="bar-bg"><div class="bar-fill" style="width: {pct}%;"></div></div>
</div>"""
            breakdown_html = f"""<div class="glass-card" style="height: 100%;">
<div class="card-header">
<h2>Environmental Quality Breakdown (Average MAE)</h2>
</div>
<h3 style="font-size:0.8rem; font-weight:700; color:#64748b; margin-bottom:0.5rem; margin-top:0px;">Wind Speed Impact</h3>
<div class="bar-chart-container" style="margin-bottom:1.5rem;">
{get_bar_row("Low Wind (< 2.0 m/s)", bands_data["velocity"]["low"])}
{get_bar_row("Moderate Wind (2.0 - 3.0 m/s)", bands_data["velocity"]["mid"])}
{get_bar_row("High Wind (> 3.0 m/s)", bands_data["velocity"]["high"])}
</div>
<h3 style="font-size:0.8rem; font-weight:700; color:#64748b; margin-bottom:0.5rem; margin-top:0px;">Temperature Range Impact</h3>
<div class="bar-chart-container" style="margin-bottom:1.5rem;">
{get_bar_row("Cool (< 23°C)", bands_data["temp"]["low"])}
{get_bar_row("Moderate (23 - 26°C)", bands_data["temp"]["mid"])}
{get_bar_row("Warm (> 26°C)", bands_data["temp"]["high"])}
</div>
<h3 style="font-size:0.8rem; font-weight:700; color:#64748b; margin-bottom:0.5rem; margin-top:0px;">Relative Humidity Impact</h3>
<div class="bar-chart-container">
{get_bar_row("Dry (< 50%)", bands_data["humidity"]["low"])}
{get_bar_row("Moderate (50 - 65%)", bands_data["humidity"]["mid"])}
{get_bar_row("Humid (> 65%)", bands_data["humidity"]["high"])}
</div>
</div>"""
            st.markdown(breakdown_html, unsafe_allow_html=True)
with tab_history:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('### SQLite Database Query Logs')
    
    col_f1, col_f2, col_f3 = st.columns([4, 4, 4])
    
    with col_f1:
        hist_gas_select = st.selectbox(
            "Filter Gas Channel",
            ["All Channels", "Carbon Monoxide (CO)", "Ethylene (C2H4)", "Nitrogen (N2)", "Ammonia (NH3)"]
        )
    with col_f2:
        hist_limit_select = st.selectbox(
            "Query Records Limit",
            [100, 250, 500, 1000]
        )
        
    # Execute query
    gas_map_val = {
        "All Channels": "all",
        "Carbon Monoxide (CO)": "co",
        "Ethylene (C2H4)": "eth",
        "Nitrogen (N2)": "nitro",
        "Ammonia (NH3)": "ammonia"
    }.get(hist_gas_select, "all")
    
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM telemetry_history ORDER BY id DESC LIMIT ?", (hist_limit_select,))
    history_rows = cursor.fetchall()
    conn.close()

    if not history_rows:
        st.warning("No records logged in SQLite telemetry database yet. Ensure the simulation is running.")
    else:
        table_records = []
        for r in history_rows:
            # Recreate expectations
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
                
            if gas_map_val == 'co':
                row = {
                    "Tick": r["timestamp"],
                    "Temp (°C)": r["temp"],
                    "Humidity (%)": r["humidity"],
                    "Wind (m/s)": r["velocity"],
                    "Gas": "CO",
                    "Actual (ppm)": r["actual_co"],
                    "Predicted (ppm)": r["predicted_co"],
                    "Abs Error": round(abs(r["actual_co"] - r["predicted_co"]), 4),
                    "Expected State": expected,
                    "Predicted State": r["predicted_state"],
                    "Action": r["agent_action"]
                }
            elif gas_map_val == 'eth':
                row = {
                    "Tick": r["timestamp"],
                    "Temp (°C)": r["temp"],
                    "Humidity (%)": r["humidity"],
                    "Wind (m/s)": r["velocity"],
                    "Gas": "Ethylene",
                    "Actual (ppm)": r["actual_eth"],
                    "Predicted (ppm)": r["predicted_eth"],
                    "Abs Error": round(abs(r["actual_eth"] - r["predicted_eth"]), 4),
                    "Expected State": expected,
                    "Predicted State": r["predicted_state"],
                    "Action": r["agent_action"]
                }
            elif gas_map_val == 'nitro':
                row = {
                    "Tick": r["timestamp"],
                    "Temp (°C)": r["temp"],
                    "Humidity (%)": r["humidity"],
                    "Wind (m/s)": r["velocity"],
                    "Gas": "Nitrogen",
                    "Actual (ppm)": r["actual_nitro"],
                    "Predicted (ppm)": r["predicted_nitro"],
                    "Abs Error": round(abs(r["actual_nitro"] - r["predicted_nitro"]), 4),
                    "Expected State": expected,
                    "Predicted State": r["predicted_state"],
                    "Action": r["agent_action"]
                }
            elif gas_map_val == 'ammonia':
                row = {
                    "Tick": r["timestamp"],
                    "Temp (°C)": r["temp"],
                    "Humidity (%)": r["humidity"],
                    "Wind (m/s)": r["velocity"],
                    "Gas": "Ammonia",
                    "Actual (ppm)": r["actual_ammonia"],
                    "Predicted (ppm)": r["predicted_ammonia"],
                    "Abs Error": round(abs(r["actual_ammonia"] - r["predicted_ammonia"]), 4),
                    "Expected State": expected,
                    "Predicted State": r["predicted_state"],
                    "Action": r["agent_action"]
                }
            else: # All overview
                row = {
                    "Tick": r["timestamp"],
                    "Temp (°C)": r["temp"],
                    "Humidity (%)": r["humidity"],
                    "Wind (m/s)": r["velocity"],
                    "Latency (ms)": r["latency_ms"],
                    "CO (Act/Pred)": f"{r['actual_co']:.3f} / {r['predicted_co']:.3f}",
                    "Eth (Act/Pred)": f"{r['actual_eth']:.3f} / {r['predicted_eth']:.3f}",
                    "Nitro (Act/Pred)": f"{r['actual_nitro']:.2f} / {r['predicted_nitro']:.2f}",
                    "Ammonia (Act/Pred)": f"{r['actual_ammonia']:.3f} / {r['predicted_ammonia']:.3f}",
                    "Classification": r["predicted_state"]
                }
            table_records.append(row)
            
        df_history = pd.DataFrame(table_records)
        st.dataframe(df_history, use_container_width=True)
        
        # Export CSV Download option
        csv_data = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Log as CSV",
            data=csv_data,
            file_name=f"canesy_nose_telemetry_history_{gas_map_val}.csv",
            mime="text/csv"
        )
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================================
# SYSTEM SETTINGS VIEW
# =========================================================================
with tab_settings:
    st.markdown('<div class="glass-card" style="max-width:700px;">', unsafe_allow_html=True)
    st.markdown('### Developer & System Settings')
    
    st.markdown('#### Telemetry Database Operations')
    st.markdown('The SQLite database tracks telemetry indices recursively. You can wipe this local data cache below to re-initialize your tests.')
    if st.button("Wipe & Re-initialize Telemetry Database", type="secondary"):
        try:
            conn = sqlite3.connect(str(DATABASE_PATH))
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS telemetry_history")
            conn.commit()
            conn.close()
            init_db()
            st.success("✅ Telemetry database wiped successfully!")
            time.sleep(1)
            st.rerun()
        except Exception as err:
            st.error(f"Error resetting database: {err}")
            
    st.markdown("---")
    st.markdown('#### System Edge Node Technical Specifications')
    specs_html = f"""<table style="width: 100%; font-size: 0.85rem; border-collapse: collapse; margin-top: 0.5rem; color:#f8fafc;">
<tr style="border-bottom: 1px solid rgba(255,255,255,0.08); height:35px;">
<td style="color:#64748b; font-weight:500;">Active Model:</td>
<td style="font-family:monospace; font-weight:600;">{ONNX_PATH.name}</td>
</tr>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.08); height:35px;">
<td style="color:#64748b; font-weight:500;">Model Path:</td>
<td style="font-family:monospace;">{str(ONNX_PATH)}</td>
</tr>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.08); height:35px;">
<td style="color:#64748b; font-weight:500;">Database Path:</td>
<td style="font-family:monospace;">{str(DATABASE_PATH)}</td>
</tr>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.08); height:35px;">
<td style="color:#64748b; font-weight:500;">Deployment Mode:</td>
<td>Multi-stage Edge-Simulation System (ONNX CPU)</td>
</tr>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.08); height:35px;">
<td style="color:#64748b; font-weight:500;">Local Logging Level:</td>
<td>SQLite Telemetry Logging Enabled (~1Hz rate)</td>
</tr>
</table>"""
    st.markdown(specs_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- RE-RUN SCHEDULER -----------------
# Trigger automatic rerun every 100ms if simulation is running to maintain live tick updates
if st.session_state.run_simulation:
    time.sleep(0.08)
    st.rerun()
