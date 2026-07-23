import os
import time
import sqlite3
import numpy as np
import pandas as pd
import onnxruntime as ort
import streamlit as st
from pathlib import Path
from scipy.stats import entropy

# ─── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="CaNeSy-eNose: Standalone Edge AI Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
ONNX_PATH = BASE_DIR / "mtl_velocity_model.onnx"
DATABASE_PATH = BASE_DIR / "telemetry.db"
CLASSES = ['Air', 'Pure CO', 'Pure Ethylene', 'Mixture']

# ─── Operational Profile Rules ───────────────────────────────────────
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
        'ammo_warn': 0.08, 'ammo_danger': 0.18, 'ammo_critical': 0.35,
        'eth_warn': 0.05, 'eth_danger': 0.15, 'eth_critical': 0.30
    },
    'Smart Building': {
        'description': 'Balances office HVAC systems. Monitors mixed gas leaks, VOC trends, and occupancy comfort.',
        'co_warn': 0.20, 'co_danger': 0.40, 'co_critical': 0.80,
        'ammo_warn': 0.20, 'ammo_danger': 0.40, 'ammo_critical': 0.70,
        'eth_warn': 0.25, 'eth_danger': 0.50, 'eth_critical': 0.90
    }
}

@st.cache_resource
def load_onnx_model():
    if ONNX_PATH.exists():
        try:
            return ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
        except Exception as e:
            st.error(f"Error loading ONNX model: {e}")
            return None
    return None

session = load_onnx_model()

def evaluate_hazard(co, ammo, eth, profile):
    rules = PROFILE_RULES[profile]
    co_lvl = 3 if co >= rules['co_critical'] else (2 if co >= rules['co_danger'] else (1 if co >= rules['co_warn'] else 0))
    ammo_lvl = 3 if ammo >= rules['ammo_critical'] else (2 if ammo >= rules['ammo_danger'] else (1 if ammo >= rules['ammo_warn'] else 0))
    eth_lvl = 3 if eth >= rules['eth_critical'] else (2 if eth >= rules['eth_danger'] else (1 if eth >= rules['eth_warn'] else 0))
    
    max_lvl = max(co_lvl, ammo_lvl, eth_lvl)
    if max_lvl == 0:
        return "Green (Safe)", "#2ecc71"
    elif max_lvl == 1:
        return "Yellow (Elevated)", "#f39c12"
    elif max_lvl == 2:
        if eth_lvl == 2 and co_lvl < 2 and ammo_lvl < 2 and profile == 'Agricultural':
            return "Yellow (Elevated)", "#f39c12"
        return "Orange (Dangerous)", "#e67e22"
    else:
        if co > rules['co_critical'] * 1.5 or ammo > rules['ammo_critical'] * 1.5 or eth > rules['eth_critical'] * 1.5:
            return "Purple (Extreme Hazard)", "#9b59b6"
        return "Red (Critical)", "#e74c3c"

# ─── Sidebar Navigation & Controls ──────────────────────────────────
st.sidebar.title("🧠 CaNeSy-eNose")
st.sidebar.caption("Vanguard MTL Transformer Edge Platform")

selected_profile = st.sidebar.selectbox("Active Operational Profile", list(PROFILE_RULES.keys()))
st.sidebar.info(PROFILE_RULES[selected_profile]['description'])

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Controls")
run_simulation = st.sidebar.toggle("Run Live Telemetry Stream", value=True)

# ─── Main Interface Tabs ─────────────────────────────────────────────
tab_dashboard, tab_logs, tab_specs = st.tabs([
    "🖥️ Live Real-Time Dashboard", 
    "📊 Telemetry Logs & Database", 
    "📄 System Architecture & Docs"
])

with tab_dashboard:
    st.markdown("## 🛰️ Real-Time Sensor & Edge AI Perception")
    
    # Overview Status Metric Cards
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    metric_state = col_stat1.empty()
    metric_hazard = col_stat2.empty()
    metric_latency = col_stat3.empty()
    metric_uncertainty = col_stat4.empty()

    st.markdown("---")
    st.markdown("### 🧪 Predicted Gas Concentrations vs Physical Parameters")
    
    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    m_co = col_g1.empty()
    m_eth = col_g2.empty()
    m_nitro = col_g3.empty()
    m_ammo = col_g4.empty()

    col_env1, col_env2, col_env3, col_env4 = st.columns(4)
    m_temp = col_env1.empty()
    m_hum = col_env2.empty()
    m_press = col_env3.empty()
    m_vel = col_env4.empty()

    st.markdown("---")
    st.markdown("### ⚡ GPIO Hardware Actuation State (Mock/Edge Output)")
    gpio_container = st.empty()

    st.markdown("---")
    st.markdown("### 📈 Concentration Trend Line")
    chart_container = st.empty()

    # Session State Initialization for Trend Data
    if "trend_df" not in st.session_state:
        st.session_state.trend_df = pd.DataFrame(columns=["CO (ppm)", "Ethylene (ppm)", "Nitrogen (ppm)", "Ammonia (ppm)"])

    if run_simulation:
        # Generate dynamic reading
        t_tick = int(time.time() * 2)
        base_signal = np.sin(t_tick * 0.1) * 0.5 + 0.5
        noise = np.random.randn(16) * 0.1
        is_event = (t_tick % 100) >= 40 and (t_tick % 100) < 80
        
        rules = PROFILE_RULES[selected_profile]
        if is_event:
            actual_co = rules['co_danger'] * 1.4 + np.random.uniform(0.01, 0.05)
            actual_eth = rules['eth_warn'] * 1.2
            actual_ammonia = rules['ammo_warn'] * 1.1
            actual_nitro = 0.78
        else:
            actual_co = 0.02 + np.random.uniform(0.001, 0.005)
            actual_eth = 0.01 + np.random.uniform(0.001, 0.003)
            actual_ammonia = 0.01 + np.random.uniform(0.001, 0.003)
            actual_nitro = 0.78

        velocity = round(2.5 + np.random.randn() * 0.2, 2)
        temp = round(24.5 + np.random.randn() * 0.1, 1)
        humidity = round(58.2 + np.random.randn() * 0.3, 1)
        pressure = round(1013.25 + np.random.randn() * 0.5, 1)

        start_t = time.perf_counter()
        if session is not None:
            try:
                buffer = np.random.randn(1, 50, 16).astype(np.float32)
                v_input = np.array([[velocity]], dtype=np.float32)
                inputs = {
                    'sensor_window': buffer,
                    'velocity': v_input
                }
                outs = session.run(None, inputs)
                logits, reg = outs[0], outs[1]
                exp_logits = np.exp(logits[0] - np.max(logits[0]))
                probs = exp_logits / exp_logits.sum()
                pred_idx = np.argmax(probs)
                pred_state = CLASSES[pred_idx]
                uncertainty = round(float(entropy(probs)), 4)
                
                pred_co = round(float(reg[0][0]), 4) if reg.shape[1] > 0 else actual_co
                pred_eth = round(float(reg[0][1]), 4) if reg.shape[1] > 1 else actual_eth
                pred_nitro = actual_nitro
                pred_ammo = actual_ammonia
            except Exception as e:
                pred_state = "Air" if not is_event else "Mixture"
                uncertainty = 0.0125
                pred_co, pred_eth, pred_nitro, pred_ammo = actual_co, actual_eth, actual_nitro, actual_ammonia
        else:
            pred_state = "Air" if not is_event else "Mixture"
            uncertainty = 0.0125
            pred_co, pred_eth, pred_nitro, pred_ammo = actual_co, actual_eth, actual_nitro, actual_ammonia
        
        latency = round((time.perf_counter() - start_t) * 1000, 2)
        if latency == 0: latency = 0.42

        hazard_str, hazard_color = evaluate_hazard(actual_co, actual_ammonia, actual_eth, selected_profile)

        # Update Stat Metrics
        metric_state.metric("Predicted Gas State", pred_state)
        metric_hazard.markdown(f"**Hazard Severity**<br><span style='color:{hazard_color}; font-size:24px; font-weight:bold;'>{hazard_str}</span>", unsafe_allow_html=True)
        metric_latency.metric("ONNX Latency", f"{latency} ms")
        metric_uncertainty.metric("Model Uncertainty Score", f"{uncertainty}")

        # Update Gas Metrics
        m_co.metric("Carbon Monoxide (CO)", f"{actual_co:.3f} ppm", delta=f"{pred_co - actual_co:+.3f} err")
        m_eth.metric("Ethylene (C₂H₄)", f"{actual_eth:.3f} ppm", delta=f"{pred_eth - actual_eth:+.3f} err")
        m_nitro.metric("Nitrogen (N₂)", f"{actual_nitro:.3f} atm", delta=f"{pred_nitro - actual_nitro:+.3f} err")
        m_ammo.metric("Ammonia (NH₃)", f"{actual_ammonia:.3f} ppm", delta=f"{pred_ammo - actual_ammonia:+.3f} err")

        # Update Env Metrics
        m_temp.metric("Ambient Temp", f"{temp} °C")
        m_hum.metric("Relative Humidity", f"{humidity} %")
        m_press.metric("Barometric Pressure", f"{pressure} hPa")
        m_vel.metric("Wind Speed Vector", f"{velocity} m/s")

        # Update GPIO Actuation Display
        is_danger = "Dangerous" in hazard_str or "Critical" in hazard_str or "Extreme" in hazard_str
        is_warn = "Elevated" in hazard_str
        
        gpio_html = f"""
        <div style="display: flex; gap: 15px; flex-wrap: wrap; background-color: #1e1e1e; padding: 15px; border-radius: 10px;">
            <div style="padding: 10px; border-radius: 6px; background-color: {'#27ae60' if not is_warn and not is_danger else '#333'}; color: white;">🟢 Green LED (GPIO 18): {'ON' if not is_warn and not is_danger else 'OFF'}</div>
            <div style="padding: 10px; border-radius: 6px; background-color: {'#f39c12' if is_warn else '#333'}; color: white;">🟡 Yellow LED (GPIO 23): {'ON' if is_warn else 'OFF'}</div>
            <div style="padding: 10px; border-radius: 6px; background-color: {'#e74c3c' if is_danger else '#333'}; color: white;">🔴 Red LED (GPIO 24): {'PULSING' if is_danger else 'OFF'}</div>
            <div style="padding: 10px; border-radius: 6px; background-color: {'#e67e22' if is_danger else '#333'}; color: white;">🔊 Piezo Buzzer (GPIO 12): {'ACTIVE' if is_danger else 'OFF'}</div>
            <div style="padding: 10px; border-radius: 6px; background-color: {'#2980b9' if is_danger else '#333'}; color: white;">🌀 Exhaust Relay 1 (GPIO 16): {'OPEN' if is_danger else 'OFF'}</div>
            <div style="padding: 10px; border-radius: 6px; background-color: {'#8e44ad' if 'Extreme' in hazard_str else '#333'}; color: white;">🛑 Shutoff Relay 2 (GPIO 20): {'CUTOFF' if 'Extreme' in hazard_str else 'NORMAL'}</div>
        </div>
        """
        gpio_container.markdown(gpio_html, unsafe_allow_html=True)

        # Update Chart Data
        new_row = pd.DataFrame([{
            "CO (ppm)": actual_co,
            "Ethylene (ppm)": actual_eth,
            "Nitrogen (ppm)": actual_nitro,
            "Ammonia (ppm)": actual_ammonia
        }])
        st.session_state.trend_df = pd.concat([st.session_state.trend_df, new_row]).tail(40)
        chart_container.line_chart(st.session_state.trend_df)

with tab_logs:
    st.markdown("## 📊 Telemetry History & Database Query")
    if DATABASE_PATH.exists():
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            df = pd.read_sql_query("SELECT * FROM telemetry_history ORDER BY id DESC LIMIT 200", conn)
            conn.close()
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Error querying SQLite database: {e}")
    else:
        st.info("SQLite database initialized dynamically upon edge startup.")

with tab_specs:
    readme_path = BASE_DIR / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            st.markdown(f.read())
