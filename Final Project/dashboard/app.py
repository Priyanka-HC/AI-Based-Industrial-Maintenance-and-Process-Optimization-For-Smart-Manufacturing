"""
Predictive Maintenance & Process Intelligence — Smart Manufacturing Dashboard
--------------------------------------------------------------------------
5 tabs:
  1. Dashboard & Digital Twin  - live 10-machine fleet, auto-cycling, gauges
  2. ML Model Performance      - metrics, feature importance, dataset insights
  3. Process Intelligence      - failure-type breakdown, trends
  4. AI Assistant (Groq)       - chat with an LLM about your machine data
  5. About & Team

⚠️ CHECK THESE BEFORE RUNNING (I can't see your real project files, so verify):
  - FEATURES list below must match the exact column names / order your
    maintenance_model.pkl was TRAINED on. AI4I2020 default names are used here.
  - Model file path: "maintenance_model.pkl" (adjust if yours lives elsewhere)
  - Dataset path: "ai4i2020.csv" (adjust if yours lives elsewhere, e.g. data/ai4i2020.csv)
  - Needs: streamlit, pandas, numpy, plotly, scikit-learn, joblib, python-dotenv, groq
      pip install streamlit pandas numpy plotly scikit-learn joblib python-dotenv groq
  - Needs a .env file next to app.py containing: GROQ_API_KEY=gsk_xxx
"""

import os
import time
import random
import yaml
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

try:
    import joblib
except ImportError:
    joblib = None

try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except ImportError:
    GROQ_SDK_AVAILABLE = False

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

MODEL_PATH = "../saved_models/model.pkl"
SCALER_PATH = "../saved_models/scaler.pkl"
DATA_PATH = "../dataset/ai4i2020.csv"

# ⚠️ VERIFY: must match training order exactly
# NOTE: the saved model/scaler were trained on 11 columns (Type + 5 sensors +
# 5 failure-mode flags: TWF/HDF/PWF/OSF/RNF). The flags are normally only
# known AFTER a failure occurs, so for live prediction we feed them as 0
# (no active failure signal) — see chat for why this is a modeling smell
# worth fixing later by retraining without those 5 leak columns.
SENSOR_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
MODEL_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "TWF", "HDF", "PWF", "OSF", "RNF",
]
TYPE_MAP = {"L": 0, "M": 1, "H": 2}

st.set_page_config(
    page_title="Predictive Maintenance & Process Intelligence",
    page_icon="⚙️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# THEME — dark navy / teal, card + badge styling (applies to login screen too)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0b1220; }
    section[data-testid="stSidebar"] { background-color: #0f1729; }
    h1, h2, h3, h4, p, span, label, .stMarkdown, .stCaption { color: #e5e9f0 !important; }

    /* Text & password inputs — dark box, light visible text while typing */
    .stTextInput input, .stTextInput input:focus,
    div[data-baseweb="input"] input, div[data-baseweb="input"] input:focus {
        background-color: #131b2e !important;
        color: #f8fafc !important;
        border: 1px solid #1f2937 !important;
        caret-color: #10b981 !important;
    }
    div[data-baseweb="input"] { background-color: #131b2e !important; }
    .stTextInput label, .stTextInput p { color: #94a3b8 !important; }

    /* Buttons */
    .stButton button, .stFormSubmitButton button {
        background-color: #10b981 !important; color: #0b1220 !important;
        border: none !important; font-weight: 700 !important;
    }
    .stButton button:hover, .stFormSubmitButton button:hover { background-color: #0da271 !important; }

    /* Tabs */
    button[data-baseweb="tab"] { color: #94a3b8 !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #10b981 !important; }
    div[data-baseweb="tab-highlight"] { background-color: #10b981 !important; }
    div[data-baseweb="tab-border"] { background-color: #1f2937 !important; }

    /* Alert boxes (info / success / error) */
    div[data-testid="stAlert"] { background-color: #131b2e !important; border: 1px solid #1f2937 !important; }
    div[data-testid="stAlert"] p { color: #e5e9f0 !important; }

    /* Forms (streamlit-authenticator renders inside st.form) */
    div[data-testid="stForm"] {
        background-color: #131b2e !important; border: 1px solid #1f2937 !important;
        border-radius: 12px !important; padding: 20px !important;
    }

    .header-bar {
        display: flex; align-items: center; justify-content: space-between;
        background: linear-gradient(135deg, #0f1729 0%, #131b2e 100%);
        border: 1px solid #1f2937; border-radius: 14px;
        padding: 18px 24px; margin-bottom: 18px;
    }
    .header-title { display: flex; align-items: center; gap: 14px; }
    .header-icon {
        width: 44px; height: 44px; border-radius: 10px;
        background: #10b98122; display: flex; align-items: center;
        justify-content: center; font-size: 24px;
    }
    .header-eyebrow { color: #10b981 !important; font-size: 11px; letter-spacing: 1.5px;
        font-weight: 700; text-transform: uppercase; margin: 0; }
    .header-main { font-size: 22px; font-weight: 800; margin: 0; color: #f8fafc !important; }
    .live-pill {
        display: inline-flex; align-items: center; gap: 8px;
        background: #10b98122; border: 1px solid #10b98155;
        border-radius: 20px; padding: 6px 14px; font-size: 13px; font-weight: 600;
        color: #10b981 !important;
    }
    .live-dot { width: 8px; height: 8px; border-radius: 50%; background: #10b981;
        box-shadow: 0 0 8px #10b981; }
    .kpi-box { text-align: right; }
    .kpi-label { font-size: 11px; color: #94a3b8 !important; margin: 0; }
    .kpi-value { font-size: 16px; font-weight: 700; margin: 0; color: #f8fafc !important; }

    .alert-banner {
        display: flex; align-items: center; justify-content: space-between;
        background: linear-gradient(135deg, #3b0d14 0%, #1a0a0f 100%);
        border: 1px solid #ef444455; border-left: 4px solid #ef4444;
        border-radius: 12px; padding: 16px 20px; margin-bottom: 18px;
    }
    .alert-title { color: #f87171 !important; font-weight: 800; font-size: 14px; margin: 0; }
    .alert-body { color: #cbd5e1 !important; font-size: 13px; margin: 4px 0 0 0; }

    .m-card {
        background: #131b2e; border: 1px solid #1f2937; border-radius: 12px;
        padding: 16px 18px; margin-bottom: 16px;
    }
    .m-card.critical { border-color: #ef444488; box-shadow: 0 0 0 1px #ef444422; }
    .m-card.warning { border-color: #f59e0b88; }
    .m-card-top { display: flex; justify-content: space-between; align-items: center; }
    .m-card-name { font-weight: 700; font-size: 15px; color: #f8fafc !important; }
    .m-id-tag { background: #1f2937; color: #94a3b8 !important; font-size: 11px;
        padding: 2px 7px; border-radius: 5px; margin-right: 8px; }
    .badge { font-size: 11px; font-weight: 800; padding: 4px 10px; border-radius: 6px; letter-spacing: 0.5px; }
    .badge-healthy { background: #10b98122; color: #10b981 !important; }
    .badge-warning { background: #f59e0b22; color: #f59e0b !important; }
    .badge-critical { background: #ef444422; color: #ef4444 !important; }
    .m-stats { display: flex; gap: 22px; margin: 12px 0 10px 0; }
    .m-stat-label { font-size: 11px; color: #94a3b8 !important; margin: 0; }
    .m-stat-value { font-size: 14px; font-weight: 700; margin: 0; }
    .risk-row { display: flex; justify-content: space-between; font-size: 12px;
        color: #94a3b8 !important; margin-bottom: 4px; }
    .risk-track { background: #1f2937; border-radius: 6px; height: 8px; overflow: hidden; }
    .risk-fill { height: 100%; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def status_for(risk):
    if risk >= 0.7:
        return "CRITICAL", "critical", "badge-critical", "#ef4444"
    if risk >= 0.3:
        return "WARNING", "warning", "badge-warning", "#f59e0b"
    return "HEALTHY", "", "badge-healthy", "#10b981"


# ---------------------------------------------------------------------------
# AUTHENTICATION — secured employee login + self-registration
# ---------------------------------------------------------------------------
AUTH_CONFIG_PATH = "auth_config.yaml"

if not os.path.exists(AUTH_CONFIG_PATH):
    st.error(
        "auth_config.yaml not found. Run `python generate_config.py` once "
        "in this folder to create it, then restart the app."
    )
    st.stop()

with open(AUTH_CONFIG_PATH) as f:
    auth_config = yaml.load(f, Loader=yaml.SafeLoader)

authenticator = stauth.Authenticate(
    auth_config["credentials"],
    auth_config["cookie"]["name"],
    auth_config["cookie"]["key"],
    auth_config["cookie"]["expiry_days"],
)

if not st.session_state.get("authentication_status"):
    st.markdown(
        """
        <style>
        /* Auth screen only: give the page a subtle radial glow */
        .stApp { background: radial-gradient(circle at 50% 0%, #16213d 0%, #0b1220 55%) !important; }
        div[data-testid="stForm"] {
            box-shadow: 0 8px 40px rgba(16,185,129,0.08), 0 2px 12px rgba(0,0,0,0.4) !important;
            border-top: 2px solid #10b981 !important;
        }
        button[data-baseweb="tab"] {
            font-size: 15px !important; padding: 10px 4px !important;
        }
        </style>
        <div style="display:flex; justify-content:center; margin-top:10px;">
            <div style="width:64px; height:64px; border-radius:16px; background:#10b98122;
                        border:1.5px solid #10b981; display:flex; align-items:center;
                        justify-content:center; font-size:30px;">⚙️</div>
        </div>
        <h2 style='text-align:center; margin-top:16px; margin-bottom:4px; color:#f8fafc; font-size:26px;'>
            Predictive Maintenance &amp; Process Intelligence
        </h2>
        <div style="display:flex; justify-content:center; margin-bottom:28px;">
            <span style="background:#10b98122; border:1px solid #10b98155; color:#10b981;
                         border-radius:20px; padding:5px 16px; font-size:12.5px; font-weight:600;">
                🔒 Secured Employee Access
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    center_l, center_mid, center_r = st.columns([1, 1.3, 1])
    with center_mid:
        login_tab, signup_tab = st.tabs(["🔑  Log In", "📝  Sign Up"])

        with login_tab:
            authenticator.login(location="main")
            if st.session_state.get("authentication_status") is False:
                st.error("Username or password is incorrect.")
            elif st.session_state.get("authentication_status") is None:
                st.info("Enter your employee credentials to access the dashboard.")

        with signup_tab:
            st.caption("Only pre-approved employee emails can create an account.")
            try:
                email, username, name = authenticator.register_user(
                    location="main",
                    pre_authorized=auth_config.get("preauthorized", {}).get("emails"),
                    captcha=False,
                )
                if email:
                    with open(AUTH_CONFIG_PATH, "w") as f:
                        yaml.dump(auth_config, f, default_flow_style=False)
                    st.success(f"Account created for {name}! Go to the Log In tab to sign in.")
            except Exception as e:
                st.error(str(e))

    st.stop()

# Authenticated from here on
with st.sidebar:
    st.success(f"Logged in as **{st.session_state.get('name')}**")
    authenticator.logout("🔓 Log Out", "sidebar")



# ---------------------------------------------------------------------------
# DATA / MODEL LOADING
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if joblib is None:
        return None
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_scaler():
    if joblib is None or not os.path.exists(SCALER_PATH):
        return None
    return joblib.load(SCALER_PATH)


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


model = load_model()
scaler = load_scaler()
df = load_data()


def check_machine(machine_type, air_temp, process_temp, rpm, torque, tool_wear):
    """Runs the trained model on one reading. Returns (prediction, risk_probability).
    Feeds Type + 5 sensors + zeros for the 5 failure-flag columns the model was
    trained on (those flags are unknown for a live/healthy machine)."""
    if model is None:
        return 0, 0.0
    row = {
        "Type": TYPE_MAP.get(machine_type, 0),
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0,
    }
    X = pd.DataFrame([row], columns=MODEL_FEATURES)
    if scaler is not None:
        X = scaler.transform(X)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    risk = proba[1] if len(proba) > 1 else 0.0
    return pred, risk


# ---------------------------------------------------------------------------
# FLEET SIMULATION (10 machines, auto-cycling)
# ---------------------------------------------------------------------------
def sample_reading():
    """Pull a real row from the dataset when available, else generate a plausible one."""
    if df is not None and all(f in df.columns for f in SENSOR_FEATURES):
        row = df.sample(1).iloc[0]
        reading = {f: float(row[f]) for f in SENSOR_FEATURES}
        reading["Type"] = row["Type"] if "Type" in df.columns else "L"
        return reading
    return {
        "Type": random.choice(["L", "M", "H"]),
        "Air temperature [K]": round(random.uniform(295, 305), 1),
        "Process temperature [K]": round(random.uniform(305, 315), 1),
        "Rotational speed [rpm]": round(random.uniform(1200, 2800), 0),
        "Torque [Nm]": round(random.uniform(3, 75), 1),
        "Tool wear [min]": round(random.uniform(0, 250), 0),
    }


def build_fleet(n=10):
    fleet = []
    for i in range(1, n + 1):
        reading = sample_reading()
        pred, risk = check_machine(
            reading["Type"], *[reading[f] for f in SENSOR_FEATURES]
        )
        fleet.append(
            {
                "id": f"Machine-{i:02d}",
                **reading,
                "prediction": pred,
                "risk": risk,
                "status": "⚠️ At Risk" if pred == 1 else "✅ Healthy",
                "updated": datetime.now().strftime("%H:%M:%S"),
            }
        )
    return fleet


if "fleet" not in st.session_state:
    st.session_state.fleet = build_fleet()
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

AUTO_REFRESH_SECONDS = 2

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("---")

    st.subheader("System Status")
    st.write("Model:", "🟢 Loaded" if model is not None else "🔴 Not found (check MODEL_PATH)")
    st.write("Scaler:", "🟢 Loaded" if scaler is not None else "🟡 Not found (predictions run unscaled)")
    st.write("Dataset:", "🟢 Loaded" if df is not None else "🔴 Not found (check DATA_PATH)")
    st.write("Groq SDK:", "🟢 Installed" if GROQ_SDK_AVAILABLE else "🔴 Run: pip install groq")
    st.write("Groq API Key:", "🟢 Found" if GROQ_API_KEY else "🔴 Missing from .env")

    st.markdown("---")
    auto_monitor = st.toggle("🔄 Auto-monitor fleet (every 2s)", value=False)
    if st.button("🔁 Refresh fleet now"):
        st.session_state.fleet = build_fleet()

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
_fleet_preview = st.session_state.fleet
_health_pct = int(100 * sum(1 for m in _fleet_preview if m["risk"] < 0.3) / len(_fleet_preview)) if _fleet_preview else 100
_avg_risk_preview = np.mean([m["risk"] for m in _fleet_preview]) * 100 if _fleet_preview else 0

st.markdown(
    f"""
    <div class="header-bar">
        <div class="header-title">
            <div class="header-icon">⚙️</div>
            <div>
                <p class="header-eyebrow">Final-Year Project</p>
                <p class="header-main">Predictive Maintenance & Process Intelligence</p>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:28px;">
            <div class="kpi-box">
                <p class="kpi-label">FLEET HEALTH</p>
                <p class="kpi-value">{_health_pct}%</p>
            </div>
            <div class="kpi-box">
                <p class="kpi-label">AVG RISK SCORE</p>
                <p class="kpi-value">{_avg_risk_preview:.1f}%</p>
            </div>
            <div class="live-pill"><span class="live-dot"></span> Monitoring Active</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🖥️ Dashboard & Digital Twin",
        "📊 ML Model Performance",
        "🏭 Process Intelligence",
        "🤖 AI Assistant",
        "👥 About & Team",
    ]
)

# ---------------------------------------------------------------------------
# TAB 1 — DASHBOARD & DIGITAL TWIN
# ---------------------------------------------------------------------------
with tab1:
    fleet = st.session_state.fleet
    at_risk = sum(1 for m in fleet if m["prediction"] == 1)
    healthy = len(fleet) - at_risk
    avg_risk = np.mean([m["risk"] for m in fleet]) if fleet else 0

    # Alert banner — worst machine, only if something is actually critical
    worst = max(fleet, key=lambda m: m["risk"]) if fleet else None
    if worst and worst["risk"] >= 0.7:
        st.markdown(
            f"""
            <div class="alert-banner">
                <div>
                    <p class="alert-title">⚠️ CRITICAL ALERT — HIGH FAILURE PROBABILITY</p>
                    <p class="alert-body">{worst['id']} shows {worst['risk']*100:.0f}% predicted failure risk
                    — air temp {worst['Air temperature [K]']:.1f}K, torque {worst['Torque [Nm]']:.1f}Nm,
                    tool wear {worst['Tool wear [min]']:.0f}min. Recommend inspection.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Machines", len(fleet))
    c2.metric("✅ Healthy", healthy)
    c3.metric("⚠️ At Risk", at_risk)
    c4.metric("Avg Risk Score", f"{avg_risk*100:.1f}%")

    st.markdown("### 🔧 Manual Machine Check")
    with st.expander("Enter sensor readings manually", expanded=False):
        colT, colA, colB, colC, colD, colE = st.columns(6)
        m_type = colT.selectbox("Type", ["L", "M", "H"])
        air_temp = colA.slider("Air temp [K]", 290.0, 310.0, 300.0)
        proc_temp = colB.slider("Process temp [K]", 300.0, 320.0, 310.0)
        rpm = colC.slider("Rotational speed [rpm]", 1000, 3000, 1500)
        torque = colD.slider("Torque [Nm]", 0.0, 80.0, 40.0)
        wear = colE.slider("Tool wear [min]", 0, 260, 100)
        if st.button("Check Machine Health"):
            pred, risk = check_machine(m_type, air_temp, proc_temp, rpm, torque, wear)
            if pred == 1:
                st.error(f"⚠️ Failure risk: {risk*100:.1f}% — needs attention")
            else:
                st.success(f"✅ Healthy — risk: {risk*100:.1f}%")

    st.markdown("### 🖥️ Live Fleet — Digital Twin")
    grid_cols = st.columns(2)
    for idx, m in enumerate(fleet):
        label, css_class, badge_class, color = status_for(m["risk"])
        with grid_cols[idx % 2]:
            st.markdown(
                f"""
                <div class="m-card {css_class}">
                    <div class="m-card-top">
                        <span class="m-card-name"><span class="m-id-tag">{m['id']}</span></span>
                        <span class="badge {badge_class}">{label}</span>
                    </div>
                    <div class="m-stats">
                        <div>
                            <p class="m-stat-label">Air Temp</p>
                            <p class="m-stat-value">{m['Air temperature [K]']:.1f} K</p>
                        </div>
                        <div>
                            <p class="m-stat-label">Torque</p>
                            <p class="m-stat-value">{m['Torque [Nm]']:.1f} Nm</p>
                        </div>
                        <div>
                            <p class="m-stat-label">Tool Wear</p>
                            <p class="m-stat-value">{m['Tool wear [min]']:.0f} min</p>
                        </div>
                    </div>
                    <div class="risk-row">
                        <span>Predicted Failure Risk</span>
                        <span style="color:{color}; font-weight:700;">{m['risk']*100:.0f}%</span>
                    </div>
                    <div class="risk-track">
                        <div class="risk-fill" style="width:{max(m['risk']*100, 2):.0f}%; background:{color};"></div>
                    </div>
                    <p style="font-size:11px; color:#64748b; margin:8px 0 0 0;">Updated {m['updated']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if auto_monitor:
        time.sleep(AUTO_REFRESH_SECONDS)
        st.session_state.fleet = build_fleet()
        st.rerun()

# ---------------------------------------------------------------------------
# TAB 2 — ML MODEL PERFORMANCE
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("### 📊 Model Performance")
    if df is not None:
        total = len(df)
        fail_col = "Machine failure" if "Machine failure" in df.columns else None
        if fail_col:
            failures = int(df[fail_col].sum())
            healthy_n = total - failures
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Records", total)
            m2.metric("Failure Records", failures)
            m3.metric("Healthy Records", healthy_n)

            fail_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]
            present = [c for c in fail_types if c in df.columns]
            if present:
                counts = df[present].sum().reset_index()
                counts.columns = ["Failure Type", "Count"]
                fig = px.bar(counts, x="Failure Type", y="Count", title="Failure Type Breakdown")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Raw Data Sample")
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.warning("Dataset not found — can't show performance stats. Check DATA_PATH.")

    if model is not None and hasattr(model, "feature_importances_"):
        st.markdown("#### Feature Importance")
        importance_df = pd.DataFrame(
            {"Feature": MODEL_FEATURES, "Importance": model.feature_importances_}
        ).sort_values("Importance", ascending=True)
        fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        if any(f in ["TWF", "HDF", "PWF", "OSF", "RNF"] for f in importance_df.nlargest(3, "Importance")["Feature"]):
            st.caption("⚠️ Note: the failure-flag columns (TWF/HDF/PWF/OSF/RNF) are likely dominating importance — these are data leakage, not real predictive signal. See chat for why.")

# ---------------------------------------------------------------------------
# TAB 3 — PROCESS INTELLIGENCE
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("### 🏭 Process Intelligence")
    if df is not None:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(
                df, x="Torque [Nm]", y="Tool wear [min]",
                color="Machine failure" if "Machine failure" in df.columns else None,
                title="Torque vs Tool Wear",
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.histogram(df, x="Rotational speed [rpm]", title="Rotational Speed Distribution")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### 📁 Upload New Process Data")
        uploaded = st.file_uploader("Upload a CSV to analyze", type="csv")
        if uploaded is not None:
            new_df = pd.read_csv(uploaded)
            st.dataframe(new_df.head(20), use_container_width=True)
            st.session_state["uploaded_df"] = new_df
    else:
        st.warning("Dataset not found — check DATA_PATH.")

# ---------------------------------------------------------------------------
# TAB 4 — AI ASSISTANT (GROQ)
# ---------------------------------------------------------------------------
with tab4:
    st.markdown("### 🤖 AI Maintenance Assistant")
    st.caption(f"Powered by Groq ({GROQ_MODEL})")

    if not GROQ_SDK_AVAILABLE:
        st.error("Groq SDK not installed. Run: `pip install groq`")
    elif not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found. Add it to a `.env` file next to app.py:\n\n`GROQ_API_KEY=gsk_xxx`")
    else:
        client = Groq(api_key=GROQ_API_KEY)

        # Build lightweight context from current fleet + dataset so answers are grounded
        fleet_summary = "\n".join(
            f"- {m['id']}: risk={m['risk']*100:.1f}%, status={m['status']}"
            for m in st.session_state.fleet
        )
        system_prompt = (
            "You are an expert assistant for a Predictive Maintenance and Process "
            "Intelligence system in smart manufacturing. Use the live fleet data "
            "below when relevant, and give concise, practical maintenance advice.\n\n"
            f"Current fleet snapshot:\n{fleet_summary}"
        )

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_msg = st.chat_input("Ask about machine health, failure risk, or maintenance advice...")
        if user_msg:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.write(user_msg)

            messages = [{"role": "system", "content": system_prompt}] + st.session_state.chat_history

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = client.chat.completions.create(
                            model=GROQ_MODEL,
                            messages=messages,
                            temperature=0.3,
                        )
                        answer = response.choices[0].message.content
                    except Exception as e:
                        answer = f"⚠️ Groq API error: {e}"
                    st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

# ---------------------------------------------------------------------------
# TAB 5 — ABOUT & TEAM
# ---------------------------------------------------------------------------
with tab5:
    st.markdown("### 👥 About This Project")
    st.markdown(
        """
**AI-Driven Predictive Maintenance and Process Intelligence System for Smart Manufacturing**

**Problem:** Unplanned equipment failures cause costly downtime in manufacturing.

**Solution:** A machine-learning model trained on the AI4I 2020 dataset predicts
failure risk from live sensor readings (air/process temperature, rotational
speed, torque, tool wear), displayed on a real-time digital-twin dashboard,
now paired with an AI assistant for natural-language maintenance guidance.

**How it works:**
1. Sensor readings streamed/simulated per machine
2. Trained classifier scores failure risk
3. Dashboard renders live cards + fleet health
4. AI Assistant answers questions grounded in the live fleet state

---
**Team:** *(edit this line with your actual team member names)*
        """
    )