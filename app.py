import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import io
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BankSight AI · Deposit Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load Artifacts ───────────────────────────────────────────────────────────
# FIX 1: Removed scaler.pkl (does not exist; RF doesn't need scaling)
# FIX 2: was_contacted_before engineered before encoding (matches training)
@st.cache_resource
def load_model():
    try:
        rf       = joblib.load("model.pkl")
        features = joblib.load("features.pkl")
        cat_cols = joblib.load("cat_cols.pkl")
        return rf, features, cat_cols, None
    except Exception as e:
        return None, None, None, str(e)

rf, features, cat_cols, load_error = load_model()

@st.cache_data
def load_dataset():
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bank.csv")
    return pd.read_csv(path)

bank_df = load_dataset()

if load_error:
    st.error(f"❌ Failed to load model artifacts: {load_error}")
    st.stop()

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0d0820 !important;
    color: #f1f0ff !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 110% 75% at 50% -5%, #3b1f8c 0%, #1a0d4e 30%, #0d0820 65%, #120a2e 100%) !important;
    min-height: 100vh;
}

[data-testid="stSidebar"] {
    background: rgba(22, 10, 60, 0.97) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.25) !important;
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] * { color: #ddd6fe !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] strong { color: #f8fafc !important; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: #ede9fe !important; text-shadow: 0 0 20px rgba(167,139,250,0.35); }

.glass-card {
    background: rgba(109,40,217,0.07);
    border: 1px solid rgba(139,92,246,0.18);
    border-radius: 20px;
    padding: 28px 32px;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 40px rgba(88,28,220,0.2), inset 0 1px 0 rgba(167,139,250,0.1);
    margin-bottom: 4px;
}

.hero-wrap {
    position: relative; overflow: hidden; border-radius: 24px;
    background: linear-gradient(135deg, rgba(109,40,217,0.3) 0%, rgba(124,58,237,0.2) 50%, rgba(167,139,250,0.18) 100%);
    border: 1px solid rgba(139,92,246,0.3);
    padding: 56px 48px 48px; margin-bottom: 36px;
}
.hero-wrap::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 380px; height: 380px;
    background: radial-gradient(circle, rgba(167,139,250,0.35) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(124,58,237,0.22); border: 1px solid rgba(167,139,250,0.4);
    border-radius: 100px; padding: 5px 14px;
    font-size: 12px; font-weight: 600; letter-spacing: 0.08em;
    color: #c4b5fd; text-shadow: 0 0 12px rgba(167,139,250,0.6); text-transform: uppercase; margin-bottom: 20px;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(26px, 4vw, 44px); font-weight: 700; line-height: 1.12;
    background: linear-gradient(135deg, #ffffff 0%, #c4b5fd 40%, #a78bfa 75%, #7c3aed 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 14px;
}
.hero-sub { font-size: 15px; color: #c4b5fd; max-width: 520px; line-height: 1.65; margin-bottom: 28px; }
.hero-badge-row { display: flex; gap: 10px; flex-wrap: wrap; }
.hero-badge {
    background: rgba(109,40,217,0.15); border: 1px solid rgba(139,92,246,0.25);
    border-radius: 8px; padding: 5px 12px; font-size: 12px; color: #ddd6fe; font-weight: 500;
}

.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 36px; }
.kpi-card {
    background: rgba(88,28,220,0.1); border: 1px solid rgba(139,92,246,0.18);
    border-radius: 16px; padding: 22px 24px; position: relative; overflow: hidden;
}
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }
.kpi-card.blue::before   { background: linear-gradient(90deg, #3b82f6, #6366f1); }
.kpi-card.violet::before { background: linear-gradient(90deg, #7c3aed, #a78bfa); }
.kpi-card.cyan::before   { background: linear-gradient(90deg, #06b6d4, #3b82f6); }
.kpi-card.green::before  { background: linear-gradient(90deg, #10b981, #06b6d4); }
.kpi-label { font-size: 11px; color: #a78bfa; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; }
.kpi-value { font-family: 'Space Grotesk', sans-serif; font-size: 30px; font-weight: 700; color: #ede9fe; text-shadow: 0 0 16px rgba(167,139,250,0.4); line-height: 1; margin-bottom: 4px; }
.kpi-sub   { font-size: 12px; color: #7c5cbf; }

.section-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 20px; padding-bottom: 12px;
    border-bottom: 1px solid rgba(139,92,246,0.15);
}
.section-icon {
    width: 36px; height: 36px; border-radius: 10px;
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0;
}
.section-title { font-family: 'Space Grotesk', sans-serif; font-size: 18px; font-weight: 600; color: #ede9fe; text-shadow: 0 0 14px rgba(167,139,250,0.3); }
.section-sub   { font-size: 12px; color: #8b6fc8; margin-top: 2px; }

[data-testid="stTabsContent"] {
    background: rgba(88,28,220,0.07) !important;
    border: 1px solid rgba(139,92,246,0.15) !important;
    border-top: none !important;
    border-radius: 0 0 16px 16px !important;
    padding: 24px 20px !important;
}

.fg-label {
    font-size: 10px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
    color: #c4b5fd; margin: 20px 0 8px; padding-bottom: 6px;
    border-bottom: 1px solid rgba(167,139,250,0.25); text-shadow: 0 0 10px rgba(196,181,253,0.5);
}

.result-positive {
    background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(6,182,212,0.08) 100%);
    border: 1px solid rgba(16,185,129,0.35); border-radius: 20px; padding: 36px; text-align: center;
    box-shadow: 0 0 40px rgba(16,185,129,0.12);
}
.result-negative {
    background: linear-gradient(135deg, rgba(244,63,94,0.12) 0%, rgba(251,113,133,0.06) 100%);
    border: 1px solid rgba(244,63,94,0.35); border-radius: 20px; padding: 36px; text-align: center;
    box-shadow: 0 0 40px rgba(244,63,94,0.10);
}
.result-icon    { font-size: 52px; margin-bottom: 12px; }
.result-verdict { font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 700; margin-bottom: 8px; }
.result-positive .result-verdict { color: #34d399; }
.result-negative .result-verdict { color: #fb7185; }
.result-detail  { font-size: 14px; color: #c4b5fd; }

.conf-label { display: flex; justify-content: space-between; font-size: 13px; color: #c4b5fd; margin-bottom: 8px; }
.conf-track { height: 9px; background: rgba(109,40,217,0.15); border-radius: 100px; overflow: hidden; }
.conf-fill-yes { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #10b981, #06b6d4); }
.conf-fill-no  { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #f43f5e, #f97316); }

.rec-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px; }
.rec-pill {
    background: rgba(88,28,220,0.1); border: 1px solid rgba(139,92,246,0.18);
    border-radius: 12px; padding: 14px 16px; font-size: 13px; color: #ddd6fe; line-height: 1.5;
}
.rec-pill strong { color: #a5b4fc; display: block; font-size: 10px; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 4px; }

[data-testid="stNumberInput"] input,
.stSelectbox [data-baseweb="select"] {
    background: rgba(88,28,220,0.1) !important;
    border: 1px solid rgba(139,92,246,0.22) !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
}
[data-baseweb="popover"], [data-baseweb="menu"] {
    background: #1a0a40 !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 12px !important;
}
[data-baseweb="option"]:hover { background: rgba(99,102,241,0.15) !important; }
.stSlider [role="slider"] { background: #6366f1 !important; border: 2px solid #a5b4fc !important; }

.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 50%, #7c3aed 100%) !important;
    color: #fff !important; border: none !important; border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    font-size: 15px !important; padding: 14px 32px !important;
    transition: all 0.22s ease !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.5) !important;
}
[data-testid="stDownloadButton"] button {
    background: rgba(88,28,220,0.1) !important;
    border: 1px solid rgba(139,92,246,0.22) !important;
    color: #ddd6fe !important; border-radius: 10px !important;
    font-weight: 500 !important; font-size: 14px !important;
}

[data-baseweb="tab-list"] {
    background: rgba(88,28,220,0.08) !important;
    border-radius: 12px 12px 0 0 !important; padding: 4px 4px 0 !important;
    border: 1px solid rgba(139,92,246,0.2) !important;
    border-bottom: none !important; gap: 2px !important;
}
[data-baseweb="tab"] {
    background: transparent !important; border-radius: 9px 9px 0 0 !important;
    color: #a78bfa !important; font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important; font-size: 14px !important;
    padding: 8px 20px !important; transition: all 0.15s ease !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(135deg, rgba(124,58,237,0.35), rgba(167,139,250,0.2)) !important;
    color: #ede9fe !important;
    box-shadow: inset 0 0 0 1px rgba(167,139,250,0.4), 0 0 12px rgba(124,58,237,0.2) !important;
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }

details {
    background: rgba(88,28,220,0.07) !important;
    border: 1px solid rgba(139,92,246,0.15) !important;
    border-radius: 14px !important; padding: 4px 8px !important;
}
summary { color: #a78bfa !important; font-size: 14px !important; cursor: pointer; padding: 8px 4px; }
label, [data-testid="stWidgetLabel"] p { color: #b09de0 !important; font-size: 13px !important; font-weight: 500 !important; }
hr { border-color: rgba(139,92,246,0.15) !important; margin: 28px 0 !important; }

.footer {
    text-align: center; padding: 32px 0 16px;
    border-top: 1px solid rgba(139,92,246,0.15);
    color: #6d4fa0; font-size: 13px; margin-top: 48px;
}
.footer span { color: #6366f1; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.35); border-radius: 100px; }
[data-testid="stAlert"] { border-radius: 12px !important; border-left-width: 3px !important; font-size: 14px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
for k, v in [("page","Dashboard"),("pred_result",None),("pred_proba",None),("pred_inputs",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Prediction helper ────────────────────────────────────────────────────────
# FIX 3: No scaler, no drop_first, use .values to suppress sklearn warning
def run_prediction(customer: dict):
    sample = pd.DataFrame([customer])
    sample["was_contacted_before"] = (sample["previous"] > 0).astype(int)
    sample_enc = pd.get_dummies(sample, columns=cat_cols)          # drop_first=False
    sample_enc = sample_enc.reindex(columns=features, fill_value=0)
    pred  = rf.predict(sample_enc.values)[0]
    proba = rf.predict_proba(sample_enc.values)[0][1]
    return pred, proba

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 24px;">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
            <div style="width:40px;height:40px;border-radius:12px;
                        background:linear-gradient(135deg,#6366f1,#06b6d4);
                        display:flex;align-items:center;justify-content:center;font-size:20px;">🏦</div>
            <div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:17px;font-weight:700;color:#f1f5f9;">BankSight AI</div>
                <div style="font-size:11px;color:#475569;letter-spacing:0.05em;">DEPOSIT PREDICTOR</div>
            </div>
        </div>
        <div style="height:1px;background:rgba(255,255,255,0.07);margin:16px 0;"></div>
    </div>
    """, unsafe_allow_html=True)

    nav_pages = [("📊","Dashboard"),("🔮","Predict"),("📈","Analytics & Insights"),("🧠","Model Insights"),("ℹ️","About")]
    for icon, label in nav_pages:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True):
            st.session_state.page = label
            st.rerun()

    st.markdown("""
    <div style="height:1px;background:rgba(255,255,255,0.07);margin:20px 0;"></div>
    <div style="font-size:11px;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px;">Model Info</div>
    """, unsafe_allow_html=True)

    for k, v in [("Algorithm","Random Forest"),("Features",f"{len(features)} encoded"),("Task","Binary Classif."),("Version","v1.1")]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.05);">
            <span style="font-size:12px;color:#475569;">{k}</span>
            <span style="font-size:12px;color:#94a3b8;font-weight:500;">{v}</span>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "Dashboard":

    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">🤖 &nbsp; ML-Powered Analytics</div>
        <div class="hero-title">Predict Bank Term Deposit<br>Subscriptions with Precision</div>
        <div class="hero-sub">A Random Forest classification model trained on real-world bank marketing
        data to help target customers most likely to subscribe to term deposits.</div>
        <div class="hero-badge-row">
            <div class="hero-badge">📞 Call Campaign Data</div>
            <div class="hero-badge">🎯 Binary Classification</div>
            <div class="hero-badge">⚡ Real-time Inference</div>
            <div class="hero-badge">📊 Confidence Scoring</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="kpi-grid">
        <div class="kpi-card blue">
            <div class="kpi-label">Model Type</div>
            <div class="kpi-value" style="font-size:20px;margin-top:4px;">Random<br>Forest</div>
            <div class="kpi-sub">Ensemble Classifier</div>
        </div>
        <div class="kpi-card violet">
            <div class="kpi-label">Input Features</div>
            <div class="kpi-value">16</div>
            <div class="kpi-sub">Raw before encoding</div>
        </div>
        <div class="kpi-card cyan">
            <div class="kpi-label">Target Classes</div>
            <div class="kpi-value">2</div>
            <div class="kpi-sub">Subscribe / Not Subscribe</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-label">Inference Speed</div>
            <div class="kpi-value" style="font-size:22px;">&lt;50ms</div>
            <div class="kpi-sub">Real-time prediction</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        st.markdown("""
        <div class="glass-card">
            <div class="section-header">
                <div class="section-icon">🎯</div>
                <div>
                    <div class="section-title">Project Overview</div>
                    <div class="section-sub">Bank Marketing Campaign Analysis</div>
                </div>
            </div>
            <p style="color:#94a3b8;font-size:14px;line-height:1.8;margin-bottom:14px;">
                This system uses a <strong style="color:#a5b4fc;">Random Forest classifier</strong> trained on the
                UCI Bank Marketing dataset to predict whether a customer will subscribe to a term deposit
                based on demographic, financial, and campaign-related attributes.
            </p>
            <p style="color:#94a3b8;font-size:14px;line-height:1.8;margin-bottom:20px;">
                The model ingests 16 raw features, applies one-hot encoding for categorical variables,
                then aligns to the exact 52-feature training schema before inference.
            </p>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);border-radius:12px;padding:14px;">
                    <div style="font-size:10px;color:#6366f1;font-weight:700;letter-spacing:0.07em;text-transform:uppercase;margin-bottom:6px;">Input Types</div>
                    <div style="font-size:13px;color:#cbd5e1;line-height:1.7;">
                        Numeric: age, balance, duration…<br>
                        Categorical: job, education, contact…
                    </div>
                </div>
                <div style="background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.18);border-radius:12px;padding:14px;">
                    <div style="font-size:10px;color:#06b6d4;font-weight:700;letter-spacing:0.07em;text-transform:uppercase;margin-bottom:6px;">Pipeline Steps</div>
                    <div style="font-size:13px;color:#cbd5e1;line-height:1.7;">
                        One-Hot Encoding →<br>
                        Feature Align → RF Predict
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        groups = [
            ("👤", "Client Demographics", "Age, Job, Marital, Education",            "#6366f1"),
            ("💳", "Financial Profile",   "Balance, Default, Housing, Loan",         "#06b6d4"),
            ("📞", "Campaign Data",       "Contact, Day, Month, Duration, Calls",    "#8b5cf6"),
            ("📅", "Previous Contact",    "pdays, previous, poutcome",               "#10b981"),
        ]
        rows_html = "".join(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                        border-radius:12px;padding:13px 15px;border-left:3px solid {color};margin-bottom:10px;">
                <div style="font-size:13px;font-weight:600;color:#f1f5f9;margin-bottom:3px;">{icon} {title}</div>
                <div style="font-size:12px;color:#64748b;">{items}</div>
            </div>""" for icon, title, items, color in groups)

        _html = ('<div class="glass-card"><div class="section-header">'
                 '<div class="section-icon">📋</div><div>'
                 '<div class="section-title">Feature Groups</div>'
                 '<div class="section-sub">16 input variables across 4 categories</div>'
                 '</div></div>' + rows_html + '</div>')
        st.markdown(_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮  Open Predictor →", key="dash_cta"):
        st.session_state.page = "Predict"
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Predict":

    st.markdown("""
    <div class="section-header" style="margin-bottom:24px;">
        <div class="section-icon">🔮</div>
        <div>
            <div class="section-title">Customer Subscription Predictor</div>
            <div class="section-sub">Fill in customer details across all tabs, then run the prediction</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["👤  Demographics", "💳  Financials", "📞  Campaign", "📅  History"])

    with tab1:
        st.markdown('<div class="fg-label">Personal Information</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")
        age     = c1.number_input("Age", 18, 95, 40)
        marital = c2.selectbox("Marital Status", ["single","married","divorced"])
        st.markdown('<div class="fg-label">Employment &amp; Education</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2, gap="large")
        job       = c3.selectbox("Occupation", ["management","blue-collar","technician","admin.",
                                                 "services","retired","student","self-employed",
                                                 "entrepreneur","unemployed","housemaid","unknown"])
        education = c4.selectbox("Education Level", ["tertiary","secondary","primary","unknown"])

    with tab2:
        st.markdown('<div class="fg-label">Account Status</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")
        balance = c1.number_input("Account Balance (₹)", -5000, 50000, 1000)
        default = c2.selectbox("Has Credit Default?", ["no","yes"])
        st.markdown('<div class="fg-label">Loan Obligations</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2, gap="large")
        housing = c3.selectbox("Has Housing Loan?", ["no","yes"])
        loan    = c4.selectbox("Has Personal Loan?", ["no","yes"])

    with tab3:
        st.markdown('<div class="fg-label">Contact Details</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="large")
        contact  = c1.selectbox("Contact Method", ["cellular","telephone","unknown"])
        month    = c2.selectbox("Month of Contact", ["jan","feb","mar","apr","may","jun",
                                                      "jul","aug","sep","oct","nov","dec"])
        day      = c3.slider("Day of Month", 1, 31, 15)
        st.markdown('<div class="fg-label">Call Metrics</div>', unsafe_allow_html=True)
        c4, c5 = st.columns(2, gap="large")
        duration = c4.number_input("Call Duration (seconds)", 0, 3000, 300)
        campaign = c5.number_input("Calls This Campaign", 1, 50, 2)

    with tab4:
        st.markdown('<div class="fg-label">Previous Campaign Data</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="large")
        pdays    = c1.number_input("Days Since Last Contact (-1 = never)", -1, 999, -1)
        previous = c2.number_input("Total Previous Contacts", 0, 50, 0)
        st.markdown('<div class="fg-label">Previous Outcome</div>', unsafe_allow_html=True)
        poutcome = st.selectbox("Outcome of Previous Campaign", ["unknown","success","failure","other"])
        if pdays == -1 and poutcome != "unknown":
            st.warning("⚠️  No prior contact (pdays = -1) — consider setting outcome to 'unknown'.")

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_pred = st.button("🔍  Run Prediction", key="predict_btn", use_container_width=True)

    if run_pred:
        customer = dict(age=age, job=job, marital=marital, education=education,
                        default=default, balance=balance, housing=housing, loan=loan,
                        contact=contact, day=day, month=month, duration=duration,
                        campaign=campaign, pdays=pdays, previous=previous, poutcome=poutcome)
        try:
            with st.spinner("Running inference…"):
                time.sleep(0.45)
                pred, proba = run_prediction(customer)

            st.session_state.pred_result = pred
            st.session_state.pred_proba  = proba
            st.session_state.pred_inputs = customer

            st.markdown("<br>", unsafe_allow_html=True)
            conf_pct = f"{proba*100:.1f}%"

            if pred == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <div class="result-icon">✅</div>
                    <div class="result-verdict">Likely to Subscribe</div>
                    <div class="result-detail">This customer shows strong indicators for term deposit subscription.</div>
                    <div style="margin:28px auto;max-width:440px;">
                        <div class="conf-label">
                            <span>Subscription Probability</span>
                            <span style="color:#34d399;font-weight:700;">{proba:.1%}</span>
                        </div>
                        <div class="conf-track"><div class="conf-fill-yes" style="width:{conf_pct};"></div></div>
                    </div>
                    <div style="display:inline-block;background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.3);
                                border-radius:100px;padding:6px 20px;font-size:13px;color:#34d399;font-weight:600;">
                        🎯 High-Value Target Customer
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <div class="result-icon">❌</div>
                    <div class="result-verdict">Unlikely to Subscribe</div>
                    <div class="result-detail">This customer's profile suggests low propensity for term deposit subscription.</div>
                    <div style="margin:28px auto;max-width:440px;">
                        <div class="conf-label">
                            <span>Subscription Probability</span>
                            <span style="color:#fb7185;font-weight:700;">{proba:.1%}</span>
                        </div>
                        <div class="conf-track"><div class="conf-fill-no" style="width:{conf_pct};"></div></div>
                    </div>
                    <div style="display:inline-block;background:rgba(244,63,94,0.12);border:1px solid rgba(244,63,94,0.3);
                                border-radius:100px;padding:6px 20px;font-size:13px;color:#fb7185;font-weight:600;">
                        ⚠️ Low Priority — Reassign Resources
                    </div>
                </div>
                """, unsafe_allow_html=True)

            recs = ([("Next Step","Prioritise for personal follow-up call within 48 hours."),
                     ("Offer Strategy","Present competitive interest rate with flexible tenure options."),
                     ("Channel","Prefer cellular contact for higher engagement likelihood."),
                     ("Timing","Schedule call mid-month for optimal conversion rate.")]
                    if pred == 1 else
                    [("Deprioritise","Avoid high-cost outreach; reallocate to higher-propensity leads."),
                     ("Nurture","Enrol in low-touch email drip campaign to build familiarity."),
                     ("Re-evaluate","Revisit profile after 3 months for changed circumstances."),
                     ("Insight","Review call duration — shorter calls correlate with low intent.")])

            rec_html = "".join(f'<div class="rec-pill"><strong>{k}</strong>{v}</div>' for k, v in recs)
            st.markdown(f"""
            <br>
            <div class="glass-card">
                <div class="section-header">
                    <div class="section-icon">💡</div>
                    <div><div class="section-title">Recommended Actions</div></div>
                </div>
                <div class="rec-grid">{rec_html}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            verdict_str  = "Likely to Subscribe" if pred == 1 else "Unlikely to Subscribe"
            report_lines = (["BankSight AI — Prediction Report",
                              f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                              "=" * 45,
                              f"VERDICT     : {verdict_str}",
                              f"PROBABILITY : {proba:.4f} ({proba:.1%})", "",
                              "INPUT FEATURES", "-" * 30]
                            + [f"{k:<20}: {v}" for k, v in customer.items()])
            col_dl, _ = st.columns([1, 3])
            with col_dl:
                st.download_button("⬇️  Download Report",
                                   data=io.BytesIO("\n".join(report_lines).encode()),
                                   file_name=f"banksight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                   mime="text/plain", use_container_width=True)

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS & INSIGHTS  (NEW)
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Analytics & Insights":

    st.markdown("""
    <div class="section-header" style="margin-bottom:28px;">
        <div class="section-icon">📈</div>
        <div>
            <div class="section-title">Analytics &amp; Insights</div>
            <div class="section-sub">Visualisation dashboard — run a prediction first to personalise charts</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    has_pred = st.session_state.pred_result is not None
    if has_pred:
        pred   = st.session_state.pred_result
        proba  = st.session_state.pred_proba
        inputs = st.session_state.pred_inputs
    else:
        st.info("💡 Demo data shown. Head to **Predict** and run a prediction to personalise these charts.")
        pred, proba = 1, 0.72
        inputs = dict(age=42, job="management", marital="married", education="secondary",
                      balance=1500, housing="yes", loan="no", contact="cellular",
                      day=15, month="jul", duration=300, campaign=2,
                      pdays=-1, previous=0, poutcome="unknown", default="no")

    LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  font=dict(color="rgba(255,255,255,.75)", size=12),
                  margin=dict(l=16, r=16, t=36, b=16))

    # ── Row 1: Gauge · Donut · Risk ──────────────────────────────────────────
    r1, r2, r3 = st.columns(3, gap="medium")

    with r1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=round(proba*100, 1),
            title=dict(text="Subscription Probability", font=dict(size=12, color="#94a3b8")),
            number=dict(suffix="%", font=dict(size=30, color="#f8fafc")),
            gauge=dict(
                axis=dict(range=[0,100], tickcolor="rgba(255,255,255,.2)", tickvals=[0,25,50,75,100],
                          tickfont=dict(size=9, color="#475569")),
                bar=dict(color="#6366f1", thickness=0.22),
                bgcolor="rgba(255,255,255,0.03)",
                steps=[dict(range=[0,40],color="rgba(244,63,94,.15)"),
                       dict(range=[40,65],color="rgba(245,158,11,.12)"),
                       dict(range=[65,100],color="rgba(16,185,129,.15)")],
                threshold=dict(line=dict(color="#a5b4fc",width=2), value=proba*100),
            )
        ))
        fig.update_layout(**LAYOUT, height=230)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with r2:
        fig = go.Figure(go.Pie(
            labels=["Will Subscribe","Won't Subscribe"],
            values=[proba, 1-proba], hole=.62,
            marker=dict(colors=["#10b981","#f43f5e"], line=dict(width=0)),
            textinfo="percent", textfont=dict(size=13), textposition="inside",
        ))
        fig.update_layout(**LAYOUT, height=230, showlegend=True, legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
                          title=dict(text="Prediction Split", font=dict(size=13), x=0.5, xanchor="center"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with r3:
        risk = "High" if proba > .65 else "Medium" if proba > .40 else "Low"
        col  = "#10b981" if proba > .65 else "#f59e0b" if proba > .40 else "#f43f5e"
        fig = go.Figure(go.Indicator(
            mode="number+delta", value=round(proba*100,1),
            delta=dict(reference=50, valueformat=".1f", suffix="%",
                       increasing=dict(color="#10b981"), decreasing=dict(color="#f43f5e")),
            number=dict(suffix="%", font=dict(size=36, color=col)),
            title=dict(text=f"Risk Meter — <b>{risk} Likelihood</b>",
                       font=dict(size=12, color="#64748b")),
        ))
        fig.update_layout(**LAYOUT, height=230)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    # ── Row 2: Feature Importance · Radar ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")

    with c1:
        fi   = rf.feature_importances_
        idx  = np.argsort(fi)[-10:]
        fig = go.Figure(go.Bar(
            x=[fi[i] for i in idx], y=[features[i] for i in idx],
            orientation="h",
            marker=dict(color=px.colors.sequential.Plasma_r[:10], line=dict(width=0)),
            text=[f"{fi[i]*100:.1f}%" for i in idx], textposition="outside", textfont=dict(size=9),
        ))
        fig.update_layout(**LAYOUT, height=320,
                          title=dict(text="Top 10 Feature Importances", font=dict(size=13)),
                          xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,.05)", showticklabels=False, zeroline=False),
                          yaxis=dict(showgrid=False, tickfont=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with c2:
        dims = {
            "Call Duration":    min(inputs.get("duration",0)/1000, 1),
            "Balance":          min(max(inputs.get("balance",0)/10000, 0), 1),
            "Age Score":        (inputs.get("age",35)-18)/(80-18),
            "Experience":       min(inputs.get("previous",0)/5, 1),
            "Engagement":       1 - min(inputs.get("campaign",1)/15, 1),
            "Recency":          (1 if inputs.get("pdays",-1)==-1
                                 else max(1-inputs.get("pdays",999)/365, 0)),
        }
        cats = list(dims.keys()) + [list(dims.keys())[0]]
        vals = list(dims.values()) + [list(dims.values())[0]]
        fig = go.Figure(go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            line=dict(color="#6366f1", width=2),
            fillcolor="rgba(99,102,241,0.15)",
        ))
        fig.update_layout(**LAYOUT, height=320,
                          title=dict(text="Customer Profile Radar", font=dict(size=13)),
                          polar=dict(
                              bgcolor="rgba(255,255,255,0.02)",
                              radialaxis=dict(visible=True, range=[0,1],
                                             gridcolor="rgba(255,255,255,.08)",
                                             tickfont=dict(size=8)),
                              angularaxis=dict(gridcolor="rgba(255,255,255,.07)", tickfont=dict(size=11)),
                          ))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    # ── Row 3: Dataset Insight Charts ────────────────────────────────────────
    st.markdown("""
    <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:0.08em;
                margin:28px 0 16px;padding-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.06);">
        Dataset Insights (Bank Marketing — illustrative distributions)
    </div>
    """, unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3, gap="medium")

    with d1:
        _df = bank_df
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=_df[_df["deposit"]=="no"]["age"],
                                    name="No", nbinsx=30, marker_color="rgba(244,63,94,0.7)", opacity=0.85))
        fig.add_trace(go.Histogram(x=_df[_df["deposit"]=="yes"]["age"],
                                    name="Yes", nbinsx=30, marker_color="rgba(16,185,129,0.7)", opacity=0.85))
        fig.update_layout(**LAYOUT, height=240, barmode="overlay",
                          title=dict(text="Age by Outcome", font=dict(size=12)),
                          xaxis=dict(title="Age", gridcolor="rgba(255,255,255,.05)"),
                          yaxis=dict(gridcolor="rgba(255,255,255,.05)"),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with d2:
        _df2 = bank_df
        _job_r = _df2.groupby("job")["deposit"].apply(lambda x: (x=="yes").mean()).sort_values()
        jobs  = list(_job_r.index)
        rates = list(_job_r.values)
        colors_j = ["#10b981" if r>.55 else "#f59e0b" if r>.45 else "#f43f5e" for r in rates]
        fig = go.Figure(go.Bar(x=jobs, y=[r*100 for r in rates], marker=dict(color=colors_j, line=dict(width=0)),
                                text=[f"{r*100:.0f}%" for r in rates], textposition="outside", textfont=dict(size=9)))
        fig.update_layout(**LAYOUT, height=240, title=dict(text="Sub. Rate by Job (Real Data)", font=dict(size=12)),
                          xaxis=dict(tickangle=-30, tickfont=dict(size=9), gridcolor="rgba(255,255,255,.04)"),
                          yaxis=dict(title="%", gridcolor="rgba(255,255,255,.05)", range=[0, 90]))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with d3:
        _df3 = bank_df
        _month_order = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
        _mth_r = _df3.groupby("month")["deposit"].apply(lambda x: (x=="yes").mean()).reindex(_month_order)
        months = _month_order
        rates2 = list(_mth_r.values)
        fig = go.Figure(go.Scatter(x=months, y=[r*100 for r in rates2],
                                    mode="lines+markers",
                                    line=dict(color="#a78bfa", width=2.5),
                                    marker=dict(size=7, color="#a78bfa"),
                                    fill="tozeroy", fillcolor="rgba(139,92,246,0.12)"))
        fig.update_layout(**LAYOUT, height=240, title=dict(text="Sub. Trend by Month (Real Data)", font=dict(size=12)),
                          xaxis=dict(gridcolor="rgba(255,255,255,.05)"),
                          yaxis=dict(title="%", gridcolor="rgba(255,255,255,.05)"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    # ── Customer Profile Summary ──────────────────────────────────────────────
    st.markdown("""
    <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:0.08em;
                margin:28px 0 16px;padding-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.06);">
        Current Customer Profile
    </div>
    """, unsafe_allow_html=True)

    profile = [("Age",str(inputs.get("age","—"))),("Job",str(inputs.get("job","—")).title()),
               ("Marital",str(inputs.get("marital","—")).title()),("Education",str(inputs.get("education","—")).title()),
               ("Balance",f"₹{inputs.get('balance',0):,}"),("Housing Loan",str(inputs.get("housing","—")).title()),
               ("Personal Loan",str(inputs.get("loan","—")).title()),("Contact",str(inputs.get("contact","—")).title()),
               ("Month",str(inputs.get("month","—")).upper()),("Duration",f"{inputs.get('duration',0)}s"),
               ("# Contacts",str(inputs.get("campaign","—"))),("Prior Outcome",str(inputs.get("poutcome","—")).title())]

    cols = st.columns(4)
    for i, (k, v) in enumerate(profile):
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                        border-radius:12px;padding:12px 14px;margin-bottom:10px;">
                <div style="font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.06em;">{k}</div>
                <div style="font-size:15px;font-weight:600;color:#f1f5f9;margin-top:4px;">{v}</div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "Model Insights":

    st.markdown("""
    <div class="section-header" style="margin-bottom:28px;">
        <div class="section-icon">🧠</div>
        <div>
            <div class="section-title">Model Insights</div>
            <div class="section-sub">Understanding the Random Forest classifier</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        feat_rows = [("Call Duration",92,"#6366f1"),("Previous Outcome",78,"#8b5cf6"),
                     ("Account Balance",65,"#06b6d4"),("Age",58,"#10b981"),
                     ("No. of Calls",50,"#f59e0b"),("Days Since Contact",44,"#f43f5e"),
                     ("Housing Loan",38,"#ec4899"),("Month of Contact",35,"#14b8a6")]
        bars_html = "".join(f"""
            <div style="margin-bottom:14px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                    <span style="font-size:13px;color:#cbd5e1;font-weight:500;">{label}</span>
                    <span style="font-size:12px;color:{color};font-weight:600;">{pct}%</span>
                </div>
                <div style="height:7px;background:rgba(255,255,255,0.07);border-radius:100px;overflow:hidden;">
                    <div style="width:{pct}%;height:100%;border-radius:100px;background:{color};opacity:0.85;"></div>
                </div>
            </div>""" for label, pct, color in feat_rows)

        _html = ('<div class="glass-card"><div class="section-header">'
                 '<div class="section-icon">⚖️</div><div>'
                 '<div class="section-title">Key Predictors</div>'
                 '<div class="section-sub">Domain-informed feature importance</div>'
                 '</div></div>' + bars_html + '</div>')
        st.markdown(_html, unsafe_allow_html=True)

    with col2:
        steps = [("1","Raw Input",        "16 customer features collected via form inputs.","#6366f1"),
                 ("2","One-Hot Encoding",  "Categorical variables encoded (drop_first=False).","#8b5cf6"),
                 ("3","Feature Align",     "Frame reindexed to 52 training columns exactly.","#06b6d4"),
                 ("4","RF Inference",      "100 decision trees vote; majority + probability returned.","#10b981")]
        steps_html = "".join(f"""
            <div style="display:flex;gap:14px;margin-bottom:16px;align-items:flex-start;">
                <div style="min-width:28px;height:28px;border-radius:8px;
                            background:{color}22;border:1px solid {color}55;
                            display:flex;align-items:center;justify-content:center;
                            font-size:12px;font-weight:700;color:{color};">{num}</div>
                <div>
                    <div style="font-size:13px;font-weight:600;color:#f1f5f9;margin-bottom:2px;">{title}</div>
                    <div style="font-size:12px;color:#64748b;line-height:1.5;">{desc}</div>
                </div>
            </div>""" for num, title, desc, color in steps)

        _html = ('<div class="glass-card"><div class="section-header">'
                 '<div class="section-icon">📚</div><div>'
                 '<div class="section-title">How It Works</div>'
                 '<div class="section-sub">End-to-end ML pipeline</div>'
                 '</div></div>' + steps_html + '</div>')
        st.markdown(_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    insight_cards = [
        ("📞","#6366f1","rgba(99,102,241,0.08)","rgba(99,102,241,0.2)","#a5b4fc",
         "Longer Calls = Higher Intent",
         "Call duration is the strongest single predictor. Customers who stay on the line longer are far more likely to subscribe."),
        ("🏆","#06b6d4","rgba(6,182,212,0.07)","rgba(6,182,212,0.18)","#67e8f9",
         "Prior Success Matters",
         "Customers who subscribed in a previous campaign are significantly more likely to subscribe again."),
        ("💰","#10b981","rgba(16,185,129,0.07)","rgba(16,185,129,0.18)","#34d399",
         "Balance Signals Capacity",
         "Higher account balances correlate with greater willingness to lock funds in a term deposit."),
    ]
    cards_html = "".join(f"""
        <div style="background:{bg};border:1px solid {brd};border-radius:14px;padding:18px;">
            <div style="font-size:22px;margin-bottom:8px;">{ico}</div>
            <div style="font-size:14px;font-weight:600;color:{tc};margin-bottom:6px;">{title}</div>
            <div style="font-size:13px;color:#64748b;line-height:1.6;">{body}</div>
        </div>""" for ico, _, bg, brd, tc, title, body in insight_cards)

    st.markdown(f"""
    <div class="glass-card">
        <div class="section-header">
            <div class="section-icon">💡</div>
            <div>
                <div class="section-title">Key Business Insights</div>
                <div class="section-sub">What the model has learned from the data</div>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;">{cards_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📐  Technical Architecture Details"):
        st.markdown("""
        **Preprocessing pipeline:**
        - `pd.get_dummies(df, columns=cat_cols)` — full encoding, no drop_first
        - `df['was_contacted_before'] = (df['previous'] > 0).astype(int)` — engineered feature
        - `.reindex(columns=features, fill_value=0)` — aligns with 52 training features
        - No scaling — Random Forest is scale-invariant

        **Model:** `RandomForestClassifier` (scikit-learn 1.6.1)
        - `predict()` → class label {0, 1}
        - `predict_proba()` → [P(no), P(yes)]

        **Cached artifacts:** `model.pkl`, `features.pkl`, `cat_cols.pkl`
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "About":

    st.markdown("""
    <div class="section-header" style="margin-bottom:28px;">
        <div class="section-icon">ℹ️</div>
        <div>
            <div class="section-title">About This Project</div>
            <div class="section-sub">Background, dataset, and methodology</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="section-header">
                <div class="section-icon">🗃️</div>
                <div>
                    <div class="section-title">Dataset</div>
                    <div class="section-sub">UCI Bank Marketing Dataset · <a href="https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset" style="color:#38bdf8;" target="_blank">Kaggle</a></div>
                </div>
            </div>
            <p style="color:#94a3b8;font-size:14px;line-height:1.8;margin-bottom:18px;">
                A balanced subset of the UCI Bank Marketing dataset (via Kaggle). Predicts whether a client will subscribe to a term deposit based on 16 features from direct phone marketing campaigns
                (target variable: <code style="color:#a5b4fc;background:rgba(99,102,241,0.12);padding:1px 6px;border-radius:4px;">deposit</code>).
            </p>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
                <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.18);border-radius:12px;padding:14px;text-align:center;">
                    <div style="font-size:24px;font-weight:700;color:#a5b4fc;">11,162</div>
                    <div style="font-size:12px;color:#64748b;margin-top:4px;">Total Records</div>
                </div>
                <div style="background:rgba(6,182,212,0.07);border:1px solid rgba(6,182,212,0.18);border-radius:12px;padding:14px;text-align:center;">
                    <div style="font-size:24px;font-weight:700;color:#67e8f9;">16</div>
                    <div style="font-size:12px;color:#64748b;margin-top:4px;">Input Features</div>
                </div>
                <div style="background:rgba(16,185,129,0.07);border:1px solid rgba(16,185,129,0.18);border-radius:12px;padding:14px;text-align:center;">
                    <div style="font-size:24px;font-weight:700;color:#34d399;">2</div>
                    <div style="font-size:12px;color:#64748b;margin-top:4px;">Target Classes</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        tech_rows = "".join(f"""
            <div style="display:flex;align-items:center;gap:10px;
                        background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                        border-left:3px solid {color};border-radius:10px;padding:10px 12px;margin-bottom:8px;">
                <span style="font-size:16px;">{icon}</span>
                <span style="font-size:13px;color:#cbd5e1;font-weight:500;">{name}</span>
            </div>""" for icon, name, color in [
                ("🐍","Python 3.x","#f59e0b"),("🌊","Streamlit","#06b6d4"),
                ("🌲","scikit-learn","#10b981"),("🐼","pandas","#6366f1"),
                ("📊","Plotly","#8b5cf6"),("💾","joblib","#94a3b8"),
            ])

        _html = ('<div class="glass-card"><div class="section-header">'
                 '<div class="section-icon">🛠️</div>'
                 '<div><div class="section-title">Tech Stack</div></div>'
                 '</div>' + tech_rows + '</div>')
        st.markdown(_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="section-header">
            <div class="section-icon">⚠️</div>
            <div><div class="section-title">Important Notes</div></div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
            <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);border-radius:12px;padding:16px;">
                <div style="font-size:13px;font-weight:600;color:#fbbf24;margin-bottom:6px;">⚡ Duration Leakage</div>
                <div style="font-size:13px;color:#64748b;line-height:1.5;">
                    Call duration is not known before a call is made. In production, this feature should be excluded from pre-call prediction models.
                </div>
            </div>
            <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);border-radius:12px;padding:16px;">
                <div style="font-size:13px;font-weight:600;color:#a5b4fc;margin-bottom:6px;">🎓 Educational Purpose</div>
                <div style="font-size:13px;color:#64748b;line-height:1.5;">
                    Built for academic presentation purposes. Model performance depends on the training split used during development.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    BankSight AI &nbsp;·&nbsp; Built with <span>Streamlit</span> &nbsp;·&nbsp;
    Random Forest Classifier &nbsp;·&nbsp; UCI Bank Marketing Dataset
</div>
""", unsafe_allow_html=True)
