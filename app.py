import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Set Page Config
st.set_page_config(
    page_title="Cookie Cats | Retention Intelligence",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Targeted CSS Overrides
st.markdown("""
    <style>
    /* Dark Gaming Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%) !important;
        background-attachment: fixed !important;
    }
    
    /* Headers & Body Text */
    h1, h2, h3, h4, h5, h6, p, label {
        color: #f8fafc !important;
        font-family: 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Glowing Title Header */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a855f7, #ec4899, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Metric Card Styling */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.75) !important;
        border: 1px solid rgba(168, 85, 247, 0.4) !important;
        border-radius: 12px !important;
        padding: 15px !important;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4) !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }
    
    /* FIX FOR DROPDOWN LISTS & SELECTBOX POPUPS */
    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border-color: #6366f1 !important;
    }

    ul[data-baseweb="menu"] {
        background-color: #1e293b !important;
    }

    li[data-baseweb="option"] {
        background-color: #1e293b !important;
        color: #ffffff !important;
    }

    li[data-baseweb="option"]:hover {
        background-color: #334155 !important;
    }
    
    /* COMPLETE FIX FOR FILE UPLOADER & DROPZONE */
    [data-testid="stFileUploader"] {
        background-color: transparent !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #1e293b !important;
        border: 2px dashed #a855f7 !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #f8fafc !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background-color: #334155 !important;
        border: 1px solid #a855f7 !important;
        color: #ffffff !important;
    }

    /* Tab bar customization */
    button[data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border-radius: 8px 8px 0px 0px !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    
    button[aria-selected="true"] {
        background: linear-gradient(90deg, #6366f1, #a855f7) !important;
        color: #ffffff !important;
    }
    
    /* Primary Predict Button */
    .stButton>button {
        background: linear-gradient(90deg, #ec4899 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Load Artifacts
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('lightgbm_model.pkl')
        thresh = joblib.load('best_threshold.pkl')
        return model, thresh
    except:
        return None, 0.629

model, best_threshold = load_artifacts()

# App Header
st.markdown('<h1 class="main-title">🎮 Cookie Cats Retention Intelligence</h1>', unsafe_allow_html=True)
st.markdown("<p style='color: #cbd5e1 !important;'>A/B Testing Analytics & Machine Learning Early Churn Prediction Engine</p>", unsafe_allow_html=True)

st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📊 A/B Experimentation", "🤖 Individual Risk Assessment", "📈 Dataset Analytics"])

with tab1:
    st.markdown("### 🏆 Level Progression & A/B Experiment Results")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Gate 30 (7-Day Retention)", "19.02%", "+0.82% boost")
    col2.metric("Gate 40 (7-Day Retention)", "18.20%", "-0.82% drop")
    col3.metric("A/B Decision Confidence", "99.90%", "Bootstrap Superiority")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    fig_ab = go.Figure(data=[
        go.Bar(name='1-Day Retention', x=['Gate 30', 'Gate 40'], y=[44.82, 44.23], marker_color='#818cf8'),
        go.Bar(name='7-Day Retention', x=['Gate 30', 'Gate 40'], y=[19.02, 18.20], marker_color='#c084fc')
    ])
    fig_ab.update_layout(
        title=dict(text="Retention Rates by Gate Variant (%)", font=dict(color='#f8fafc', size=18)),
        barmode='group',
        paper_bgcolor='rgba(15, 23, 42, 0.6)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        font=dict(color='#f8fafc'),
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
    )
    st.plotly_chart(fig_ab, use_container_width=True)

with tab2:
    st.markdown("### 🤖 Player Retention Risk Calculator")
    
    col_a, col_b = st.columns(2)
    with col_a:
        version_input = st.selectbox("Gate Variant", options=["gate_30", "gate_40"])
        sum_gamerounds = st.number_input("Game Rounds Played", min_value=0, max_value=5000, value=15)
        retention_1 = st.selectbox("Returned on Day 1?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
    with col_b:
        is_gate_30 = 1 if version_input == "gate_30" else 0
        played_zero = 1 if sum_gamerounds == 0 else 0
        log_gamerounds = np.log1p(sum_gamerounds)
        ret1_x_rounds = retention_1 * log_gamerounds
        
        input_data = pd.DataFrame([[
            is_gate_30, sum_gamerounds, log_gamerounds, retention_1, played_zero, ret1_x_rounds
        ]], columns=['is_gate_30', 'sum_gamerounds', 'log_gamerounds', 'retention_1', 'played_zero_rounds', 'ret1_x_rounds'])
        
        if st.button("Predict 7-Day Retention"):
            if model is not None:
                prob = model.predict_proba(input_data)[0][1]
            else:
                prob = 0.85 if sum_gamerounds > 30 and retention_1 == 1 else 0.15
                
            is_retained = prob >= best_threshold
            
            st.markdown("#### Prediction Outcome")
            if is_retained:
                st.success(f"**Status: Retained** (Retention Probability: {prob:.1%})")
            else:
                st.error(f"**Status: Churned** (Retention Probability: {prob:.1%})")
                
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Retention Probability Score", 'font': {'color': 'white', 'size': 16}},
                number={'suffix': "%", 'font': {'color': '#38bdf8'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "white"},
                    'bar': {'color': "#a855f7"},
                    'steps': [
                        {'range': [0, best_threshold * 100], 'color': "rgba(239, 68, 68, 0.4)"},
                        {'range': [best_threshold * 100, 100], 'color': "rgba(34, 197, 94, 0.4)"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': best_threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_gauge, use_container_width=True)

with tab3:
    st.markdown("### 📈 Exploratory Data Overview")
    uploaded_file = st.file_uploader("Upload CSV for dynamic analysis", type=['csv'])
    
    if uploaded_file is not None:
        df_user = pd.read_csv(uploaded_file)
        st.write("Dataset Sample:", df_user.head())
        
        fig_dist = px.histogram(
            df_user[df_user['sum_gamerounds'] < 100], 
            x='sum_gamerounds', 
            color='retention_7', 
            barmode='overlay',
            title="Game Rounds Distribution (<= 100 Rounds)"
        )
        fig_dist.update_layout(
            paper_bgcolor='rgba(15, 23, 42, 0.6)', 
            plot_bgcolor='rgba(15, 23, 42, 0.6)', 
            font=dict(color='white')
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("Upload `cookie_cats.csv` to render real-time interactive exploratory plots.")
            
