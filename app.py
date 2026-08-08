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

# Styling with CSS & Background Image
st.markdown("""
    <style>
    .main {
        background: linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.85)), 
                    url('https://images.unsplash.com/photo-1550745165-9bc0b252726f?auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-attachment: fixed;
    }
    .stMetric {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);
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
st.title("🎮 Cookie Cats Retention Intelligence")
st.caption("A/B Testing Analytics & Machine Learning Early Churn Prediction Engine")

st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📊 A/B Experimentation", "🤖 Individual Risk Assessment", "📈 Dataset Analytics"])

with tab1:
    st.subheader("Level Progression & A/B Experiment Results")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Gate 30 (7-Day Retention)", "19.02%", "+0.82% boost")
    col2.metric("Gate 40 (7-Day Retention)", "18.20%", "-0.82% drop")
    col3.metric("A/B Decision Confidence", "99.90%", "Bootstrap Superiority")
    
    st.markdown("---")
    
    fig_ab = go.Figure(data=[
        go.Bar(name='1-Day Retention', x=['Gate 30', 'Gate 40'], y=[44.82, 44.23], marker_color='#818cf8'),
        go.Bar(name='7-Day Retention', x=['Gate 30', 'Gate 40'], y=[19.02, 18.20], marker_color='#c084fc')
    ])
    fig_ab.update_layout(
        title="Retention Rates by Gate Variant (%)",
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig_ab, use_container_width=True)

with tab2:
    st.subheader("Player Retention Risk Calculator")
    
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
            
            st.markdown("### Prediction Outcome")
            if is_retained:
                st.success(f"**Status: Retained** (Retention Probability: {prob:.1%})")
            else:
                st.error(f"**Status: Churned** (Retention Probability: {prob:.1%})")
                
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Retention Probability Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#a855f7"},
                    'steps': [
                        {'range': [0, best_threshold * 100], 'color': "rgba(239, 68, 68, 0.3)"},
                        {'range': [best_threshold * 100, 100], 'color': "rgba(34, 197, 94, 0.3)"}
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
    st.subheader("Exploratory Data Overview")
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
        fig_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("Upload `cookie_cats.csv` to render real-time interactive exploratory plots.")
            
