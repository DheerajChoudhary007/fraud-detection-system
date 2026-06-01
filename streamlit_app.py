# streamlit_app.py

import streamlit as st
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ─────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────

st.markdown("""
<style>
    .fraud-alert {
        background-color: #ff4b4b;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .safe-alert {
        background-color: #00c853;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #1e1e2e;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────
# API CONFIGURATION
# ─────────────────────────────────────

# Change this to your Render URL
API_URL = "https://fraud-detection-api-h8g9.onrender.com"  

# ─────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=30
        )
        return response.json()
    except:
        return None


def get_model_info():
    """Get model performance info"""
    try:
        response = requests.get(
            f"{API_URL}/model-info",
            timeout=30
        )
        return response.json()
    except:
        return None


def predict_transaction(transaction):
    """Send transaction to API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=transaction,
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def create_gauge_chart(probability):
    """Create fraud probability gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fraud Probability %",
               'font': {'size': 20}},
        delta={'reference': 50},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1
            },
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 30],
                 'color': '#00c853'},
                {'range': [30, 70],
                 'color': '#ffd600'},
                {'range': [70, 100],
                 'color': '#ff4b4b'}
            ],
            'threshold': {
                'line': {'color': "black",
                         'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    return fig


def create_shap_chart(explanation):
    """Create SHAP contribution chart"""
    fraud_signals = explanation.get(
        'top_fraud_signals', []
    )
    legit_signals = explanation.get(
        'top_legit_signals', []
    )

    features = []
    values = []
    colors = []

    for sig in fraud_signals:
        features.append(sig['feature'])
        values.append(sig['impact'])
        colors.append('#ff4b4b')

    for sig in legit_signals:
        features.append(sig['feature'])
        values.append(sig['impact'])
        colors.append('#00c853')

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition='outside'
    ))

    fig.update_layout(
        title="SHAP Feature Contributions",
        xaxis_title="Impact on Fraud Probability",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)'
        )
    )
    return fig

# ─────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────

with st.sidebar:
    st.title("🔍 Fraud Detection")
    st.markdown("---")

    # API Status
    st.subheader("API Status")
    health = check_api_health()

    if health:
        st.success("✅ API Online")
        st.metric("AUC-ROC",
                  health.get('auc_roc', 'N/A'))
        st.metric("Threshold",
                  health.get('threshold', 0.5))
    else:
        st.error("❌ API Offline")
        st.warning("Starting up... wait 30 secs")

    st.markdown("---")

    # Model Info
    st.subheader("Model Info")
    model_info = get_model_info()

    if model_info:
        st.metric("Model",
                  model_info.get('model_type',
                                 'XGBoost'))
        st.metric("Precision",
                  model_info.get('fraud_precision',
                                 'N/A'))
        st.metric("Recall",
                  model_info.get('fraud_recall',
                                 'N/A'))
        st.metric("F1 Score",
                  model_info.get('fraud_f1', 'N/A'))

    st.markdown("---")
    st.caption("Built with XGBoost + SHAP")
    st.caption("Deployed on Render")

# ─────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────

st.title("🔍 Real-Time Fraud Detection System")
st.markdown(
    "Enter transaction details to check "
    "if it's fraudulent"
)
st.markdown("---")

# ─────────────────────────────────────
# TABS
# ─────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🔍 Check Transaction",
    "📊 Model Performance",
    "ℹ️ About"
])

# ─────────────────────────────────────
# TAB 1: CHECK TRANSACTION
# ─────────────────────────────────────

with tab1:
    st.subheader("Transaction Details")

    # Use sample transactions
    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        if st.button("📋 Load Sample Legit",
                     use_container_width=True):
            st.session_state['sample'] = 'legit'
            # Directly set all values in session state

            for key, val in SAMPLE_LEGIT.items():
                st.session_state[f'input_{key}'] = float(val)
            st.rerun()

    with col_btn2:
        if st.button("🚨 Load Sample Fraud",
                     use_container_width=True):
            st.session_state['sample'] = 'fraud'
            # Directly set all values in session state
            for key, val in SAMPLE_FRAUD.items():
                st.session_state[f'input_{key}'] = float(val)
            st.rerun()

    with col_btn3:
        if st.button("🗑️ Clear Form",
                     use_container_width=True):
            st.session_state['sample'] = 'clear'
            # Clear all values
            for key in SAMPLE_LEGIT.keys():
                st.session_state[f'input_{key}'] = 0.0
            st.rerun()

    st.markdown("---")

    # Sample data
    SAMPLE_LEGIT = {
        "Time": 50000.0, "Amount": 50.0,
        "V1": 1.1918, "V2": 0.2661,
        "V3": 0.1665, "V4": 0.4481,
        "V5": 0.0600, "V6": -0.0824,
        "V7": -0.0788, "V8": 0.0851,
        "V9": -0.2552, "V10": -0.1669,
        "V11": 1.6127, "V12": 1.0650,
        "V13": 0.4890, "V14": -0.1437,
        "V15": 0.6355, "V16": 0.4639,
        "V17": -0.1140, "V18": -0.1837,
        "V19": -0.1459, "V20": -0.0691,
        "V21": -0.2257, "V22": -0.6386,
        "V23": -0.0801, "V24": 0.4173,
        "V25": 0.4126, "V26": 0.3943,
        "V27": 0.1008, "V28": 0.0579
    }

    SAMPLE_FRAUD = {
        "Time": 10000.0, "Amount": 1.0,
        "V1": -3.0435, "V2": -3.1572,
        "V3": 1.0887, "V4": 2.2886,
        "V5": 1.3598, "V6": -1.0197,
        "V7": -0.7238, "V8": 0.1654,
        "V9": -0.4516, "V10": -2.5579,
        "V11": 1.6420, "V12": -2.5921,
        "V13": 0.5202, "V14": -3.4742,
        "V15": 0.9434, "V16": -2.2246,
        "V17": -3.5495, "V18": -0.7593,
        "V19": 0.4126, "V20": 0.0610,
        "V21": 0.6603, "V22": 0.5543,
        "V23": -0.2060, "V24": -0.2516,
        "V25": -0.5329, "V26": 0.0324,
        "V27": 0.2432, "V28": 0.0977
    }

    # Determine default values
    defaults = {}
    for key in SAMPLE_LEGIT.keys():
        defaults[key] = st.session_state.get(
            f'input_{key}', 0.0
        )

    # Input form
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Basic Info")
        time_val = st.number_input(
            "Time (seconds)",
            value=float(defaults.get('Time', 0.0)),
            help="Seconds elapsed since first transaction",
            key="display_time"
        )
        amount_val = st.number_input(
            "Amount (€)",
            value=float(defaults.get('Amount', 0.0)),
            min_value=0.0,
            help="Transaction amount",
            key="display_amount"
        )

    with col2:
        st.subheader("Quick Stats")
        hour = int((time_val // 3600) % 24)
        st.metric("Hour of Day", f"{hour}:00")
        st.metric("Amount Category",
                  "High" if amount_val > 1000
                  else "Medium" if amount_val > 100
                  else "Low")
        st.metric("Time Period",
                  "Night 🌙" if hour < 6 or hour > 22
                  else "Day ☀️")

    # V Features
    st.subheader("PCA Features (V1-V28)")
    st.caption(
        "These are anonymized transaction "
        "features from PCA transformation"
    )

    # Create input grid
    v_values = {}
    cols = st.columns(7)

    for i in range(1, 29):
        col_idx = (i-1) % 7
        with cols[col_idx]:
            v_values[f'V{i}'] = st.number_input(
                f"V{i}",
                value=float(
                    defaults.get(f'V{i}', 0.0)
                ),
                format="%.4f",
                key=f"display_v{i}"
            )

    st.markdown("---")

    # Predict button
    if st.button("🔍 Check Transaction",
                 use_container_width=True,
                 type="primary"):

        # Build transaction
        transaction = {
            "Time": time_val,
            "Amount": amount_val,
            **v_values
        }

        # Call API
        with st.spinner(
            "Analyzing transaction... "
            "(may take 30s if API sleeping)"
        ):
            result = predict_transaction(
                transaction
            )

        if "error" in result:
            st.error(f"API Error: {result['error']}")
        else:
            st.markdown("---")
            st.subheader("🎯 Prediction Result")

            # Result columns
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                # Alert box
                if result['is_fraud']:
                    st.markdown(
                        '<div class="fraud-alert">'
                        '🚨 FRAUD DETECTED!</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="safe-alert">'
                        '✅ LEGITIMATE TRANSACTION</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # Metrics
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(
                        "Probability",
                        f"{result['fraud_probability']:.4f}"
                    )
                with m2:
                    st.metric(
                        "Risk Level",
                        result['risk_level']
                    )
                with m3:
                    st.metric(
                        "Threshold",
                        result['threshold_used']
                    )

                # Gauge chart
                gauge = create_gauge_chart(
                    result['fraud_probability']
                )
                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )

            with res_col2:
                # SHAP explanation
                if 'explanation' in result:
                    st.subheader("🔍 Why This Decision?")

                    explanation = result['explanation']

                    # SHAP chart
                    shap_chart = create_shap_chart(
                        explanation
                    )
                    st.plotly_chart(
                        shap_chart,
                        use_container_width=True
                    )

                    # Fraud signals
                    st.markdown("**🚨 Fraud Signals:**")
                    for sig in explanation[
                        'top_fraud_signals'
                    ]:
                        st.error(
                            f"**{sig['feature']}** = "
                            f"{sig['value']:.4f} → "
                            f"impact: +{sig['impact']:.4f}"
                        )

                    # Legit signals
                    st.markdown("**✅ Legit Signals:**")
                    for sig in explanation[
                        'top_legit_signals'
                    ]:
                        st.success(
                            f"**{sig['feature']}** = "
                            f"{sig['value']:.4f} → "
                            f"impact: {sig['impact']:.4f}"
                        )

# ─────────────────────────────────────
# TAB 2: MODEL PERFORMANCE
# ─────────────────────────────────────

with tab2:
    st.subheader("📊 Model Performance Metrics")

    model_info = get_model_info()

    if model_info:
        # Metrics row
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "AUC-ROC",
                model_info.get('auc_roc', 'N/A'),
                help="Overall model quality"
            )
        with c2:
            st.metric(
                "Precision",
                model_info.get('fraud_precision',
                               'N/A'),
                help="Of flagged frauds, how many real?"
            )
        with c3:
            st.metric(
                "Recall",
                model_info.get('fraud_recall', 'N/A'),
                help="Of real frauds, how many caught?"
            )
        with c4:
            st.metric(
                "F1 Score",
                model_info.get('fraud_f1', 'N/A'),
                help="Balance of precision and recall"
            )

        st.markdown("---")

        # Model details
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model Details")
            details = {
                "Algorithm": model_info.get(
                    'model_type', 'XGBoost'
                ),
                "Version": model_info.get(
                    'version', '1.0.0'
                ),
                "Training Data": model_info.get(
                    'training_data',
                    'Original (no SMOTE)'
                ),
                "Total Features": model_info.get(
                    'total_features', 31
                ),
                "Threshold": model_info.get(
                    'threshold', 0.5
                )
            }
            for key, val in details.items():
                st.write(f"**{key}:** {val}")

        with col2:
            st.subheader("Performance Chart")
            metrics_data = {
                'Metric': ['AUC-ROC', 'Precision',
                           'Recall', 'F1'],
                'Score': [
                    model_info.get('auc_roc', 0),
                    model_info.get('fraud_precision',
                                   0),
                    model_info.get('fraud_recall', 0),
                    model_info.get('fraud_f1', 0)
                ]
            }
            df_metrics = pd.DataFrame(metrics_data)

            fig = px.bar(
                df_metrics,
                x='Metric',
                y='Score',
                color='Score',
                color_continuous_scale='RdYlGn',
                range_y=[0, 1],
                title="Model Metrics"
            )
            fig.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            st.plotly_chart(
                fig,
                use_container_width=True
            )
    else:
        st.warning("Could not load model info")

# ─────────────────────────────────────
# TAB 3: ABOUT
# ─────────────────────────────────────

with tab3:
    st.subheader("ℹ️ About This Project")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🔍 Real-Time Fraud Detection System

        An end-to-end ML system that detects
        fraudulent financial transactions in
        real-time using XGBoost and SHAP.

        ### 🛠️ Tech Stack
        - **ML Model**: XGBoost
        - **Explainability**: SHAP
        - **API**: FastAPI
        - **Dashboard**: Streamlit
        - **Deployment**: Render
        - **Container**: Docker

        ### 📊 Dataset
        - 284,807 transactions
        - 0.17% fraud rate
        - 30 PCA features
        - Source: Kaggle MLG-ULB
        """)

    with col2:
        st.markdown("""
        ### 📈 Model Performance
        | Metric | Score |
        |--------|-------|
        | AUC-ROC | 0.9817 |
        | Precision | 0.8542 |
        | Recall | 0.8367 |
        | F1 Score | 0.8454 |
        | Threshold | 0.5 |

        ### 🎯 Key Decisions
        - XGBoost on original data
        - scale_pos_weight for imbalance
        - Threshold = 0.5 (high recall)
        - SHAP for explainability

        ### 🔗 Links
        - [GitHub Repository](#)
        - [Live API](/docs)
        - [Model Documentation](#)
        """)