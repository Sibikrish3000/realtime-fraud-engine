import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as ob
import json
from datetime import datetime, time
import plotly.express as px
from src.features.constants import category_names, job_names

# 1. Configuration & Layout
st.set_page_config(page_title="PayShield Monitor", layout="wide", page_icon="🛡️")

# Custom CSS for FinTech Aesthetic
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-banner {
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .banner-safe {
        background-color: #28a745;
    }
    .banner-fraud {
        background-color: #dc3545;
    }
    .banner-shadow {
        background-color: #ffc107;
        color: #212529;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ PayShield-ML Analyst Workbench")
st.markdown("---")

# 2. Sidebar: Transaction Simulator
with st.sidebar:
    st.header("🛒 Transaction Simulator")
    
    with st.expander("👤 User Profile", expanded=True):
        user_id = st.text_input("User ID", value="u12345")
        job = st.selectbox("Job Title", options=sorted(job_names), index=sorted(job_names).index("Engineer, biomedical") if "Engineer, biomedical" in job_names else 0)
        dob = st.date_input("Date of Birth", value=datetime(1985, 3, 20))
        gender = st.radio("Gender", ["M", "F"], horizontal=True)

    with st.expander("💳 Transaction Details", expanded=True):
        amt = st.number_input("Amount ($)", min_value=0.01, value=150.0, step=10.0)
        category = st.selectbox("Category", options=sorted(category_names), index=sorted(category_names).index("grocery_pos") if "grocery_pos" in category_names else 0)
        trans_date = st.date_input("Transaction Date", value=datetime.now())
        trans_time = st.time_input("Transaction Time", value=time(14, 30))

    with st.expander("📍 Location Details", expanded=True):
        st.caption("User Coordinates")
        lat = st.number_input("User Lat", value=40.7128, format="%.4f")
        long = st.number_input("User Long", value=-74.0060, format="%.4f")
        
        st.caption("Merchant Coordinates")
        merch_lat = st.number_input("Merchant Lat", value=40.7200, format="%.4f")
        merch_long = st.number_input("Merchant Long", value=-74.0100, format="%.4f")

    with st.expander("🛠️ Advanced: Feature Overrides", expanded=False):
        st.caption("Force specific values for sensitivity analysis")
        override_trans_count = st.number_input("Count (24h)", min_value=0, value=0, help="Leave 0 to use real-time data")
        override_avg_spend = st.number_input("Avg Spend (24h)", min_value=0.0, value=0.0, help="Leave 0 to use real-time data")
        override_user_avg_all_time = st.number_input("User Avg (All Time)", min_value=0.0, value=0.0, help="Leave 0 to use calculated value")
        
    analyze_btn = st.button("🚀 Analyze Transaction", use_container_width=True, type="primary")

# 3. Main Dashboard Logic
if analyze_btn:
    # Construct Payload
    trans_dt = datetime.combine(trans_date, trans_time)
    payload = {
        "user_id": user_id,
        "trans_date_trans_time": trans_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "amt": amt,
        "lat": lat,
        "long": long,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
        "job": job,
        "category": category,
        "gender": gender,
        "dob": dob.strftime("%Y-%m-%d")
    }
    
    # Add overrides if set
    if override_trans_count > 0:
        payload["trans_count_24h"] = override_trans_count
    if override_avg_spend > 0:
        payload["avg_spend_24h"] = override_avg_spend
    if override_user_avg_all_time > 0:
        payload["user_avg_amt_all_time"] = override_user_avg_all_time

    try:
        # Step 1: Call API
        with st.spinner("Analyzing with XGBoost Engine..."):
            # Use docker-internal DNS or localhost depending on environment
            import os
            api_url = os.getenv("API_URL", "http://127.0.0.1:8000/v1/predict")
            response = requests.post(api_url, json=payload, timeout=5)
            
        if response.status_code == 200:
            res = response.json()
            score = res["risk_score"]
            decision = res["decision"]
            latency = res["latency_ms"]
            is_shadow = res.get("shadow_mode", False)
            features = res.get("features", {}) # Get real features used

            # Columns for high-level metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Decision Result", decision)
            with m2:
                st.metric("Risk Score", f"{score:.1f}/100")
            with m3:
                st.metric("Inference Latency", f"{latency:.2f}ms")

            # Decision Banner
            if decision == "BLOCK":
                st.markdown('<div class="status-banner banner-fraud">❌ FRAUD DETECTED - TRANSACTION BLOCKED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-banner banner-safe">✅ TRANSACTION APPROVED</div>', unsafe_allow_html=True)

            if is_shadow and decision == "BLOCK":
                 st.markdown('<div class="status-banner banner-shadow">⚠️ SHADOW MODE: Transaction allowed in simulation.</div>', unsafe_allow_html=True)

            # 4. Visualizations
            v1, v2 = st.columns([1, 1])

            with v1:
                st.subheader("🎯 Risk Gauge")
                fig = ob.Figure(ob.Indicator(
                    mode = "gauge+number",
                    value = score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence Score (%)"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#333"},
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(40, 167, 69, 0.3)"},
                            {'range': [50, 82], 'color': "rgba(255, 193, 7, 0.3)"},
                            {'range': [82, 100], 'color': "rgba(220, 53, 69, 0.3)"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 82
                        }
                    }
                ))
                fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with v2:
                st.subheader("📊 Feature Explainability (SHAP)")
                shap_data = res.get("shap_values", {})
                
                if shap_data:
                    # Create DataFrame from real SHAP values
                    shap_df = pd.DataFrame([
                        {"Feature": k, "Impact": v, "Abs_Impact": abs(v)}
                        for k, v in shap_data.items()
                    ]).sort_values("Abs_Impact", ascending=True)
                    
                    # Color based on positive/negative contribution
                    colors = ["#dc3545" if x > 0 else "#28a745" for x in shap_df["Impact"]]
                    
                    fig_shap = px.bar(
                        shap_df,
                        x="Impact",
                        y="Feature",
                        orientation="h",
                        title="Top Feature Contributions to Risk Score",
                        color="Impact",
                        color_continuous_scale=["#28a745", "#ffc107", "#dc3545"],
                        labels={"Impact": "SHAP Value (Impact on Prediction)"}
                    )
                    fig_shap.update_layout(
                        height=350,
                        margin=dict(l=20, r=20, t=50, b=20),
                        showlegend=False
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                    st.caption("🔴 Red = Increases fraud risk | 🟢 Green = Decreases fraud risk")
                else:
                    st.info("SHAP explainability not available. Enable it in API settings.")

            # 5. Internal Data State (Architectural Demo)
            st.markdown("---")
            d1, d2 = st.columns([1, 1])
            
            with d1:
                st.subheader("🗄️ Feature Store Payload")
                if features:
                    st.info("Real-time features used for inference (from Redis or Overrides):")
                    # Format for better readability
                    display_features = {
                        "Velocity (24h)": features.get("trans_count_24h"),
                        "Avg Spend (24h)": f"${features.get('avg_spend_24h', 0):.2f}",
                        "Current/Avg Ratio": f"{features.get('amt_to_avg_ratio_24h', 0):.2f}x",
                        "User Avg (All Time)": f"${features.get('user_avg_amt_all_time', 0):.2f}"
                    }
                    st.table(pd.DataFrame([display_features]))
                else:
                     st.warning("No feature data returned from API.")

            with d2:
                st.subheader("🔍 API RAW Response")
                with st.expander("View JSON", expanded=False):
                    st.json(res)

        else:
            st.error(f"API Error: Status {response.status_code}")
            st.json(response.json())

    except requests.exceptions.ConnectionError:
        st.error("🛑 Connection Failed: Could not connect to Inference API. Is the server running at http://127.0.0.1:8000?")
        st.info("Try running: `uv run uvicorn src.api.main:app --reload`")
    except Exception as e:
        st.error(f"⚠️ Unexpected Error: {str(e)}")

else:
    # Landing State
    st.info("👈 Fill in the details in the sidebar and click 'Analyze Transaction' to start.")
    
    # Hero/Demo Content
    c1, c2, c3 = st.columns(3)
    c1.markdown("### ⚡ Low Latency\nSub-50ms inference utilizing XGBoost and Redis.")
    c2.markdown("### 📋 Explainable\nSHAP integration for transparent fraud scoring.")
    c3.markdown("### 🧪 Shadow Mode\nSafe production testing of new model versions.")
