
import streamlit as st
import pandas as pd
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "artifacts" / "models"
METRICS_FILE = BASE / "artifacts" / "metrics.json"

st.set_page_config(
    page_title="Smart City AI Dashboard",
    page_icon="🏙️",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #f3f6fb;
}
[data-testid="stSidebar"] {
    background-color: #14213d;
}
[data-testid="stSidebar"] * {
    color: white;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🏙️ SMART CITY")
st.sidebar.caption("Intelligence Platform")
st.sidebar.divider()

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Dashboard Home",
        "Traffic Intelligence",
        "Smart Waste Management",
        "Citizen Complaints",
        "Energy Forecasting"
    ]
)

st.sidebar.divider()
st.sidebar.caption("AI Powered Smart City Project")

# Read metrics
if METRICS_FILE.exists():
    try:
        metrics = json.loads(
            METRICS_FILE.read_text(encoding="utf-8")
        )
    except Exception:
        metrics = {}
else:
    metrics = {}

# Home
if page == "Dashboard Home":

    st.title("🏙️ Smart City Intelligence Dashboard")
    st.write("Welcome to AI-powered city management system.")

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Modules", "04")
    c2.metric("ML Modules", "03")
    c3.metric("Deep Learning", "01")
    c4.metric("Platform", "Active")

    st.subheader("AI Intelligence Modules")

    col1, col2 = st.columns(2)

    with col1:
        st.info("🚦 Traffic Intelligence")
        st.write("Traffic monitoring and vehicle count prediction.")

        st.info("♻️ Smart Waste Management")
        st.write("Waste image classification into four categories.")

    with col2:
        st.info("📝 Citizen Complaints")
        st.write("Sentiment analysis and complaint classification.")

        st.info("⚡ Energy Forecasting")
        st.write("Electricity consumption prediction.")

    st.divider()
    st.subheader("Model Availability")

    model_files = {
        "Traffic": "traffic_model.joblib",
        "Waste": "waste_model.keras",
        "Citizen Sentiment": "sentiment_model.joblib",
        "Complaint Category": "complaint_category_model.joblib",
        "Energy": "energy_model.joblib"
    }

    for name, filename in model_files.items():
        if (MODEL_DIR / filename).exists():
            st.success(f"{name}: Model Available")
        else:
            st.warning(f"{name}: Model Not Found")

    st.subheader("Model Performance")

    if metrics:
        st.json(metrics)
    else:
        st.info("Train your models to display performance metrics.")

# Other module pages
elif page == "Traffic Intelligence":
    st.title("🚦 Traffic Intelligence")
    st.write("Vehicle count forecasting and traffic analysis.")
    st.info("Traffic prediction interface will be connected here.")

elif page == "Smart Waste Management":
    st.title("♻️ Smart Waste Management")
    st.write("Upload waste images for classification.")
    st.info("Waste image prediction interface will be connected here.")

elif page == "Citizen Complaints":
    st.title("📝 Citizen Sentiment & Complaints")
    st.write("Analyze citizen complaints using NLP.")
    st.info("Sentiment and category prediction interface will be connected here.")

elif page == "Energy Forecasting":
    st.title("⚡ Energy Consumption Forecasting")
    st.write("Predict future electricity consumption.")
    st.info("Energy forecasting interface will be connected here.")

st.divider()
st.caption("AI-Powered Smart City Intelligence Platform | Academic Project")
