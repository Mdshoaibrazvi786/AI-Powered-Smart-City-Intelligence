
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from PIL import Image
import tensorflow as tf

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "artifacts" / "models"
METRICS_PATH = BASE / "artifacts" / "metrics.json"

st.set_page_config(
    page_title="AI Smart City Intelligence",
    page_icon="🏙️",
    layout="wide"
)

# ---------- Load saved metrics ----------
if METRICS_PATH.exists():
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        metrics = {}
else:
    metrics = {}

# ---------- Model loading ----------
@st.cache_resource
def load_joblib_model(filename):
    path = MODEL_DIR / filename
    if path.exists():
        return joblib.load(path)
    return None

@st.cache_resource
def load_waste_model():
    path = MODEL_DIR / "waste_model.keras"
    if path.exists():
        return tf.keras.models.load_model(path)
    return None

st.sidebar.title("🏙️ Smart City")
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Home",
        "Traffic Intelligence",
        "Smart Waste Management",
        "Citizen Complaints",
        "Energy Forecasting"
    ]
)

st.title("🏙️ AI-Powered Smart City Intelligence Platform")
st.caption("Machine Learning and Deep Learning based city analytics")

# ---------- HOME ----------
if page == "Dashboard Home":

    st.subheader("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("AI Modules", "4")
    col2.metric("Traffic", "ML")
    col3.metric("Waste", "CNN")
    col4.metric("Citizen & Energy", "NLP / ML")

    st.divider()

    st.subheader("Module Status")

    modules = [
        ("Traffic Intelligence", "traffic_model.joblib"),
        ("Smart Waste Management", "waste_model.keras"),
        ("Citizen Sentiment", "sentiment_model.joblib"),
        ("Energy Forecasting", "energy_model.joblib")
    ]

    for name, filename in modules:
        exists = (MODEL_DIR / filename).exists()
        if exists:
            st.success(f"{name}: Model available")
        else:
            st.warning(f"{name}: Model file not found")

    st.info(
        "Select a module from the sidebar to make predictions."
    )

# ---------- TRAFFIC ----------
elif page == "Traffic Intelligence":

    st.header("🚦 Traffic Intelligence")

    model = load_joblib_model("traffic_model.joblib")

    if model is None:
        st.error("Traffic model not found. Train Module 1 first.")
    else:
        st.write("Enter traffic information to predict vehicle count.")

        feature_names = list(
            getattr(model, "feature_names_in_", [])
        )

        if not feature_names:
            st.warning(
                "Traffic model feature names are unavailable. "
                "Check the Module 1 training notebook."
            )
        else:
            defaults = {
                "Junction": 1,
                "Hour": 14,
                "Day": 15,
                "Month": 10,
                "DayOfWeek": 2,
                "PreviousHourVehicles": 100,
                "Previous3HourAvg": 95
            }

            values = {}

            for feature in feature_names:
                default = defaults.get(feature, 0)
                values[feature] = st.number_input(
                    feature,
                    value=float(default)
                )

            if st.button("Predict Traffic"):
                input_df = pd.DataFrame(
                    [values],
                    columns=feature_names
                )
                prediction = model.predict(input_df)[0]

                st.metric(
                    "Predicted Vehicle Count",
                    f"{prediction:.0f}"
                )

# ---------- WASTE ----------
elif page == "Smart Waste Management":

    st.header("♻️ Smart Waste Management")

    model = load_waste_model()

    if model is None:
        st.error("Waste model not found. Train Module 2 first.")
    else:
        uploaded_file = st.file_uploader(
            "Upload a waste image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Waste Image", width=300)

            if st.button("Classify Waste"):
                image = image.resize((128, 128))
                array = np.array(image, dtype=np.float32) / 255.0
                array = np.expand_dims(array, axis=0)

                prediction = model.predict(array, verbose=0)[0]

                classes = [
                    "Hazardous",
                    "Non-Recyclable",
                    "Organic",
                    "Recyclable"
                ]

                index = int(np.argmax(prediction))

                st.success(f"Predicted Category: {classes[index]}")
                st.metric(
                    "Model Confidence",
                    f"{prediction[index]:.2%}"
                )

# ---------- CITIZEN COMPLAINTS ----------
elif page == "Citizen Complaints":

    st.header("📝 Citizen Sentiment & Complaints")

    sentiment_model = load_joblib_model("sentiment_model.joblib")
    category_model = load_joblib_model(
        "complaint_category_model.joblib"
    )

    if sentiment_model is None or category_model is None:
        st.error("Citizen complaint models not found. Train Module 3 first.")
    else:
        complaint = st.text_area(
            "Enter citizen complaint",
            placeholder="Example: Garbage has not been collected for three days."
        )

        if st.button("Analyze Complaint"):
            if not complaint.strip():
                st.warning("Please enter a complaint.")
            else:
                sentiment = sentiment_model.predict([complaint])[0]
                category = category_model.predict([complaint])[0]

                col1, col2 = st.columns(2)
                col1.metric("Sentiment", sentiment.title())
                col2.metric("Complaint Category", category.title())

                st.write("Complaint:", complaint)

# ---------- ENERGY ----------
elif page == "Energy Forecasting":

    st.header("⚡ Energy Consumption Forecasting")

    model = load_joblib_model("energy_model.joblib")

    if model is None:
        st.error("Energy model not found. Train Module 4 first.")
    else:
        features = [
            "Hour",
            "DayOfWeek",
            "Month",
            "IsWeekend",
            "Temperature",
            "PreviousHour",
            "Previous24Hours"
        ]

        defaults = {
            "Hour": 14,
            "DayOfWeek": 2,
            "Month": 10,
            "IsWeekend": 0,
            "Temperature": 30.0,
            "PreviousHour": 145.0,
            "Previous24Hours": 130.0
        }

        values = {}

        for feature in features:
            values[feature] = st.number_input(
                feature,
                value=float(defaults[feature])
            )

        if st.button("Forecast Energy"):
            input_df = pd.DataFrame([values], columns=features)
            prediction = model.predict(input_df)[0]

            st.metric(
                "Predicted Energy Consumption",
                f"{prediction:.2f} kWh"
            )

st.divider()
st.caption("AI-Powered Smart City Intelligence Platform | Academic Project")
