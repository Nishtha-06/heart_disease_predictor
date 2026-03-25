import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

# Load model & scaler
@st.cache_resource
def load():
    with open("heart_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler1.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load()

tab1, tab2 = st.tabs(["🔍 Predictor", "📊 Model Info"])

# ================= TAB 1 ================= #
with tab1:
    st.title("❤️ Heart Disease Risk Predictor")

    # -------- NORMAL INPUTS -------- #
    age = st.number_input("Age", 1, 120, 25)
    resting_bp = st.number_input("Resting Blood Pressure", value=120)
    cholesterol = st.number_input("Cholesterol", value=200)
    fasting_bs = st.selectbox("Fasting Blood Sugar >120", [0, 1])
    max_hr = st.number_input("Max Heart Rate", value=150)
    oldpeak = st.number_input("Oldpeak", value=1.0)

    sex = st.selectbox("Sex", ["F", "M"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    restecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    exang = st.selectbox("Exercise Angina", ["N", "Y"])
    slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    # -------- ONE HOT ENCODING -------- #
    def encode():
        return [
            age,
            resting_bp,
            cholesterol,
            fasting_bs,
            max_hr,
            oldpeak,

            1 if sex == "M" else 0,

            1 if chest_pain == "ATA" else 0,
            1 if chest_pain == "NAP" else 0,
            1 if chest_pain == "TA" else 0,

            1 if restecg == "Normal" else 0,
            1 if restecg == "ST" else 0,

            1 if exang == "Y" else 0,

            1 if slope == "Flat" else 0,
            1 if slope == "Up" else 0
        ]

    # -------- PREDICT -------- #
    if st.button("🔍 Predict"):
        try:
            input_data = np.array([encode()])

            input_scaled = scaler.transform(input_data)

            pred = model.predict(input_scaled)[0]

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_scaled)[0][1]
                st.write(f"📊 Risk Probability: **{prob*100:.2f}%**")

            if pred == 1:
                st.error("⚠️ High Risk")
            else:
                st.success("✅ Low Risk")

        except Exception as e:
            st.error(f"Error: {e}")

# ================= TAB 2 ================= #
with tab2:
    st.title("📊 Model Explanation")

    st.image("shap_summary.png")
    st.image("shap_waterfall.png")

try:
    with open("heart_model.pkl", "rb") as f:
        model = pickle.load(f)
except ModuleNotFoundError as e:
    print("Missing module:", e.name)