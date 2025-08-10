import streamlit as st
import pandas as pd
import joblib 
from streamlit_lottie import st_lottie
import requests

# Load Lottie animation
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

heart_animation = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_4kx2q32n.json")

# Load model & assets
model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')
# App title
st.set_page_config(page_title="Heart Stroke Predictor", page_icon="❤️", layout="centered")

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>❤️ Heart Stroke Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict the likelihood of a heart stroke using your health data</p>", unsafe_allow_html=True)

# Animation
st_lottie(heart_animation, height=200)

st.sidebar.header('🩺 Input Your Health Data')

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 30)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

with col2:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("🔍 Predict", use_container_width=True):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease - Consult a doctor immediately.")
    else:
        st.success("✅ Low Risk of Heart Disease - Keep up the healthy lifestyle!")


