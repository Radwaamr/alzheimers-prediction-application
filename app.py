import streamlit as st
import pandas as pd
import pickle

# Load trained LightGBM model + selected top 15 features
model = pickle.load(open("alz_lgbm_model.pkl", "rb"))
model_features = pickle.load(open("model_features.pkl", "rb"))  # يجب أن تحتوي على الـ 15 feature بالترتيب

st.title("🧠 Alzheimer’s Disease Prediction App")
st.write("Fill in the following inputs to predict the probability of Alzheimer's:")

# --- User Inputs (Top 15 Features Only) ---
ADL = st.number_input("ADL Score (0–10)", min_value=0, max_value=10, value=4)
FunctionalAssessment = st.number_input("Functional Assessment (1–10)", min_value=1, max_value=10, value=5)
MMSE = st.number_input("MMSE Score (0–30)", min_value=0, max_value=30, value=20)
DietQuality = st.number_input("Diet Quality (1–10)", min_value=1, max_value=10, value=5)
CholesterolTotal = st.number_input("Total Cholesterol", min_value=100, max_value=400, value=220)
CholesterolLDL = st.number_input("LDL Cholesterol", min_value=50, max_value=250, value=125)
BehavioralProblems = st.selectbox("Behavioral Problems", [0, 1])
SleepQuality = st.number_input("Sleep Quality (1–10)", min_value=1, max_value=10, value=7)
CholesterolTriglycerides = st.number_input("Triglycerides", min_value=50, max_value=500, value=230)
Age = st.number_input("Age", min_value=0, max_value=120, value=70)
MemoryComplaints = st.selectbox("Memory Complaints", [0, 1])
BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
PhysicalActivity = st.number_input("Physical Activity Level (1–10)", min_value=1, max_value=10, value=5)

# --- Derived Features ---
TG_HDL_Index = CholesterolTriglycerides / CholesterolLDL
Cholesterol_Ratio = CholesterolLDL / CholesterolTotal

# --- Create dataframe ---
user_data = {
    "ADL": st.number_input("ADL", min_value=0.0, max_value=10.0, value=4.0),
    "FunctionalAssessment": st.number_input("FunctionalAssessment", min_value=0.0, max_value=10.0, value=5.0),
    "MMSE": st.number_input("MMSE", min_value=0.0, max_value=30.0, value=15.0),
    "DietQuality": st.number_input("DietQuality", min_value=0.0, max_value=10.0, value=6.0),
    "CholesterolTotal": st.number_input("CholesterolTotal", min_value=100.0, max_value=400.0, value=220.0),
    "CholesterolLDL": st.number_input("CholesterolLDL", min_value=50.0, max_value=300.0, value=125.0),
    "BehavioralProblems": st.number_input("BehavioralProblems", min_value=0.0, max_value=10.0, value=0.0),
    "SleepQuality": st.number_input("SleepQuality", min_value=0.0, max_value=10.0, value=7.0),
    "CholesterolTriglycerides": st.number_input("CholesterolTriglycerides", min_value=50.0, max_value=500.0, value=230.0),
    "Age": st.number_input("Age", min_value=1, max_value=120, value=75),
    "MemoryComplaints": st.number_input("MemoryComplaints", min_value=0.0, max_value=10.0, value=1.0),
    "BMI": st.number_input("BMI", min_value=5.0, max_value=60.0, value=28.0),
    "PhysicalActivity": st.number_input("PhysicalActivity", min_value=0.0, max_value=10.0, value=2.0),
    "TG_HDL_Index": st.number_input("TG_HDL_Index", min_value=0.0, max_value=20.0, value=4.5),
    "Cholesterol_Ratio": st.number_input("Cholesterol_Ratio", min_value=1.0, max_value=500.0, value=5.0)
}


df_user = pd.DataFrame(user_data)
df_user = df_user[model_features]  # keep only top 15 features in correct order

# --- Prediction ---
if st.button("Predict Alzheimer’s"):
    prediction = model.predict(df_user)[0]
    prediction_proba = model.predict_proba(df_user)[0]

    # Mapping class to text
    class_mapping = {0: "No Alzheimer's detected", 1: "Alzheimer's detected"}
    pred_class_name = class_mapping[prediction]

    # Probabilities
    prob_no = prediction_proba[0]
    prob_yes = prediction_proba[1]

    # Display results
    st.write("=== Final Prediction ===")
    st.write(f"Prediction: {pred_class_name}")
    st.write(f"Probability:")
    st.write(f"  - No Alzheimer's: {prob_no*100:.2f}%")
    st.write(f"  - Alzheimer's: {prob_yes*100:.2f}%")

    # Interpretation
    if prediction == 1:
        st.error("Interpretation: The model predicts a high likelihood of Alzheimer's for this patient.")
    else:
        st.success("Interpretation: The model predicts a low likelihood of Alzheimer's for this patient.")
