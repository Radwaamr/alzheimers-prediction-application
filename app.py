import streamlit as st
import pandas as pd
import pickle

# --- Load trained model & top 15 features ---
model = pickle.load(open("alz_lgbm_model.pkl", "rb"))
model_features = pickle.load(open("model_features.pkl", "rb"))

st.title("🧠 Alzheimer’s Disease Prediction App")
st.write("Fill in the following inputs to predict the probability of Alzheimer's:")

# =======================
# --- User Inputs ---
# =======================
st.header("Patient Data (Top 15 Features)")

ADL = st.number_input("ADL Score (0–10)", min_value=0, max_value=10, value=4)
FunctionalAssessment = st.number_input("Functional Assessment (1–10)", min_value=1, max_value=10, value=5)
MMSE = st.number_input("MMSE Score (0–30)", min_value=0, max_value=30, value=20)
DietQuality = st.number_input("Diet Quality (1–10)", min_value=1, max_value=10, value=5)
CholesterolTotal = st.number_input("Total Cholesterol", min_value=100, max_value=400, value=220)
CholesterolLDL = st.number_input("LDL Cholesterol", min_value=50, max_value=300, value=125)
BehavioralProblems = st.selectbox("Behavioral Problems", [0, 1])
SleepQuality = st.number_input("Sleep Quality (1–10)", min_value=1, max_value=10, value=7)
CholesterolTriglycerides = st.number_input("Triglycerides", min_value=50, max_value=500, value=230)
Age = st.number_input("Age", min_value=0, max_value=120, value=70)
MemoryComplaints = st.selectbox("Memory Complaints", [0, 1])
BMI = st.number_input("BMI", min_value=10, max_value=60, value=25)
PhysicalActivity = st.number_input("Physical Activity Level (1–10)", min_value=1, max_value=10, value=5)

# --- Derived features (computed automatically) ---
TG_HDL_Index = CholesterolTriglycerides / CholesterolLDL
Cholesterol_Ratio = CholesterolLDL / CholesterolTotal

# =======================
# --- Create dataframe ---
# =======================
user_data = {
    "ADL": ADL,
    "FunctionalAssessment": FunctionalAssessment,
    "MMSE": MMSE,
    "DietQuality": DietQuality,
    "CholesterolTotal": CholesterolTotal,
    "CholesterolLDL": CholesterolLDL,
    "BehavioralProblems": BehavioralProblems,
    "SleepQuality": SleepQuality,
    "CholesterolTriglycerides": CholesterolTriglycerides,
    "Age": Age,
    "MemoryComplaints": MemoryComplaints,
    "BMI": BMI,
    "PhysicalActivity": PhysicalActivity,
    "TG_HDL_Index": TG_HDL_Index,
    "Cholesterol_Ratio": Cholesterol_Ratio
}

df_user = pd.DataFrame([user_data])
df_user = df_user[model_features]  # ensure correct order of columns

# =======================
# --- Prediction ---
# =======================
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
