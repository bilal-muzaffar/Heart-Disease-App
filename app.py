import streamlit as st
import pandas as pd
import joblib

# ---------- Load saved model + preprocessing info ----------
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
numerical_columns = joblib.load("numerical_columns.pkl")
training_columns = joblib.load("training_columns.pkl")

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️")
st.title("❤️ Heart Disease Risk Predictor")
st.write(
    "Fill in the details below and click **Predict**. "
    "This is a demo model for educational purposes only — not medical advice."
)

# ---------- Raw user inputs (same fields as the original dataset) ----------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox(
        "Chest Pain Type",
        ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
    )
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 50, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
    thalch = st.number_input("Max Heart Rate Achieved", 50, 250, 150)

with col2:
    restecg = st.selectbox(
        "Resting ECG Results", ["normal", "st-t abnormality", "lv hypertrophy"]
    )
    exang = st.selectbox("Exercise Induced Angina", ["True", "False"])
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST", ["upsloping", "flat", "downsloping"])
    thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])
    dataset = st.selectbox(
        "Data Source Hospital", ["Cleveland", "Hungary", "Switzerland", "VA Long Beach"]
    )
    ca = st.number_input(
        "Major Vessels Colored by Fluoroscopy (0-3)", 0, 3, 0,
        help="From a fluoroscopy test. If unknown, leave at 0."
    )

if st.button("Predict", type="primary"):
    # ---------- Build a single-row dataframe matching raw training data shape ----------
    raw = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "dataset": dataset,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs == "True",
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang == "True",
        "oldpeak": oldpeak,
        "slope": slope,
        "thal": thal,
        "ca": ca,
    }])

    # ---------- Recreate the same engineered features used in training ----------
    raw["age_per_trestbps"] = raw["age"] / (raw["trestbps"] + 1)
    raw["chol_per_thalch"] = raw["chol"] / (raw["thalch"] + 1)
    raw["chol_per_age"] = raw["chol"] / (raw["age"] + 1)
    raw["trest_per_age"] = raw["trestbps"] / (raw["age"] + 1)
    raw["pulse_pressure"] = raw["trestbps"] - 80
    raw["high_bp"] = (raw["trestbps"] >= 140).astype(int)
    raw["high_chol"] = (raw["chol"] >= 240).astype(int)

    # ---------- One-hot encode categorical columns the same way ----------
    categorical_cols = ["sex", "dataset", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    encoded = pd.get_dummies(raw, columns=categorical_cols, drop_first=True)

    # ---------- Align columns to exactly match what the model was trained on ----------
    # Any column the model expects but wasn't created here gets filled with 0.
    # Any extra column not seen during training gets dropped.
    final_input = encoded.reindex(columns=training_columns, fill_value=0)

    # ---------- Scale numeric columns using the saved scaler ----------
    final_input[numerical_columns] = scaler.transform(final_input[numerical_columns])

    # ---------- Predict ----------
    prediction = model.predict(final_input)[0]
    proba = model.predict_proba(final_input)[0][1] if hasattr(model, "predict_proba") else None

    st.subheader("Result")
    if prediction == 1:
        st.error("⚠️ Higher likelihood of heart disease")
    else:
        st.success("✅ Lower likelihood of heart disease")

    if proba is not None:
        st.write(f"Model confidence (probability of disease): **{proba:.1%}**")

    st.caption(
        "Disclaimer: This tool is built for a learning project and is not a substitute "
        "for professional medical diagnosis."
    )
