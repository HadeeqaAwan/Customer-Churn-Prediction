import streamlit as st
import pandas as pd
import joblib

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# -----------------------
# Load Model & Scaler
# -----------------------
model = joblib.load("models/logistic_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# -----------------------
# Title
# -----------------------
st.title("📊 Customer Churn Prediction")

st.write(
    "Predict whether a telecom customer is likely to churn."
)

st.markdown("---")

# -----------------------
# User Inputs
# -----------------------

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior = st.selectbox(
    "Senior Citizen",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

partner = st.selectbox(
    "Partner",
    ["No", "Yes"]
)

tenure = st.slider(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12
)

internet = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)

contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

# -----------------------
# Prediction Button
# -----------------------
if st.button("🔍 Predict Churn"):

    # Create input dictionary
    input_data = {
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,

        "gender_Male": 1 if gender == "Male" else 0,

        "Partner_Yes": 1 if partner == "Yes" else 0,

        "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
        "InternetService_No": 1 if internet == "No" else 0,

        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Reorder columns to match training data
    input_df = input_df[
        [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "gender_Male",
            "Partner_Yes",
            "InternetService_Fiber optic",
            "InternetService_No",
            "Contract_One year",
            "Contract_Two year"
        ]
    ]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is NOT likely to Churn")

    st.write(f"### Prediction Confidence: **{probability*100:.2f}%**")

    st.progress(float(probability))

    st.markdown("---")

    st.subheader("📋 Customer Information")

    st.dataframe(input_df, use_container_width=True)