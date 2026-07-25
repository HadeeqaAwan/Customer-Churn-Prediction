import pandas as pd

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Original Shape:", df.shape)

# -----------------------------
# Convert TotalCharges to Numeric
# -----------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# -----------------------------
# Remove Missing Values
# -----------------------------
df.dropna(inplace=True)

print("Shape after removing missing values:", df.shape)

# -----------------------------
# Select Important Features Only
# -----------------------------
selected_features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "tenure",
    "InternetService",
    "Contract",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]

df = df[selected_features]

# -----------------------------
# Encode Target Variable
# -----------------------------
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# -----------------------------
# One-Hot Encode Categorical Features
# -----------------------------
categorical_cols = [
    "gender",
    "Partner",
    "InternetService",
    "Contract"
]

df = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)

print("\nColumns after Encoding:")
print(df.columns.tolist())

print("\nFinal Shape:", df.shape)

print("\nFirst Five Rows:")
print(df.head())

# -----------------------------
# Save Preprocessed Dataset
# -----------------------------
df.to_csv(
    "data/preprocessed_data.csv",
    index=False
)

print("\nPreprocessed dataset saved successfully!")