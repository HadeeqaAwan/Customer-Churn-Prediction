import pandas as pd

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("=" * 50)
print("DATA VALIDATION REPORT")
print("=" * 50)

# -----------------------
# Dataset Shape
# -----------------------
print(f"\nDataset Shape: {df.shape}")

# -----------------------
# Missing Values
# -----------------------
print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------
# Duplicate Records
# -----------------------
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

# -----------------------
# Data Types
# -----------------------
print("\nData Types:")
print(df.dtypes)

# -----------------------
# Required Columns
# -----------------------
required_columns = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]

missing_columns = []

for column in required_columns:
    if column not in df.columns:
        missing_columns.append(column)

if len(missing_columns) == 0:
    print("\n✅ All required columns are present.")
else:
    print("\n❌ Missing Columns:")
    print(missing_columns)

print("\n✅ Data Validation Completed Successfully!")