import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Set plot style
sns.set_style("whitegrid")

# -----------------------------
# Basic Information
# -----------------------------
print("Dataset Shape:", df.shape)
print("\nColumns:")

print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nStatistical Summary:")
print(df.describe())

# -----------------------------
# Target Variable Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.show()

# -----------------------------
# Gender Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="gender")
plt.title("Gender Distribution")
plt.show()

# -----------------------------
# Contract Type
# -----------------------------
plt.figure(figsize=(7,4))
sns.countplot(data=df, x="Contract")
plt.title("Contract Types")
plt.show()

# -----------------------------
# Payment Method
# -----------------------------
plt.figure(figsize=(10,4))
sns.countplot(data=df, x="PaymentMethod")
plt.xticks(rotation=25)
plt.title("Payment Methods")
plt.show()

# -----------------------------
# Monthly Charges
# -----------------------------
plt.figure(figsize=(8,4))
sns.histplot(df["MonthlyCharges"], bins=30, kde=True)
plt.title("Monthly Charges Distribution")
plt.show()

# -----------------------------
# Tenure
# -----------------------------
plt.figure(figsize=(8,4))
sns.histplot(df["tenure"], bins=30, kde=True)
plt.title("Customer Tenure")
plt.show()