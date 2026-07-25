import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report

# -----------------------
# Load Preprocessed Data
# -----------------------
df = pd.read_csv("data/preprocessed_data.csv")

# -----------------------
# Split Features and Target
# -----------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# -----------------------
# Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------
# Feature Scaling
# -----------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save Scaler
joblib.dump(scaler, "models/scaler.pkl")

# -----------------------
# Logistic Regression
# -----------------------
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

print("\n========== Logistic Regression ==========")
print("Accuracy:", accuracy_score(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions))

# -----------------------
# Decision Tree
# -----------------------
dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

print("\n========== Decision Tree ==========")
print("Accuracy:", accuracy_score(y_test, dt_predictions))
print(classification_report(y_test, dt_predictions))

# -----------------------
# Random Forest
# -----------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

print("\n========== Random Forest ==========")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))

# -----------------------
# Compare Models
# -----------------------
models = {
    "Logistic Regression": accuracy_score(y_test, lr_predictions),
    "Decision Tree": accuracy_score(y_test, dt_predictions),
    "Random Forest": accuracy_score(y_test, rf_predictions)
}

best_model_name = max(models, key=models.get)

print("\n==============================")
print("Model Comparison")
print("==============================")

for model, score in models.items():
    print(f"{model}: {score:.4f}")

print(f"\nBest Model: {best_model_name}")

# -----------------------
# Save Best Model
# -----------------------
if best_model_name == "Logistic Regression":
    joblib.dump(lr_model, "models/logistic_regression_model.pkl")

elif best_model_name == "Decision Tree":
    joblib.dump(dt_model, "models/decision_tree_model.pkl")

else:
    joblib.dump(rf_model, "models/random_forest_model.pkl")

print("\nBest model saved successfully!")