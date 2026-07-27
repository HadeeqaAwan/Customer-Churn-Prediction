import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    accuracy_score,
    roc_auc_score
)

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("data/preprocessed_data.csv")

# -----------------------
# Features and Target
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
# Load Scaler
# -----------------------
scaler = joblib.load("models/scaler.pkl")

# Scale Test Data
X_test = scaler.transform(X_test)

# -----------------------
# Load Trained Model
# -----------------------
model = joblib.load("models/best_model.pkl")

# -----------------------
# Make Predictions
# -----------------------
y_pred = model.predict(X_test)

# -----------------------
# Accuracy
# -----------------------
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")

# -----------------------
# Classification Report
# -----------------------
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------
# ROC-AUC Score
# -----------------------
y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)

print(f"\nROC-AUC Score: {auc:.4f}")

# -----------------------
# Confusion Matrix
# -----------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.close()

# -----------------------
# ROC Curve
# -----------------------
RocCurveDisplay.from_estimator(
    model,
    X_test,
    y_test
)

plt.title("ROC Curve")

plt.savefig("roc_curve.png")
plt.close()

print("\nModel evaluation completed successfully!")