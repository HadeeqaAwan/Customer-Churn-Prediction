import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ======================================
# Create Models Folder
# ======================================

os.makedirs("models", exist_ok=True)


# ======================================
# MLflow Setup
# ======================================

mlflow.set_experiment("Customer Churn Prediction")


# ======================================
# Load Dataset
# ======================================

df = pd.read_csv("data/preprocessed_data.csv")


# ======================================
# Features & Target
# ======================================

X = df.drop("Churn", axis=1)
y = df["Churn"]


# ======================================
# Train Test Split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ======================================
# Feature Scaling
# ======================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)


# ======================================
# Train & Log Model Function
# ======================================

def train_and_log_model(model, model_name):

    with mlflow.start_run(run_name=model_name):

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(X_test)


        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions
        )

        recall = recall_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )


        print("\n==============================")
        print(model_name)
        print("==============================")

        print("Accuracy:", accuracy)

        print(
            classification_report(
                y_test,
                predictions
            )
        )


        # Parameters

        mlflow.log_param(
            "Model",
            model_name
        )


        if model_name == "Logistic Regression":

            mlflow.log_param(
                "Max Iterations",
                1000
            )


        if model_name == "Random Forest":

            mlflow.log_param(
                "Number of Trees",
                100
            )


        # Metrics

        mlflow.log_metric(
            "Accuracy",
            accuracy
        )

        mlflow.log_metric(
            "Precision",
            precision
        )

        mlflow.log_metric(
            "Recall",
            recall
        )

        mlflow.log_metric(
            "F1 Score",
            f1
        )


        # Save MLflow run model

        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name.replace(" ", "_")
        )


        return accuracy, model



# ======================================
# Train Different Models
# ======================================


lr_accuracy, lr_model = train_and_log_model(
    LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Logistic Regression"
)



dt_accuracy, dt_model = train_and_log_model(
    DecisionTreeClassifier(
        random_state=42
    ),
    "Decision Tree"
)



rf_accuracy, rf_model = train_and_log_model(
    RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Random Forest"
)



# ======================================
# Compare Models
# ======================================

models_accuracy = {

    "Logistic Regression": lr_accuracy,

    "Decision Tree": dt_accuracy,

    "Random Forest": rf_accuracy

}



best_model_name = max(
    models_accuracy,
    key=models_accuracy.get
)



print("\n==============================")
print("Model Comparison")
print("==============================")


for model, score in models_accuracy.items():

    print(
        f"{model}: {score:.4f}"
    )


print(
    f"\nBest Model: {best_model_name}"
)



# ======================================
# Select Best Model
# ======================================

if best_model_name == "Logistic Regression":

    best_model = lr_model


elif best_model_name == "Decision Tree":

    best_model = dt_model


else:

    best_model = rf_model



# ======================================
# Save Best Model
# ======================================

joblib.dump(
    best_model,
    "models/best_model.pkl"
)



# ======================================
# Register Best Model in MLflow
# ======================================

with mlflow.start_run(
    run_name="Best_Model_Registration"
):

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="best_model",
        registered_model_name="Customer_Churn_Model"
    )



print("\nBest model saved successfully!")

print("\nMLflow Experiment Logged Successfully!")