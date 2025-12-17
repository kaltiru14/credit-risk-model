import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# =========================================================
# 1. Load Processed Data
# =========================================================
# processed_path = r"../data/processed/data_with_target.csv"
processed_path = r"D:\tenx\week 4\credit-risk-model\data\processed\data_with_target.csv"

 # already correct if running from project root

df = pd.read_csv(processed_path)

# Separate features and target
target_col = "is_high_risk"
X = df.drop(columns=[target_col, "TransactionId", "BatchId", "AccountId", "SubscriptionId"])
y = df[target_col]

# =========================================================
# 2. Train/Test Split
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# 3. Preprocessing Pipeline
# =========================================================
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

scaler = StandardScaler()

# =========================================================
# 4. Model Definitions
# =========================================================
models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(random_state=42),
}

# Hyperparameter grids
param_grids = {
    "logistic_regression": {
        "C": np.logspace(-3, 3, 10),
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
}

# =========================================================
# 5. MLflow Experiment
# =========================================================
mlflow.set_experiment("credit_risk_model")

best_models = {}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        print(f"Training {name}...")

        # Randomized search
        search = RandomizedSearchCV(
            model,
            param_distributions=param_grids[name],
            n_iter=5,
            scoring="roc_auc",
            cv=3,
            random_state=42,
            n_jobs=-1,
        )
        # Fit
        search.fit(scaler.fit_transform(X_train[numeric_features]), y_train)
        best_model = search.best_estimator_

        # Predict
        y_pred = best_model.predict(scaler.transform(X_test[numeric_features]))
        y_proba = best_model.predict_proba(scaler.transform(X_test[numeric_features]))[:, 1]

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        print(f"{name} metrics:", metrics)

        # Log to MLflow
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        best_models[name] = best_model

print("\nBest models trained and logged in MLflow.")
