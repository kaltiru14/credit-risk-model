import mlflow.sklearn
from mlflow.tracking import MlflowClient

MODEL_NAME = "credit_risk_model"
client = MlflowClient()

def load_latest_model(experiment_id="1"):
    # Get latest run
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        raise RuntimeError(f"No runs found for experiment {experiment_id}")
    latest_run = runs[0]
    model_uri = f"runs:/{latest_run.info.run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    return model

# Load once
model = load_latest_model()

def make_prediction(features: list):
    prob = model.predict_proba([features])[0]
    predicted_class = int(model.predict([features])[0])
    return {
        "class_0_prob": float(prob[0]),
        "class_1_prob": float(prob[1]),
        "predicted_class": predicted_class
    }
