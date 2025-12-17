from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Credit Risk API" in response.json()["message"]

def test_predict():
    response = client.post("/predict", json={
        "feature_1": 35,
        "feature_2": 50000,
        "feature_3": 10000,
        "feature_4": 12,
        "feature_5": 700
    })
    assert response.status_code == 200
    assert "risk_probability" in response.json()
