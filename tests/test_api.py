from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "High-Frequency Transaction Fraud Engine"
    assert data["status"] == "ONLINE"

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True
    assert data["threshold"] == 0.09

def test_predict_fraud_normal_transaction():
    payload = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 9839.64,
        "nameOrig": "C1231006815",
        "oldbalanceOrg": 170136.0,
        "newbalanceOrig": 160296.36,
        "nameDest": "M1979787155",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
    }
    response = client.post("/predict_fraud", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_detected" in data
    assert "probability" in data
    assert data["cost_sensitive_threshold"] == 0.09
    assert data["decision"] in ["APPROVED", "FLAGGED_FOR_REVIEW"]

def test_predict_fraud_suspicious_transfer():
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 500000.0,
        "nameOrig": "C1305486145",
        "oldbalanceOrg": 500000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C553264065",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
    }
    response = client.post("/predict_fraud", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["fraud_detected"] is True
    assert data["decision"] == "FLAGGED_FOR_REVIEW"
    assert data["probability"] >= 0.09

def test_explain_fraud():
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 181.0,
        "nameOrig": "C1305486145",
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C553264065",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
    }
    response = client.post("/explain_fraud", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "top_risk_factors" in data
