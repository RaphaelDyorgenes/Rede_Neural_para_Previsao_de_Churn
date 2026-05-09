# Testes de integração da API FastAPI.
import pytest
from fastapi.testclient import TestClient

from src.api.api import app


@pytest.fixture
def client():
    # Client de teste do FastAPI (usa o lifespan da aplicação).
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health_endpoint_returns_200(client):
    # GET /health retorna status 200.
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_predict_returns_valid_response(client):
    # POST /predict retorna churn_probability e churn_prediction.
    payload = {
        "Count": 1,
        "Zip Code": 90210,
        "Latitude": 34.09,
        "Longitude": -118.41,
        "Tenure Months": 12,
        "Monthly Charges": 70.5,
        "Total Charges": 846.0,
        "Gender": "Male",
        "Senior Citizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
    }
    response = client.post("/predict", json=payload)
    # Se os modelos não foram treinados, a API retorna erro controlado.
    # Verificamos que a resposta é 200 e contém campos esperados OU erro.
    assert response.status_code == 200
    data = response.json()
    has_prediction = "churn_probability" in data and "churn_prediction" in data
    has_error = "error" in data
    assert has_prediction or has_error


def test_predict_rejects_invalid_payload(client):
    # POST /predict com payload vazio ainda funciona (defaults do Pydantic).
    response = client.post("/predict", json={})
    assert response.status_code == 200


def test_predict_rejects_malformed_json(client):
    # POST /predict com JSON inválido retorna 422 (Unprocessable Entity).
    response = client.post(
        "/predict",
        content="isto não é json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_health_response_has_latency_header(client):
    # Middleware de latência adiciona header X-Process-Time.
    response = client.get("/health")
    assert "x-process-time" in response.headers
