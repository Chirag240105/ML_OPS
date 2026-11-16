"""
Tests for the RiskStream Fraud Detection API.

Fake-model tests run unconditionally.
Integration tests are skipped when the real model file is absent.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ensure src/ is on the path before importing anything that may load the model
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from main import create_app  # noqa: E402

# ---------------------------------------------------------------------------
# Fake model (implements the same interface as RiskStreamModel)
# ---------------------------------------------------------------------------

class FakeRiskStreamModel:
    """Deterministic stub – always returns legit with score 0.0."""

    def predict(self, df: pd.DataFrame):
        return [0] * len(df)

    def score(self, df: pd.DataFrame):
        return [0.0] * len(df)

    def explain(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "rule_exact_drain": [False] * len(df),
                "rule_cap_10m": [False] * len(df),
                "rule_zero_cashout": [False] * len(df),
            }
        )


def _fake_model_loader():
    return FakeRiskStreamModel()


def _fake_version_loader():
    return "test_version"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    app = create_app(
        model_loader=_fake_model_loader,
        version_loader=_fake_version_loader,
    )
    with TestClient(app) as test_client:
        yield test_client


_REAL_MODEL_PATH = Path(__file__).parent.parent / "models" / "riskstream_v1.joblib"

skip_no_model = pytest.mark.skipif(
    not _REAL_MODEL_PATH.exists(),
    reason="Real model file not found – skipping integration test",
)


@pytest.fixture(scope="module")
def real_client():
    """Client backed by the actual joblib model."""
    import warnings
    warnings.filterwarnings("ignore")
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Unit tests (fake model)
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    """Health check returns 200 with expected fields."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_name"] == "riskstream"
    assert body["model_version"] == "test_version"


def test_valid_prediction_returns_expected_keys(client):
    """A valid request returns is_fraud, fraud_score, rules_fired, model_version."""
    payload = {
        "type": "CASH_OUT",
        "amount": 100.0,
        "oldbalanceOrg": 500.0,
        "oldbalanceDest": 0.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "is_fraud" in body
    assert "fraud_score" in body
    assert "rules_fired" in body
    assert "model_version" in body
    assert isinstance(body["is_fraud"], bool)
    assert isinstance(body["fraud_score"], float)
    assert isinstance(body["rules_fired"], list)


def test_invalid_type_returns_422(client):
    """An unknown transaction type must be rejected."""
    payload = {
        "type": "WIRE_TRANSFER",
        "amount": 100.0,
        "oldbalanceOrg": 500.0,
        "oldbalanceDest": 0.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_negative_amount_returns_422(client):
    """A negative amount violates the ge=0 constraint."""
    payload = {
        "type": "CASH_OUT",
        "amount": -1.0,
        "oldbalanceOrg": 500.0,
        "oldbalanceDest": 0.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_missing_field_returns_422(client):
    """A request missing a required field must be rejected."""
    payload = {
        "type": "PAYMENT",
        "amount": 50.0,
        # oldbalanceOrg omitted
        "oldbalanceDest": 0.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_zero_amount_is_accepted(client):
    """Amount of 0 is valid (ge=0, not gt=0)."""
    payload = {
        "type": "PAYMENT",
        "amount": 0.0,
        "oldbalanceOrg": 0.0,
        "oldbalanceDest": 0.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Integration tests (real model)
# ---------------------------------------------------------------------------

@skip_no_model
def test_integration_exact_drain_flagged(real_client):
    """CASH_OUT that drains the account exactly must fire rule_exact_drain."""
    payload = {
        "type": "CASH_OUT",
        "amount": 146908.38,
        "oldbalanceOrg": 146908.38,
        "oldbalanceDest": 0.0,
    }
    response = real_client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["is_fraud"] is True
    assert "rule_exact_drain" in body["rules_fired"]


@skip_no_model
def test_integration_cap_10m_flagged(real_client):
    """TRANSFER of exactly 10 000 000 where sender had more must fire rule_cap_10m."""
    payload = {
        "type": "TRANSFER",
        "amount": 10_000_000.0,
        "oldbalanceOrg": 15_999_376.79,
        "oldbalanceDest": 0.0,
    }
    response = real_client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["is_fraud"] is True
    assert "rule_cap_10m" in body["rules_fired"]


@skip_no_model
def test_integration_payment_not_flagged(real_client):
    """A routine PAYMENT must not be flagged."""
    payload = {
        "type": "PAYMENT",
        "amount": 9839.64,
        "oldbalanceOrg": 170136.0,
        "oldbalanceDest": 0.0,
    }
    response = real_client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["is_fraud"] is False
    assert body["rules_fired"] == []
