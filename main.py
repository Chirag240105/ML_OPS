import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure src/ is on the path before joblib unpickling so Python can find
# riskstream_model.RiskStreamModel regardless of where the process starts.
_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Default loaders (overridable via factory for tests)
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "riskstream_v1.joblib"


def _load_joblib_model():
    path = Path(os.getenv("MODEL_PATH", _DEFAULT_MODEL_PATH))
    return joblib.load(path)


def _load_model_version():
    return os.getenv("MODEL_VERSION", "riskstream_v1")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TransactionInput(BaseModel):
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_score: float
    rules_fired: list[str]
    model_version: str


class BatchRequest(BaseModel):
    transactions: list[TransactionInput] = Field(max_length=1000)


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    model_loader: Callable = _load_joblib_model,
    version_loader: Callable = _load_model_version,
) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.model = model_loader()
        except Exception as exc:
            raise RuntimeError(f"Failed to load model: {exc}") from exc
        app.state.model_version = version_loader()
        yield

    app = FastAPI(
        title="RiskStream Fraud Detection API",
        description="Real-time fraud scoring powered by RiskStream",
        version="1.0.0",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_prediction(transactions: list[TransactionInput]) -> list[dict]:
        model = app.state.model
        version = app.state.model_version

        df = pd.DataFrame(
            [
                {
                    "type": t.type,
                    "amount": t.amount,
                    "oldbalanceOrg": t.oldbalanceOrg,
                    "oldbalanceDest": t.oldbalanceDest,
                }
                for t in transactions
            ]
        )

        preds = model.predict(df)
        scores = model.score(df)
        explanations = model.explain(df)

        results = []
        for i in range(len(transactions)):
            rules_fired = [
                col
                for col in explanations.columns
                if explanations.iloc[i][col]
            ]
            results.append(
                {
                    "is_fraud": bool(preds[i]),
                    "fraud_score": float(scores[i]),
                    "rules_fired": rules_fired,
                    "model_version": version,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/")
    def ready():
        return {"status": "Ready to serve"}

    @app.get("/health", response_model=HealthResponse)
    def health_check():
        return {
            "status": "healthy",
            "model_name": "riskstream",
            "model_version": app.state.model_version,
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(data: TransactionInput):
        return _run_prediction([data])[0]

    @app.post("/predict/batch", response_model=list[PredictionResponse])
    def predict_batch(request: BatchRequest):
        if not request.transactions:
            raise HTTPException(status_code=422, detail="transactions list is empty")
        return _run_prediction(request.transactions)

    return app


# Production application instance
app = create_app()
