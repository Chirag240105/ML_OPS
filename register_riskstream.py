"""
One-time script to register riskstream_v1.joblib into MLflow.
Run from ml-firstPipeline/ with the MLflow server already running on port 7001.

    $env:PYTHONPATH = "src"
    python register_riskstream.py
"""
import sys
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:7001"
MODEL_NAME   = "riskstream"
JOBLIB_PATH  = "models/riskstream_v1.joblib"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("riskstream-fraud-detection")


# ---------------------------------------------------------------------------
# Wrap RiskStreamModel as a pyfunc so MLflow can log and load it properly
# ---------------------------------------------------------------------------

class RiskStreamWrapper(mlflow.pyfunc.PythonModel):
    """
    Thin pyfunc wrapper. MLflow calls predict(context, model_input).
    We return the full fraud response as a DataFrame.
    """

    def load_context(self, context):
        sys.path.insert(0, context.artifacts["src_path"])
        self.model = joblib.load(context.artifacts["model_path"])

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        preds   = self.model.predict(model_input)
        scores  = self.model.score(model_input)
        explain = self.model.explain(model_input)

        rules_fired = [
            [col for col in explain.columns if explain.iloc[i][col]]
            for i in range(len(model_input))
        ]

        return pd.DataFrame({
            "is_fraud":    preds.astype(bool),
            "fraud_score": scores,
            "rules_fired": [str(r) for r in rules_fired],
        })


# ---------------------------------------------------------------------------
# Log the model as a proper pyfunc
# ---------------------------------------------------------------------------

print("Logging run...")

artifacts = {
    "model_path": JOBLIB_PATH,
    "src_path":   "src",
}

sample_input = pd.DataFrame([{
    "type":           "CASH_OUT",
    "amount":         146908.38,
    "oldbalanceOrg":  146908.38,
    "oldbalanceDest": 0.0,
}])

with mlflow.start_run(run_name="riskstream_v1_register"):
    mlflow.log_param("model_type",      "RiskStreamModel")
    mlflow.log_param("source",          "colab")
    mlflow.log_param("sklearn_version", "1.6.1")
    mlflow.log_param("xgboost_version", "3.2.0")
    mlflow.log_param("python_version",  "3.11")

    mlflow.pyfunc.log_model(
        artifact_path="riskstream_model",
        python_model=RiskStreamWrapper(),
        artifacts=artifacts,
        input_example=sample_input,
        registered_model_name=MODEL_NAME,   # registers in one step
    )

    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")

# ---------------------------------------------------------------------------
# Assign champion alias to version 1
# ---------------------------------------------------------------------------

print("Setting champion alias...")
client = MlflowClient(tracking_uri=TRACKING_URI)

# Wait briefly for registration to complete, then fetch version
import time
time.sleep(2)

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest   = sorted(versions, key=lambda v: int(v.version))[-1]

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=latest.version,
)

print(f"\nDone.")
print(f"Model '{MODEL_NAME}' version {latest.version} aliased as 'champion'.")
print(f"Open http://127.0.0.1:7001 -> Models -> {MODEL_NAME}")
