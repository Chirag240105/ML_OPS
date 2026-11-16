"""
Rebuilds riskstream_v1.joblib by:
1. Loading the XGBoost booster from the JSON file (xgb_booster.json)
2. Reconstructing the full RiskStreamModel with correct sklearn pipeline
3. Re-saving as a fresh joblib compatible with xgboost 3.4.1

Run:
    python rebuild_model.py
"""
import os
import sys
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from riskstream_model import RiskStreamModel

SCORE_NUM = ["amount", "oldbalanceOrg", "amount_to_balance",
             "exact_drain", "orig_zero", "balance_gap"]
SCORE_CAT = ["type"]

BOOSTER_JSON = BASE_DIR / "models" / "xgb_booster.json"
OUTPUT_PATH  = BASE_DIR / "models" / "riskstream_v1.joblib"

# ------------------------------------------------------------------
# 1. Load the XGBoost booster from JSON
# ------------------------------------------------------------------
if not BOOSTER_JSON.exists():
    print(f"ERROR: {BOOSTER_JSON} not found.")
    print("Save the booster JSON from Colab with:")
    print("  model.scorer_.named_steps['model'].get_booster().save_model('xgb_booster.json')")
    sys.exit(1)

booster = xgb.Booster()
booster.load_model(str(BOOSTER_JSON))
print(f"Loaded booster from {BOOSTER_JSON}")

# ------------------------------------------------------------------
# 2. Reconstruct the XGBClassifier wrapping that booster
# ------------------------------------------------------------------
clf = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=32, tree_method="hist", random_state=42, n_jobs=-1
)
# Fit on dummy binary-class data so sklearn initializes classes_, n_classes_, etc.
X_dummy = np.zeros((2, booster.num_features()))
y_dummy = np.array([0, 1])
clf.fit(X_dummy, y_dummy)

# Inject the loaded booster
clf._Booster = booster

# ------------------------------------------------------------------
# 3. Build the sklearn Pipeline (same structure as training)
# ------------------------------------------------------------------
# Force OHE to know all 5 transaction types so output = 11 features
# (6 numeric + 5 OHE columns), matching the booster's expected feature count
ALL_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

pre = ColumnTransformer([
    ("num", "passthrough", SCORE_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore",
                          categories=[ALL_TYPES]), SCORE_CAT),
])

dummy = pd.DataFrame({
    "amount":           [100.0, 200.0, 300.0, 400.0, 500.0],
    "oldbalanceOrg":    [500.0,   0.0, 100.0, 200.0, 300.0],
    "amount_to_balance":[0.2,     1.0,   3.0,   1.0,   1.5],
    "exact_drain":      [0,       0,     0,     1,     0],
    "orig_zero":        [0,       1,     0,     0,     0],
    "balance_gap":      [400.0, -200.0,-200.0,-200.0,-200.0],
    "type": ALL_TYPES,
})
pre.fit(dummy[SCORE_NUM + SCORE_CAT])
print("Fitted preprocessor — OHE categories:", ALL_TYPES)

pipeline = Pipeline([
    ("preprocessor", pre),
    ("model", clf),
])

# ------------------------------------------------------------------
# 4. Assemble the RiskStreamModel
# ------------------------------------------------------------------
m = RiskStreamModel(
    scale_pos_weight=32.0, n_estimators=300, max_depth=6,
    learning_rate=0.1, hybrid_threshold=None, random_state=42
)
m.scorer_  = pipeline
m.classes_ = np.array([0, 1])

# ------------------------------------------------------------------
# 5. Quick smoke test (including predict, score, explain)
# ------------------------------------------------------------------
test_cases = [
    {"type": "CASH_OUT", "amount": 146908.38, "oldbalanceOrg": 146908.38, "oldbalanceDest": 0.0},
    {"type": "TRANSFER", "amount": 10_000_000.0, "oldbalanceOrg": 15_999_376.79, "oldbalanceDest": 0.0},
    {"type": "PAYMENT",  "amount": 9839.64,       "oldbalanceOrg": 170136.0,      "oldbalanceDest": 0.0},
]
labels = ["CASH_OUT drain", "TRANSFER cap", "PAYMENT legit"]

print("\nSmoke test:")
for label, row in zip(labels, test_cases):
    df = pd.DataFrame([row])
    pred  = m.predict(df)
    score = m.score(df)
    expl  = m.explain(df).to_dict("records")[0]
    fired = [k for k, v in expl.items() if v]
    print(f"  {label}: is_fraud={bool(pred[0])}, score={score[0]:.6f}, rules={fired}")

# ------------------------------------------------------------------
# 6. Save
# ------------------------------------------------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(m, str(OUTPUT_PATH))
print(f"\nSaved: {OUTPUT_PATH}")
print("Run: python -m pytest -v")
