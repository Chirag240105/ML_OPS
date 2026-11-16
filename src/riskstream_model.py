import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

RISKY = ["CASH_OUT", "TRANSFER"]
RAW_COLS = ["type", "amount", "oldbalanceOrg", "oldbalanceDest"]
SCORE_NUM = ["amount", "oldbalanceOrg", "amount_to_balance",
             "exact_drain", "orig_zero", "balance_gap"]   # no oldbalanceDest
SCORE_CAT = ["type"]


def build_features(d):
    d = d.copy()
    d["amount_to_balance"] = d["amount"] / (d["oldbalanceOrg"] + 1)
    d["orig_zero"] = (d["oldbalanceOrg"] == 0).astype(int)
    d["exact_drain"] = (((d["amount"] - d["oldbalanceOrg"]).abs() < 0.01)
                        & (d["oldbalanceOrg"] > 0)).astype(int)
    d["balance_gap"] = d["oldbalanceOrg"] - d["amount"]
    return d


def rule_flags(d):
    risky = d["type"].isin(RISKY)
    exact = risky & (d["exact_drain"] == 1)
    cap = ((d["type"] == "TRANSFER") & (d["amount"] == 10_000_000)
           & (d["oldbalanceOrg"] > 10_000_000))
    zero = (d["type"] == "CASH_OUT") & (d["amount"] == 0)
    return exact, cap, zero


class RiskStreamModel(BaseEstimator, ClassifierMixin):
    """Rule layer + XGBoost score. Input: raw transaction columns."""

    def __init__(self, scale_pos_weight=32.0, n_estimators=300,
                 max_depth=6, learning_rate=0.1, hybrid_threshold=None,
                 random_state=42):
        self.scale_pos_weight = scale_pos_weight
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.hybrid_threshold = hybrid_threshold  # None = rules decide alone
        self.random_state = random_state

    def fit(self, X, y):
        F = build_features(X[RAW_COLS])
        pre = ColumnTransformer([
            ("num", "passthrough", SCORE_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), SCORE_CAT),
        ])
        self.scorer_ = Pipeline([
            ("preprocessor", pre),
            ("model", xgb.XGBClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                scale_pos_weight=self.scale_pos_weight,
                tree_method="hist", random_state=self.random_state, n_jobs=-1)),
        ])
        self.scorer_.fit(F[SCORE_NUM + SCORE_CAT], y)
        self.classes_ = np.array([0, 1])
        return self

    def _features(self, X):
        return build_features(X[RAW_COLS])

    def score(self, X):
        """XGBoost fraud score (ranking signal, not a calibrated probability)."""
        F = self._features(X)
        return self.scorer_.predict_proba(F[SCORE_NUM + SCORE_CAT])[:, 1]

    def explain(self, X):
        """Which rules fired, per row."""
        F = self._features(X)
        exact, cap, zero = rule_flags(F)
        return pd.DataFrame({"rule_exact_drain": exact.values,
                             "rule_cap_10m": cap.values,
                             "rule_zero_cashout": zero.values}, index=X.index)

    def predict(self, X):
        F = self._features(X)
        exact, cap, zero = rule_flags(F)
        flag = (exact | cap | zero).to_numpy()
        if self.hybrid_threshold is not None:
            flag = flag | (self.score(X) >= self.hybrid_threshold)
        return flag.astype(int)

    def predict_proba(self, X):
        p = self.score(X)
        return np.column_stack([1 - p, p])
