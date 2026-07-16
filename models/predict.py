"""
Apply a trained pipeline (from train_model.py) to the current-month
feature set to produce a ranked list. Kept separate from training so
you can load a pickled model without retraining every run.
"""
from __future__ import annotations
import joblib
import pandas as pd
from models.train_model import FEATURES


def load_model(path: str):
    return joblib.load(path)


def save_model(pipeline, path: str) -> None:
    joblib.dump(pipeline, path)


def predict_current_universe(pipeline, current_features: pd.DataFrame) -> pd.DataFrame:
    available_features = [f for f in FEATURES if f in current_features.columns]
    result = current_features.copy()
    result["ml_probability"] = pipeline.predict_proba(result[available_features])[:, 1]
    return result.sort_values("ml_probability", ascending=False)
