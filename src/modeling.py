import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

SEED = 42


def evaluate(model, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "avg_precision": round(float(average_precision_score(y, proba)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
    }


def cv_roc_auc(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = SEED,
) -> tuple:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, scoring="roc_auc", cv=skf, n_jobs=-1)
    return round(float(scores.mean()), 4), round(float(scores.std()), 4)


def save_model(model, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)


def build_results_table(records: list) -> pd.DataFrame:
    return (
        pd.DataFrame(records)
        .sort_values("val_roc_auc", ascending=False)
        .reset_index(drop=True)
    )
