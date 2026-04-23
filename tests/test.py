"""Unit tests for preprocessing and modeling utilities."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.preprocessing import (
    FEATURE_COLS,
    TARGET_COL,
    add_row_statistics,
    data_quality_report,
    scale_features,
    split_data,
)
from src.modeling import build_results_table, cv_roc_auc, evaluate

SEED = 42
N_SAMPLES = 500
N_FEATURES = 200


def _make_fake_df(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, N_FEATURES))
    cols = {f"var_{i}": data[:, i] for i in range(N_FEATURES)}
    cols["ID_code"] = [f"train_{i}" for i in range(n)]
    cols[TARGET_COL] = rng.integers(0, 2, size=n)
    return pd.DataFrame(cols)


# --- preprocessing tests ---


def test_data_quality_report_no_missing():
    df = _make_fake_df()
    report = data_quality_report(df)
    assert report["missing_total"] == 0
    assert report["duplicates"] == 0
    assert report["shape"] == (N_SAMPLES, N_FEATURES + 2)


def test_split_data_sizes():
    df = _make_fake_df()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, val_size=0.1, test_size=0.1)
    total = len(y_train) + len(y_val) + len(y_test)
    assert total == N_SAMPLES
    assert len(y_test) == pytest.approx(N_SAMPLES * 0.1, abs=5)
    assert len(y_val) == pytest.approx(N_SAMPLES * 0.1, abs=5)


def test_split_preserves_class_ratio():
    df = _make_fake_df()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    orig_ratio = df[TARGET_COL].mean()
    for y_split in [y_train, y_val, y_test]:
        assert abs(y_split.mean() - orig_ratio) < 0.05


def test_scale_features_zero_mean():
    df = _make_fake_df()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_tr_s, X_val_s, X_test_s, _ = scale_features(X_train, X_val, X_test)
    assert abs(X_tr_s.mean()) < 0.01


def test_add_row_statistics_shape():
    X = np.random.rand(100, N_FEATURES)
    X_aug = add_row_statistics(X)
    assert X_aug.shape == (100, N_FEATURES + 5)


# --- modeling tests ---


def test_evaluate_keys():
    df = _make_fake_df()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_tr_s, X_val_s, _, _ = scale_features(X_train, X_val, X_test)
    model = LogisticRegression(max_iter=200, random_state=SEED)
    model.fit(X_tr_s, y_train)
    metrics = evaluate(model, X_val_s, y_val)
    assert set(metrics.keys()) == {"roc_auc", "avg_precision", "f1", "accuracy"}
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_cv_roc_auc_returns_tuple():
    df = _make_fake_df()
    X_train, X_val, X_test, y_train, _, _ = split_data(df)
    model = LogisticRegression(max_iter=200, random_state=SEED)
    mean_auc, std_auc = cv_roc_auc(model, X_train, y_train, n_splits=3)
    assert 0.0 <= mean_auc <= 1.0
    assert std_auc >= 0.0


def test_build_results_table_sorting():
    records = [
        {"model": "A", "val_roc_auc": 0.80},
        {"model": "B", "val_roc_auc": 0.90},
        {"model": "C", "val_roc_auc": 0.75},
    ]
    table = build_results_table(records)
    assert table.iloc[0]["val_roc_auc"] == 0.90
    assert table.shape == (3, 2)
