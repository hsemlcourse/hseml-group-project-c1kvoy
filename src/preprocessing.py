import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURE_COLS = [f"var_{i}" for i in range(200)]
TARGET_COL = "target"
ID_COL = "ID_code"


def load_data(train_path: str, test_path: str = None):
    train = pd.read_csv(train_path)
    if test_path is not None:
        test = pd.read_csv(test_path)
        return train, test
    return train


def data_quality_report(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "missing_total": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": df.dtypes.value_counts().to_dict(),
    }


def split_data(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = SEED,
):
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, random_state=seed, stratify=y_tv
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def add_row_statistics(X: np.ndarray) -> np.ndarray:
    stats = np.column_stack([
        X.mean(axis=1),
        X.std(axis=1),
        X.min(axis=1),
        X.max(axis=1),
        X.sum(axis=1),
    ])
    return np.hstack([X, stats])


def add_frequency_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame = None,
) -> tuple:
    train_df = train_df.copy()
    if test_df is not None:
        test_df = test_df.copy()
        combined = pd.concat(
            [train_df[FEATURE_COLS], test_df[FEATURE_COLS]], ignore_index=True
        )
    else:
        combined = train_df[FEATURE_COLS]

    for col in FEATURE_COLS:
        freq = combined[col].round(2).value_counts()
        train_df[f"{col}_freq"] = train_df[col].round(2).map(freq).fillna(0).astype(np.int32)
        if test_df is not None:
            test_df[f"{col}_freq"] = test_df[col].round(2).map(freq).fillna(0).astype(np.int32)

    if test_df is not None:
        return train_df, test_df
    return train_df
