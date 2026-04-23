"""
preprocessing.py
Preprocessing pipeline: outlier removal, normalisation, sliding-window sequencing.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib, os

FEATURE_COLS = [
    "T_hot_in", "T_hot_out", "T_cold_in", "T_cold_out",
    "flow_hot", "flow_cold",
    "delta_T_hot", "delta_T_cold", "NTU", "Re",
]
TARGET_COL   = "Rf"
SEQ_LEN      = 24   # 24-hour look-back window


# ──────────────────────────────────────────────────────────────────────────────
def remove_outliers_iqr(df: pd.DataFrame, cols: list, k: float = 1.5) -> pd.DataFrame:
    """IQR-based outlier clipping (inplace on a copy)."""
    df = df.copy()
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(q1 - k * iqr, q3 + k * iqr)
    return df


def log_transform_target(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform Rf to reduce right skew."""
    df = df.copy()
    df["Rf_log"] = np.log1p(df[TARGET_COL] * 1e6)   # scale to µ units first
    return df


def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Sliding-window sequencing → (N, seq_len, n_features), (N,)."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len: i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_test_split_time(
    df: pd.DataFrame,
    train_frac: float = 0.8
):
    """Chronological (not random) split to avoid data leakage."""
    split = int(len(df) * train_frac)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


class FoulingPreprocessor:
    """Fit on train, transform train & test, serialise scalers."""

    def __init__(self, seq_len: int = SEQ_LEN):
        self.seq_len      = seq_len
        self.feat_scaler  = MinMaxScaler()
        self.tgt_scaler   = MinMaxScaler()

    # ── fit ──────────────────────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame):
        df = remove_outliers_iqr(df, FEATURE_COLS + [TARGET_COL])
        df = log_transform_target(df)

        X_raw = df[FEATURE_COLS].values
        y_raw = df["Rf_log"].values.reshape(-1, 1)

        X_scaled = self.feat_scaler.fit_transform(X_raw)
        y_scaled = self.tgt_scaler.fit_transform(y_raw).ravel()

        return build_sequences(X_scaled, y_scaled, self.seq_len)

    # ── transform only ───────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame):
        df = remove_outliers_iqr(df, FEATURE_COLS + [TARGET_COL])
        df = log_transform_target(df)

        X_raw = df[FEATURE_COLS].values
        y_raw = df["Rf_log"].values.reshape(-1, 1)

        X_scaled = self.feat_scaler.transform(X_raw)
        y_scaled = self.tgt_scaler.transform(y_raw).ravel()

        return build_sequences(X_scaled, y_scaled, self.seq_len)

    # ── inverse transform predictions ────────────────────────────────────
    def inverse_target(self, y_pred_scaled: np.ndarray) -> np.ndarray:
        y_log = self.tgt_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        return np.expm1(y_log) / 1e6   # back to m²K/W

    # ── persistence ──────────────────────────────────────────────────────
    def save(self, path: str = "models/preprocessor.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"[✓] Preprocessor saved → {path}")

    @staticmethod
    def load(path: str = "models/preprocessor.pkl") -> "FoulingPreprocessor":
        return joblib.load(path)
