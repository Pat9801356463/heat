"""
model.py
LSTM architecture, training, and evaluation for fouling prediction.
"""

import os, time
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────────────────────────────────────

def build_lstm(seq_len: int, n_features: int,
               units: int = 64, dropout: float = 0.2) -> tf.keras.Model:
    """
    Two-layer stacked LSTM → Dense(32, relu) → Dense(1).
    Matches architecture in project spec.
    """
    model = Sequential([
        LSTM(units, return_sequences=True,
             input_shape=(seq_len, n_features)),
        Dropout(dropout),
        BatchNormalization(),

        LSTM(units),
        Dropout(dropout),
        BatchNormalization(),

        Dense(32, activation="relu"),
        Dense(1),
    ], name="LSTM_FoulingPredictor")

    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse",
                  metrics=["mae"])
    return model


def train_lstm(model, X_train, y_train, X_val, y_val,
               epochs: int = 100, batch_size: int = 32,
               checkpoint_path: str = "models/lstm_best.keras"):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True,
                      verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                          min_lr=1e-6, verbose=1),
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss",
                        verbose=0),
    ]

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"\n[✓] Training finished in {elapsed:.1f}s  "
          f"(best val_loss={min(history.history['val_loss']):.6f})")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Baseline models
# ─────────────────────────────────────────────────────────────────────────────

def train_rf(X_train_2d, y_train, **kwargs):
    """Random Forest on flattened sequences."""
    rf = RandomForestRegressor(n_estimators=200, random_state=42,
                                n_jobs=-1, **kwargs)
    rf.fit(X_train_2d, y_train)
    return rf


def train_xgb(X_train_2d, y_train):
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=300, learning_rate=0.05,
                           max_depth=6, random_state=42,
                           tree_method="hist", verbosity=0)
        xgb.fit(X_train_2d, y_train,
                eval_set=[(X_train_2d, y_train)], verbose=False)
        return xgb
    except ImportError:
        print("[!] xgboost not installed – using GradientBoosting instead")
        gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                        max_depth=4, random_state=42)
        gb.fit(X_train_2d, y_train)
        return gb


def empirical_linear(t: np.ndarray, phi: float = 1.5e-7) -> np.ndarray:
    """Baseline: Rf = φ · t  (linear rate model)."""
    return phi * t


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             label: str = "Model") -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  [{label:12s}]  MAE={mae:.2e}  RMSE={rmse:.2e}  R²={r2:.4f}")
    return {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2}


def compare_models(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df["MAE_1e-4"] = df["MAE"] * 1e4
    df["R2_%"]     = df["R2"]  * 100
    return df.set_index("label")[["MAE_1e-4", "RMSE", "R2_%"]].round(4)
