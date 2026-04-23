"""
api.py
Flask / FastAPI deployment endpoint.
Accepts the last 24 sensor readings and returns predicted Rf with alert status.

Usage:
  python api.py          # starts on http://0.0.0.0:8000
  POST /predict  JSON body: { "readings": [ {sensor fields…}, …24 items ] }
"""

import os, sys
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import FoulingPreprocessor, FEATURE_COLS, remove_outliers_iqr
import tensorflow as tf

# ─────────────────────────────────────────────────────────────────────────────
ALERT_THRESHOLD_RF = 3e-4   # m²K/W  → 50% efficiency drop
MODEL_PATH         = "models/lstm_best.keras"
PREP_PATH          = "models/preprocessor.pkl"
SEQ_LEN            = 24

app  = Flask(__name__)

# Lazy-load on first request to keep startup fast
_lstm = None
_prep = None

def get_models():
    global _lstm, _prep
    if _lstm is None:
        _lstm = tf.keras.models.load_model(MODEL_PATH)
        _prep = FoulingPreprocessor.load(PREP_PATH)
    return _lstm, _prep


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def readings_to_sequence(readings: list[dict]) -> np.ndarray:
    """
    readings : list of dicts, each with keys matching FEATURE_COLS.
    Returns  : (1, SEQ_LEN, n_features)
    """
    if len(readings) < SEQ_LEN:
        raise ValueError(f"Need at least {SEQ_LEN} readings, got {len(readings)}")

    df = pd.DataFrame(readings[-SEQ_LEN:])
    # Ensure derived features exist
    if "delta_T_hot" not in df.columns:
        df["delta_T_hot"]  = df["T_hot_in"]  - df["T_hot_out"]
    if "delta_T_cold" not in df.columns:
        df["delta_T_cold"] = df["T_cold_out"] - df["T_cold_in"]

    return df[FEATURE_COLS].values.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Body (JSON):
    {
      "readings": [
        {"T_hot_in": 200, "T_hot_out": 150, "T_cold_in": 25, "T_cold_out": 45,
         "flow_hot": 6.0, "flow_cold": 8.0,
         "delta_T_hot": 50, "delta_T_cold": 20, "NTU": 2.1, "Re": 18000},
        …  (24 timesteps)
      ]
    }

    Response:
    {
      "Rf_predicted": 0.000312,
      "alert": true,
      "alert_message": "Fouling threshold exceeded – schedule cleaning"
    }
    """
    try:
        lstm, prep = get_models()
        data = request.get_json(force=True)
        readings = data["readings"]

        X_raw = readings_to_sequence(readings)   # (SEQ_LEN, n_features)

        # Scale features using fitted scaler
        X_scaled = prep.feat_scaler.transform(X_raw)
        X_seq    = X_scaled[np.newaxis, ...]     # (1, SEQ_LEN, n_features)

        y_scaled = lstm.predict(X_seq, verbose=0).ravel()
        Rf_pred  = prep.inverse_target(y_scaled)[0]

        alert = bool(Rf_pred > ALERT_THRESHOLD_RF)
        return jsonify({
            "Rf_predicted": float(Rf_pred),
            "Rf_threshold": ALERT_THRESHOLD_RF,
            "alert": alert,
            "alert_message": (
                "  Fouling threshold exceeded – schedule cleaning"
                if alert else "  Operating within normal limits"
            ),
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/simulate", methods=["GET"])
def simulate_demo():
    """Demo endpoint: returns simulated prediction for Rf at 500 hours."""
    from data_generator import simulate_fouling
    df = simulate_fouling(n_hours=500 + SEQ_LEN + 5)
    readings = df.iloc[500: 500 + SEQ_LEN][FEATURE_COLS].to_dict(orient="records")

    lstm, prep = get_models()
    X_raw    = np.array([[r[f] for f in FEATURE_COLS] for r in readings], dtype=np.float32)
    X_scaled = prep.feat_scaler.transform(X_raw)[np.newaxis, ...]
    y_scaled = lstm.predict(X_scaled, verbose=0).ravel()
    Rf_pred  = prep.inverse_target(y_scaled)[0]

    return jsonify({
        "demo_hour": 500,
        "Rf_predicted": float(Rf_pred),
        "Rf_actual":    float(df.iloc[500 + SEQ_LEN]["Rf"]),
        "alert":        bool(Rf_pred > ALERT_THRESHOLD_RF),
    })


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"[API] Starting on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
