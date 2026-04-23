"""
train.py
End-to-end training pipeline:
  1. Generate / load data
  2. Preprocess + create sequences
  3. Train LSTM + baselines
  4. Evaluate & print comparison table
  5. Save artefacts
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from data_generator import simulate_fouling
from preprocessing   import FoulingPreprocessor, train_test_split_time, FEATURE_COLS
from model           import (build_lstm, train_lstm,
                              train_rf, train_xgb, empirical_linear,
                              evaluate, compare_models)

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

SEED     = 42
SEQ_LEN  = 24
N_HOURS  = 720   # 30 days


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Generating simulation data …")

csv_path = "data/fouling_simulation.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    print(f"  Loaded cached data: {len(df)} rows")
else:
    df = simulate_fouling(n_hours=N_HOURS)
    df.to_csv(csv_path, index=False)
    print(f"  Generated {len(df)} rows → {csv_path}")

print(f"  Rf range: {df['Rf'].min():.3e} – {df['Rf'].max():.3e} m²K/W")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 2 — Preprocessing …")

df_train, df_test = train_test_split_time(df, train_frac=0.8)
prep = FoulingPreprocessor(seq_len=SEQ_LEN)

X_train, y_train = prep.fit_transform(df_train)
X_test,  y_test  = prep.transform(df_test)
prep.save("models/preprocessor.pkl")

print(f"  Train sequences : {X_train.shape}   targets: {y_train.shape}")
print(f"  Test  sequences : {X_test.shape}    targets: {y_test.shape}")

n_features = X_train.shape[2]
# Flat versions for tree-based models
X_train_2d = X_train.reshape(len(X_train), -1)
X_test_2d  = X_test.reshape(len(X_test),  -1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 3 — Training …")

# ── LSTM ─────────────────────────────────────────────────────────────────────
val_split = int(len(X_train) * 0.9)
X_tr, X_val = X_train[:val_split], X_train[val_split:]
y_tr, y_val = y_train[:val_split], y_train[val_split:]

lstm = build_lstm(SEQ_LEN, n_features)
lstm.summary()
history = train_lstm(lstm, X_tr, y_tr, X_val, y_val,
                     epochs=80, batch_size=32)

# ── Random Forest ─────────────────────────────────────────────────────────────
print("\n  Training Random Forest …")
rf = train_rf(X_train_2d, y_train)
joblib.dump(rf, "models/rf.pkl")

# ── XGBoost / GBM ────────────────────────────────────────────────────────────
print("  Training XGBoost/GBM …")
xgb = train_xgb(X_train_2d, y_train)
joblib.dump(xgb, "models/xgb.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 4. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 4 — Evaluation (test set) …")

# Predictions (in scaled space → inverse transform)
lstm_pred_scaled = lstm.predict(X_test, verbose=0).ravel()
rf_pred_scaled   = rf.predict(X_test_2d)
xgb_pred_scaled  = xgb.predict(X_test_2d)

# Convert back to physical units (m²K/W)
lstm_pred = prep.inverse_target(lstm_pred_scaled)
rf_pred   = prep.inverse_target(rf_pred_scaled)
xgb_pred  = prep.inverse_target(xgb_pred_scaled)
y_true    = prep.inverse_target(y_test)

# Empirical linear baseline (time in seconds)
t_test_s = np.arange(len(y_true)) * 3600.0
lin_pred  = empirical_linear(t_test_s)

results = [
    evaluate(y_true, lstm_pred, "LSTM"),
    evaluate(y_true, rf_pred,   "Random Forest"),
    evaluate(y_true, xgb_pred,  "XGBoost/GBM"),
    evaluate(y_true, lin_pred,  "Linear (empirical)"),
]

print("\nComparison Table:")
print(compare_models(results).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 5 — Saving plots …")

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Predictive Maintenance – Heat Exchanger Fouling\n"
             "LSTM vs Baselines", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

# ── (a) Fouling growth ───────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
t_h = np.arange(len(y_true))
ax1.plot(t_h, y_true  * 1e4, "k-",  lw=1.5, label="Ground Truth", alpha=0.9)
ax1.plot(t_h, lstm_pred*1e4, "b--", lw=1.5, label="LSTM",         alpha=0.85)
ax1.plot(t_h, rf_pred  *1e4, "g-.", lw=1.2, label="RF",           alpha=0.75)
ax1.plot(t_h, xgb_pred *1e4, "r:",  lw=1.2, label="XGBoost",      alpha=0.75)
ax1.plot(t_h, lin_pred *1e4, "m-",  lw=1.0, label="Linear",       alpha=0.60)
ax1.axhline(y=0.0003*1e4, color="darkred", ls="--", lw=1,
            label="Alert threshold (3×10⁻⁴)")
ax1.set_xlabel("Test Hour")
ax1.set_ylabel("Rf  [×10⁻⁴ m²K/W]")
ax1.set_title("(a) Fouling Resistance – Predicted vs Ground Truth")
ax1.legend(fontsize=8, ncol=3)
ax1.grid(True, alpha=0.3)

# ── (b) Training loss ────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(history.history["loss"],     label="Train loss")
ax2.plot(history.history["val_loss"], label="Val loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("MSE (scaled)")
ax2.set_title("(b) LSTM Training Curve")
ax2.legend()
ax2.grid(True, alpha=0.3)

# ── (c) Scatter: LSTM predicted vs actual ────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.scatter(y_true * 1e4, lstm_pred * 1e4, s=4, alpha=0.4, color="steelblue")
lims = [min(y_true.min(), lstm_pred.min()) * 1e4,
        max(y_true.max(), lstm_pred.max()) * 1e4]
ax3.plot(lims, lims, "r--", lw=1, label="Perfect fit")
ax3.set_xlabel("Actual Rf [×10⁻⁴ m²K/W]")
ax3.set_ylabel("Predicted Rf [×10⁻⁴ m²K/W]")
ax3.set_title("(c) LSTM: Predicted vs Actual")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

plt.savefig("plots/fouling_predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → plots/fouling_predictions.png")

# ── Feature importance from RF ───────────────────────────────────────────────
feat_labels = []
for f in FEATURE_COLS:
    for t in range(SEQ_LEN):
        feat_labels.append(f"{f}_t-{SEQ_LEN - t}")

importances = rf.feature_importances_
# aggregate by feature name
feat_agg = {}
for i, lbl in enumerate(feat_labels):
    name = lbl.rsplit("_t-", 1)[0]
    feat_agg[name] = feat_agg.get(name, 0) + importances[i]

fi_df = pd.Series(feat_agg).sort_values(ascending=True)
fig2, ax = plt.subplots(figsize=(8, 5))
fi_df.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Random Forest – Aggregated Feature Importance")
ax.set_xlabel("Importance")
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → plots/feature_importance.png")

print("\n[✓] All done.")
