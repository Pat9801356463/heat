"""
dashboard.py
Streamlit real-time monitoring dashboard.

Run:  streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from data_generator import simulate_fouling

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heat Exchanger Fouling Monitor",
    page_icon="🔥",
    layout="wide",
)

st.title("🔥 Heat Exchanger Fouling – Predictive Maintenance Dashboard")
st.caption("LSTM-based real-time Rf monitoring | IIT Guwahati | CL653")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Simulation Parameters")
    n_hours   = st.slider("Run length (hours)", 240, 1440, 720, 24)
    m_dot_hot = st.slider("Hot-side flow (kg/s)", 4.0, 8.5, 6.0, 0.5)
    T_fi      = st.slider("Hot-side inlet T (°C)", 170, 250, 210, 5)
    phi       = st.select_slider("Fouling rate φ",
                    options=[5e-8, 1e-7, 1.5e-7, 3e-7, 5e-7],
                    value=1.5e-7,
                    format_func=lambda x: f"{x:.1e}")
    noise     = st.slider("Sensor noise std", 0.001, 0.05, 0.01, 0.001)
    run_btn   = st.button("▶ Run Simulation", type="primary")

ALERT_THRESHOLD = 3e-4   # m²K/W

# ─────────────────────────────────────────────────────────────────────────────
# Load / generate data
# ─────────────────────────────────────────────────────────────────────────────
if "df" not in st.session_state or run_btn:
    with st.spinner("Simulating …"):
        st.session_state["df"] = simulate_fouling(
            n_hours=n_hours, m_dot_flue=m_dot_hot,
            T_fi=T_fi, phi=phi, noise_std=noise,
        )

df: pd.DataFrame = st.session_state["df"]

# ─────────────────────────────────────────────────────────────────────────────
# KPI cards
# ─────────────────────────────────────────────────────────────────────────────
Rf_last  = df["Rf"].iloc[-1]
Rf_max   = df["Rf"].max()
eps_last = df["effectiveness"].iloc[-1]
alert    = Rf_last > ALERT_THRESHOLD

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Rf (m²K/W)",  f"{Rf_last:.3e}", delta=f"{(Rf_last/df['Rf'].iloc[0]-1)*100:.1f}% from start")
c2.metric("Peak Rf (m²K/W)",     f"{Rf_max:.3e}")
c3.metric("Effectiveness ε",     f"{eps_last:.3%}", delta=f"{(eps_last/df['effectiveness'].iloc[0]-1)*100:.1f}%")
c4.metric("Alert Status",        "⚠️ FOUL" if alert else "✅ OK",
          delta="Threshold exceeded" if alert else "Within limits",
          delta_color="inverse")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Fouling Resistance Over Time")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["timestamp"], df["Rf"] * 1e4, lw=1.5, color="steelblue",
            label="Rf (simulated)")
    ax.axhline(ALERT_THRESHOLD * 1e4, color="red", ls="--", lw=1.2,
               label=f"Alert threshold ({ALERT_THRESHOLD:.1e} m²K/W)")

    # Shade fouled region
    ax.fill_between(df["timestamp"], df["Rf"] * 1e4,
                    ALERT_THRESHOLD * 1e4,
                    where=df["Rf"] > ALERT_THRESHOLD,
                    alpha=0.2, color="red", label="Fouled zone")

    ax.set_ylabel("Rf  [×10⁻⁴ m²K/W]")
    ax.set_xlabel("Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_right:
    st.subheader("Sensor Snapshot")
    last = df.iloc[-1]
    snap = pd.DataFrame({
        "Sensor": ["T_hot_in", "T_hot_out", "T_cold_in", "T_cold_out",
                   "flow_hot", "flow_cold", "NTU", "Re"],
        "Value":  [f"{last['T_hot_in']:.1f} °C",
                   f"{last['T_hot_out']:.1f} °C",
                   f"{last['T_cold_in']:.1f} °C",
                   f"{last['T_cold_out']:.1f} °C",
                   f"{last['flow_hot']:.2f} kg/s",
                   f"{last['flow_cold']:.2f} kg/s",
                   f"{last['NTU']:.3f}",
                   f"{last['Re']:.0f}"],
    })
    st.dataframe(snap, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Temperature profiles
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Temperature Profiles")
fig2, axes = plt.subplots(1, 2, figsize=(14, 3))

axes[0].plot(df["timestamp"], df["T_hot_in"],  label="T_hot_in")
axes[0].plot(df["timestamp"], df["T_hot_out"], label="T_hot_out")
axes[0].set_ylabel("°C"); axes[0].set_title("Hot-side temperatures")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

axes[1].plot(df["timestamp"], df["T_cold_out"], label="T_cold_out", color="orange")
axes[1].plot(df["timestamp"], df["T_cold_in"],  label="T_cold_in",  color="lightblue")
axes[1].set_ylabel("°C"); axes[1].set_title("Cold-side temperatures")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig2)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Data table
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("📋 View Raw Data"):
    st.dataframe(df.tail(50), use_container_width=True)
    st.download_button("⬇ Download CSV", df.to_csv(index=False),
                       file_name="fouling_data.csv", mime="text/csv")
