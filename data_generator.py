"""
data_generator.py
Simulates heat exchanger fouling data using ε-NTU method.
Based on: Predictive Maintenance for Heat Exchanger Fouling Using AI/ML
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
RHO_WATER   = 1000.0   # kg/m³
CP_WATER    = 4186.0   # J/(kg·K)
RHO_FLUE    = 1.2      # kg/m³
CP_FLUE     = 1005.0   # J/(kg·K)
D_i         = 0.025    # m  inner tube diameter
L           = 5.0      # m  tube length
N_TUBES     = 20
U_CLEAN     = 800.0    # W/(m²·K)  overall HTC (clean)
K_F         = 2.3      # W/(m·K)   fouling layer conductivity (calcium salts)


def fouling_ode(Rf, t, phi, tau):
    """Asymptotic fouling model: dRf/dt = (R∞ - Rf) / τ"""
    R_inf = phi * tau
    return (R_inf - Rf) / tau


def simulate_fouling(
    n_hours: int = 720,         # 30-day run
    m_dot_flue: float = 6.0,    # kg/s  hot-side mass flow
    T_fi: float = 210.0,        # °C    hot-side inlet
    m_dot_water: float = 8.0,   # kg/s  cold-side mass flow
    T_ci: float = 25.0,         # °C    cold-side inlet
    phi: float = 1.5e-7,        # fouling rate constant  m²K/(W·s)
    tau: float = 3e6,           # time constant  s
    noise_std: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a time-series of heat exchanger sensor readings
    with realistic fouling growth and Gaussian measurement noise.
    """
    rng = np.random.default_rng(seed)
    t_seconds = np.arange(0, n_hours * 3600, 3600, dtype=float)

    # ── Fouling resistance over time (ODE solution) ──────────────────────
    Rf_t = odeint(fouling_ode, y0=0.0, t=t_seconds, args=(phi, tau)).ravel()

    records = []
    for i, (t_s, Rf) in enumerate(zip(t_seconds, Rf_t)):
        # ── ε-NTU analysis ────────────────────────────────────────────────
        A = np.pi * D_i * L * N_TUBES
        # Effective U accounting for fouling
        U_eff = 1.0 / (1.0 / U_CLEAN + Rf)

        C_hot  = m_dot_flue  * CP_FLUE
        C_cold = m_dot_water * CP_WATER
        C_min  = min(C_hot, C_cold)
        C_max  = max(C_hot, C_cold)
        C_r    = C_min / C_max

        NTU = U_eff * A / C_min
        # Counter-flow effectiveness
        if abs(C_r - 1.0) < 1e-6:
            eps = NTU / (1 + NTU)
        else:
            eps = (1 - np.exp(-NTU * (1 - C_r))) / (1 - C_r * np.exp(-NTU * (1 - C_r)))

        q_max = C_min * (T_fi - T_ci)
        q     = eps * q_max

        T_fo = T_fi  - q / C_hot
        T_co = T_ci  + q / C_cold

        # Reynold's number (hot-side, approx)
        mu     = 1.8e-5   # Pa·s  air viscosity
        Re     = (4 * m_dot_flue / N_TUBES) / (np.pi * D_i * mu)
        Pr     = 0.71     # air Prandtl number

        # ── Add measurement noise ─────────────────────────────────────────
        def n(scale=1.0):
            return rng.normal(0, noise_std * scale)

        records.append({
            "timestamp":     pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i),
            "T_hot_in":      T_fi  + n(5),
            "T_hot_out":     T_fo  + n(3),
            "T_cold_in":     T_ci  + n(2),
            "T_cold_out":    T_co  + n(3),
            "flow_hot":      m_dot_flue  + n(0.2),
            "flow_cold":     m_dot_water + n(0.2),
            "NTU":           NTU,
            "Re":            Re,
            "Pr":            Pr,
            "effectiveness": eps,
            "Rf":            Rf,
        })

    df = pd.DataFrame(records)
    # Derived features
    df["delta_T_hot"]  = df["T_hot_in"]  - df["T_hot_out"]
    df["delta_T_cold"] = df["T_cold_out"] - df["T_cold_in"]
    df["LMTD"]         = (df["delta_T_hot"] - df["delta_T_cold"]) / np.log(
        (df["T_hot_in"] - df["T_cold_out"] + 1e-6) /
        (df["T_hot_out"] - df["T_cold_in"] + 1e-6)
    )
    return df


if __name__ == "__main__":
    df = simulate_fouling()
    df.to_csv("data/fouling_simulation.csv", index=False)
    print(df.head(10).to_string())
    print(f"\nFouling range: {df['Rf'].min():.2e} – {df['Rf'].max():.2e} m²K/W")
