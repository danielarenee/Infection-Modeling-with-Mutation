#!/usr/bin/env python3
"""
Seasonally-forced SIRC vs SIRCmw_I — time-series comparison
============================================================
Plots prevalence I(t) for the original SIRC model and one SIRCmw_I run.
All tunable parameters live in the CONFIG block below.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -- path setup ---------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[1]   # .../SIRCmw_I/
SIRC_ROOT = ROOT.parent / "SIRC"                  # .../SIRC/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SIRC_ROOT))

from sircmw_I_utils import sircmw   # SIRCmw_I ODE  (I-feedback, eps1=eps2=eps)
from sirc_utils    import sirc      # Original SIRC ODE

# =============================================================================
#                           *** CONFIG ***
# =============================================================================

# --- Model parameters (Casagrandi baseline) ----------------------------------
MU    = 0.02            # birth / death rate
ALPHA = 365.0 / 3.0     # recovery rate   (~121.7 yr^-1)
DELTA = 1.0 / 1.6      # R -> C waning rate
GAMMA = 0.35            # C -> S loss-of-immunity rate
SIGMA = 0.07874         # cross-immunity factor
BETA0 = 1200.0           # baseline contact rate

# --- Seasonal forcing --------------------------------------------------------
ETA = 0.07            # forcing amplitude  (0 = unforced)

# --- SIRCmw_I feedback -------------------------------------------------------
# Set TILDE_EPS (the scaled, dimensionless feedback strength ε̃).
# The script converts to the physical coefficient:  eps = tilde_eps / I_0
# TILDE_EPS = 0  <=>  standard SIRC  (no feedback)
TILDE_EPS = 10       # ε̃  (e.g. 0.0, 0.3, 0.5, 1.0, 1.5 ...)
I_0       = 0.00114321      # reference prevalence scale for the conversion

# --- Simulation --------------------------------------------------------------
YEARS      = 100     # total integration time (years)         ← full burn-in + plot window
PLOT_YEARS = 50    # years to show in the plot              ← last N years after transient
                   # transient discarded = YEARS - PLOT_YEARS
Y0    = np.array([0.20, 0.001, 0.499, 0.30])   # initial conditions (S, I, R, C)

# --- Solver ------------------------------------------------------------------
# max_step=1/365 ensures the solver resolves the annual seasonal cycle
SOLVER_KW = dict(method='DOP853', rtol=1e-6, atol=1e-9, max_step=1/365)

# --- Plot colours ------------------------------------------------------------
COLOR_SIRC   = '#E64B35'   # warm coral/orange  — SIRC baseline
COLOR_SIRCMW = '#3B5998'   # medium blue        — SIRCmw_I

# Derived (do not edit) -------------------------------------------------------
EPS       = TILDE_EPS / I_0          # physical feedback coefficient
TRANSIENT = YEARS - PLOT_YEARS       # years to drop before plotting
if PLOT_YEARS > YEARS:
    raise ValueError(f"PLOT_YEARS ({PLOT_YEARS}) cannot exceed YEARS ({YEARS})")
# =============================================================================


def integrate(rhs, y0, params):
    sol = solve_ivp(rhs, (0, YEARS), y0, args=(params,), **SOLVER_KW)
    if not sol.success:
        raise RuntimeError(f"Solver failed: {sol.message}")
    return sol.t, sol.y


def main():
    # -- build parameter dicts ------------------------------------------------
    # sirc_utils.sirc uses 'eps' as the seasonal forcing amplitude
    p_sirc = dict(beta0=BETA0, eps=ETA,
                  mu=MU, alpha=ALPHA, delta=DELTA, gamma=GAMMA, sigma=SIGMA)

    # sircmw_I_utils.sircmw uses 'eta' for forcing, 'eps' for I-feedback
    p_mwi  = dict(beta0=BETA0, eta=ETA, eps=EPS,
                  mu=MU, alpha=ALPHA, delta=DELTA, gamma=GAMMA, sigma=SIGMA)

    # -- integrate and trim transient -----------------------------------------
    print("Integrating SIRC ...")
    t_s, Y_s = integrate(sirc,   Y0, p_sirc)
    mask_s   = t_s >= TRANSIENT
    t_s, Y_s = t_s[mask_s] - TRANSIENT, Y_s[:, mask_s]

    print(f"Integrating SIRCmw_I (tilde_eps = {TILDE_EPS}, eps = {EPS:.1f}) ...")
    t_m, Y_m = integrate(sircmw, Y0, p_mwi)
    mask_m   = t_m >= TRANSIENT
    t_m, Y_m = t_m[mask_m] - TRANSIENT, Y_m[:, mask_m]

    # -- plot -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.grid(True, linestyle='-', linewidth=0.5, color='#e5e5e5')

    ax.plot(t_s, Y_s[1], color=COLOR_SIRC,   linewidth=1.5,
            label=r'SIRC  ($\varepsilon = 0$)')
    ax.plot(t_m, Y_m[1], color=COLOR_SIRCMW,  linewidth=1.5,
            label=rf'SIRCmw-I  ($\tilde{{\varepsilon}} = {TILDE_EPS}$)')

    ax.set_xlabel('Time (years, post-transient)', fontsize=12, labelpad=8)
    ax.set_ylabel('Prevalence $I(t)$', fontsize=12)
    ax.set_xlim(0, PLOT_YEARS)
    # y-axis: auto-scale with a small top margin
    i_all = np.concatenate([Y_s[1], Y_m[1]])
    ax.set_ylim(0, i_all.max() * 1.12)

    ax.set_title(
        rf'Seasonally-Forced SIRC vs SIRCmw-I  '
        rf'($\eta={ETA}$, $\beta_0={BETA0}$, $\varepsilon_1=\varepsilon_2=\varepsilon$)',
        fontsize=12, pad=10
    )
    ax.legend(loc='upper right', fontsize=10, frameon=True, edgecolor='#e5e5e5')
    plt.tight_layout()

    # -- save -----------------------------------------------------------------
    out_dir = Path(__file__).resolve().parent
    out_png = out_dir / "sircmwI_vs_sirc_timeseries.png"
    out_pdf = out_dir / "sircmwI_vs_sirc_timeseries.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    print(f"Saved -> {out_png}")
    print(f"Saved -> {out_pdf}")
    plt.close()


if __name__ == "__main__":
    main()
