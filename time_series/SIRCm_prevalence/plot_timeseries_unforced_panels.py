#!/usr/bin/env python3
"""
SIRC vs. SIRCm unforced time series comparison

Plots unforced (η = 0) SIRC vs SIRCm under multiple feedback values (tilde_eps).
Allows selecting between prevalence-driven and transmission-driven variants at the top.

"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

# =============================================================================
#                        *** MODEL VARIANT SELECTION ***
# Choose between:
#   'prevalence'   : Prevalence-driven variant (1 + eps * I)
#   'transmission' : Transmission/infection-driven variant (1 + eps * S * I)
# =============================================================================
MODEL_VARIANT = 'prevalence'

# -- Path & Module imports based on selected variant --------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

if MODEL_VARIANT == 'prevalence':
    sys.path.insert(0, str(REPO_ROOT / "SIRCm_prevalence"))
    from sircmw_I_utils import (
        sircmw,
        integrate_with_reseeding,
        get_algebraic_equilibria
    )
    VARIANT_LABEL = "SIRCm (prevalence-driven variant)"
elif MODEL_VARIANT in ('transmission', 'infection'):
    sys.path.insert(0, str(REPO_ROOT / "SIRCm_transmission"))
    from sircmw_utils import (
        sircmw,
        integrate_with_reseeding,
        get_algebraic_equilibria
    )
    VARIANT_LABEL = "SIRCm (transmission-driven variant)"
else:
    raise ValueError(f"Unknown MODEL_VARIANT '{MODEL_VARIANT}'. Choose 'prevalence' or 'transmission'.")

# CONFIGURATION ================================================================

# --- Biological baseline parameters 
MU    = 0.02            # Birth / death rate
ALPHA = 365.0 / 3.0     # Recovery rate (~121.7 yr^-1)
DELTA = 1.0 / 1.61      # Loss of full immunity rate
GAMMA = 0.35            # Loss of partial immunity rate
SIGMA = 0.07874         # Cross-immunity factor
BETA0 = 600.0           # Baseline contact rate
ETA   = 0.0             # Seasonal forcing amplitude (0.0 = unforced)


def get_scale_factor(beta_val):
    """Computes reference scale at tilde_eps = 0:
    - I* for prevalence-driven variant
    - S* * I* for transmission-driven variant
    """
    p_base = {
        'beta0': beta_val, 'sigma': SIGMA, 'mu': MU, 'alpha': ALPHA,
        'delta': DELTA, 'gamma': GAMMA, 'si_0': 1.0
    }
    eqs = get_algebraic_equilibria(0.0, p_base)
    endemic = [eq for eq in eqs if eq[1] > 1e-5]
    if endemic:
        S_star, I_star, R_star, C_star = endemic[0]
        if MODEL_VARIANT == 'prevalence':
            return I_star
        else:
            return S_star * I_star
    return 0.00114321 if MODEL_VARIANT == 'prevalence' else 0.0002045


# --- Panels Configuration 
# Configure feedback value (TILDE_EPS) and displayed time window (PLOT_YEARS) per panel:
PANELS = [
    dict(TILDE_EPS=0.25, PLOT_YEARS=20.0),
    dict(TILDE_EPS=0.75, PLOT_YEARS=20.0),
    dict(TILDE_EPS=1, PLOT_YEARS=20.0),
]

# --- Initial conditions (S, I, R, C) 
Y0 = np.array([0.20, 0.001, 0.499, 0.30])

# --- Solver settings  
SOLVER_KW = dict(method='DOP853', rtol=1e-6, atol=1e-9, max_step=1/365)

# --- Colors 
COLOR_SIRC = '#E64B35'
BLUE_LIGHT = '#87bce6'         
BLUE_BASE  = '#1f77b4'         
BLUE_DARK  = '#114467'        


def get_blue_gradient(n):
    """Generate n shades of blue centered around matplotlib blue."""
    if n == 1:
        return [BLUE_BASE]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'sircm_blues', [BLUE_LIGHT, BLUE_BASE, BLUE_DARK]
    )
    return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]


def main():
    # 1. Compute scaling factor dynamically at tilde_eps = 0
    scale_factor = get_scale_factor(BETA0)

    # 2. Setup subplot grid (sharex=False to allow independent year windows)
    num_panels = len(PANELS)
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]

    fig, axs = plt.subplots(1, num_panels, figsize=(5.2 * num_panels, 4.8),
                            sharex=False, sharey=False, dpi=300)
    if num_panels == 1:
        axs = [axs]

    blues = get_blue_gradient(num_panels)

    for i, p_cfg in enumerate(PANELS):
        ax = axs[i]
        te = p_cfg['TILDE_EPS']
        pyrs = p_cfg['PLOT_YEARS']
        eps = te / scale_factor
        col = blues[i]
        t_span = (0.0, pyrs)

        # 2a. Simulate SIRC baseline for this panel's time span
        p_sirc = dict(
            beta0=BETA0, eta=ETA, eps=0.0,
            mu=MU, alpha=ALPHA, delta=DELTA, gamma=GAMMA, sigma=SIGMA
        )
        t_sirc, Y_sirc, _ = integrate_with_reseeding(
            sircmw, t_span, Y0, p_sirc,
            threshold=1e-15, I_seed=1e-14,
            **SOLVER_KW
        )
        I_sirc = Y_sirc[1, :]

        # 2b. Simulate SIRCm for this panel
        print(f"Simulating {MODEL_VARIANT} SIRCm (panel {i+1}) with tilde_eps = {te:.2f}, {pyrs} years...")
        p_mwi = dict(
            beta0=BETA0, eta=ETA, eps=eps,
            mu=MU, alpha=ALPHA, delta=DELTA, gamma=GAMMA, sigma=SIGMA
        )
        t_mod, Y_mod, _ = integrate_with_reseeding(
            sircmw, t_span, Y0, p_mwi,
            threshold=1e-15, I_seed=1e-14,
            **SOLVER_KW
        )
        I_mod = Y_mod[1, :]

        # Plot curves
        ax.plot(t_sirc, I_sirc, color=COLOR_SIRC, linewidth=1.6, alpha=0.9, zorder=3)
        line_mwi, = ax.plot(t_mod, I_mod, color=col, linewidth=1.6, zorder=4)

        # Panel styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.grid(True, linestyle='-', linewidth=0.5, color='#e5e5e5')

        ax.set_xlim(0, pyrs)
        
        # Adaptive tick step
        if pyrs <= 5:
            step = 1
        elif pyrs <= 20:
            step = 5
        elif pyrs <= 50:
            step = 10
        else:
            step = 20
        ax.set_xticks(np.arange(0, int(pyrs) + 1, step))

        max_y = max(np.max(I_sirc), np.max(I_mod))
        ax.set_ylim(0, max_y * 1.15)
        ax.set_xlabel('Time (years)', fontsize=11, labelpad=6)

        # Local legend badge for epsilon
        ax.legend(
            handles=[line_mwi],
            labels=[rf'$\tilde{{\varepsilon}} = {te:.2f}$'],
            loc='upper right',
            frameon=True,
            edgecolor='#e5e5e5',
            fontsize=11
        )

        # Panel label (a), (b), (c)
        plabel = panel_labels[i] if i < len(panel_labels) else f"({chr(97+i)})"
        ax.text(0.03, 0.97, plabel, transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='top')

    axs[0].set_ylabel('Prevalence $I(t)$', fontsize=11)

    # 3. Global shared legend above panels
    legend_handles = [
        Line2D([0], [0], color=COLOR_SIRC, linewidth=1.8, label=r'SIRC ($\varepsilon = 0$)'),
        Line2D([0], [0], color=BLUE_BASE, linewidth=1.8, label=VARIANT_LABEL)
    ]
    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.50, 0.98),
        ncol=2,
        frameon=True,
        edgecolor='#e5e5e5',
        fontsize=12
    )

    plt.tight_layout(rect=[0, 0, 1, 0.91])

    # 4. Save PNG output
    out_png = SCRIPT_DIR / "timeseries_comparison_three_panels.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved plot to: {out_png}")


if __name__ == "__main__":
    main()
