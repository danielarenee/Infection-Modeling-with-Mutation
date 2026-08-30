#!/usr/bin/env python3
"""
Side-by-side SIRC vs SIRCm (prevalence-driven) comparison under seasonal forcing
Left panel  : Regime 1 (e.g., β0 = 1200, η = 0.07, short time window in months)
Right panel : Regime 2 (e.g., β0 = 400,  η = 0.18, multi-year window in years)
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

# -- Import sircmw from SIRCm_prevalence utils 
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "SIRCm_prevalence"))

from sircmw_I_utils import sircmw, get_algebraic_equilibria

# CONFIGURATION ================================================================

# --- Biological baseline parameters 
MU    = 0.02
ALPHA = 365.0 / 3.0
DELTA = 1.0 / 1.61
GAMMA = 0.35
SIGMA = 0.07874

def get_sirc_endemic_I(beta_val):
    """Computes I*, the SIRC endemic equilibrium fraction at tilde_eps = 0."""
    p_base = {
        'beta0': beta_val, 'sigma': SIGMA, 'mu': MU, 'alpha': ALPHA,
        'delta': DELTA, 'gamma': GAMMA, 'si_0': 1.0
    }
    eqs = get_algebraic_equilibria(0.0, p_base)
    endemic = [eq for eq in eqs if eq[1] > 1e-5]
    if endemic:
        return endemic[0][1]
    # Fallback to standard baseline I* if below transcritical
    return 0.00114321

# --- Initial conditions (S, I, R, C) 
Y0 = np.array([0.20, 0.001, 0.499, 0.30])

# --- Default feedback values to compare (list of tilde_eps) 
# You can set 1 value [0.3], or multiple e.g. [0.2, 0.5, 0.8]
TILDE_EPS_VALUES = [0.3]

# --- Left Panel Configuration 
LEFT = dict(
    BETA0            = 1200.0,   # Contact rate
    ETA              = 0.07,     # Seasonal forcing amplitude
    YEARS            = 100,      # Total integration duration (years)
    PLOT_YEARS       = 1,        # Years to display in the plot (<=1 uses months)
    TILDE_EPS_VALUES = None,     # None = use global TILDE_EPS_VALUES, or custom list
)

# --- Right Panel Configuration 
RIGHT = dict(
    BETA0            = 400.0,    # Contact rate
    ETA              = 0.18,     # Seasonal forcing amplitude
    YEARS            = 100,      # Total integration duration (years)
    PLOT_YEARS       = 20,       # Years to display in the plot
    TILDE_EPS_VALUES = None,     # None = use global TILDE_EPS_VALUES, or custom list
)

# --- Solver settings 
SOLVER_KW = dict(method='DOP853', rtol=1e-6, atol=1e-9, max_step=1/365)

# --- Colors 
COLOR_SIRC = '#E64B35'
BLUE_LIGHT = '#87bce6'
BLUE_BASE  = '#1f77b4'
BLUE_DARK  = '#114467'


def get_blue_gradient(n):
    """Generate n shades of blue centered around matplotlib blue"""
    if n == 1:
        return [BLUE_BASE]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'sircm_blues', [BLUE_LIGHT, BLUE_BASE, BLUE_DARK]
    )
    return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]


def integrate_panel(cfg):
    """Integrate SIRC (eps=0) and all SIRCm variants for one panel configuration"""
    beta0 = cfg['BETA0']
    eta   = cfg['ETA']
    years = cfg['YEARS']
    pyrs  = cfg['PLOT_YEARS']
    eps_list = cfg['TILDE_EPS_VALUES'] if cfg.get('TILDE_EPS_VALUES') is not None else TILDE_EPS_VALUES
    transient = years - pyrs

    # SIRC endemic equilibrium scaling factor (I*)
    I_star = get_sirc_endemic_I(beta0)

    # 1. SIRC baseline (eps = 0)
    p_sirc = dict(
        beta0=beta0, eta=eta, eps=0.0,
        mu=MU, alpha=ALPHA, delta=DELTA, gamma=GAMMA, sigma=SIGMA
    )
    sol_sirc = solve_ivp(sircmw, (0, years), Y0, args=(p_sirc,), **SOLVER_KW)
    if not sol_sirc.success:
        raise RuntimeError(f"SIRC integration failed: {sol_sirc.message}")

    mask_s = sol_sirc.t >= transient
    t_s = sol_sirc.t[mask_s] - transient
    I_s = sol_sirc.y[1, mask_s]

    # 2. SIRCm variants
    mwi_results = []
    for te in eps_list:
        eps = te / I_star
        p_mwi = dict(
            beta0=beta0, eta=eta, eps=eps,
            mu=MU, alpha=ALPHA, delta=DELTA, gamma=GAMMA, sigma=SIGMA
        )
        sol_m = solve_ivp(sircmw, (0, years), Y0, args=(p_mwi,), **SOLVER_KW)
        if not sol_m.success:
            raise RuntimeError(f"SIRCm integration failed for eps={te}: {sol_m.message}")

        mask_m = sol_m.t >= transient
        t_m = sol_m.t[mask_m] - transient
        I_m = sol_m.y[1, mask_m]
        mwi_results.append((te, t_m, I_m))

    return t_s, I_s, mwi_results


def draw_panel(ax, t_s, I_s, mwi_results, cfg, label=""):
    """Render time series onto a single panel."""
    pyrs = cfg['PLOT_YEARS']
    use_months = (pyrs <= 1.0)
    scale = 12.0 if use_months else 1.0

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.grid(True, linestyle='-', linewidth=0.5, color='#e5e5e5')

    # SIRC baseline
    ax.plot(t_s * scale, I_s, color=COLOR_SIRC, linewidth=1.6, zorder=3)

    # SIRCm curves
    blues = get_blue_gradient(len(mwi_results))
    for (te, t_m, I_m), col in zip(mwi_results, blues):
        ax.plot(t_m * scale, I_m, color=col, linewidth=1.6, zorder=4)

    # Axis labels and limits
    xlabel = 'Time (months)' if use_months else 'Time (years)'
    ax.set_xlabel(xlabel, fontsize=11, labelpad=6)
    ax.set_ylabel('Prevalence $I(t)$', fontsize=11)
    ax.set_xlim(0, pyrs * scale)

    all_I = [I_s] + [res[2] for res in mwi_results]
    max_val = max(np.max(arr) for arr in all_I)
    ax.set_ylim(0, max_val * 1.15)

    if use_months:
        tick_step = 3 if pyrs >= 0.75 else 1
        ax.set_xticks(np.arange(0, pyrs * 12.0 + 0.1, tick_step))

    # Annotations
    ax.text(0.03, 0.97, label, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')
    ax.text(0.97, 0.97,
            rf'$\beta_0={cfg["BETA0"]:.0f}$,  $\eta={cfg["ETA"]}$',
            transform=ax.transAxes, fontsize=10, va='top', ha='right')


def main():
    print("Integrating left panel (Regime 1)...")
    t_s_L, I_s_L, mwi_L = integrate_panel(LEFT)

    print("Integrating right panel (Regime 2)...")
    t_s_R, I_s_R, mwi_R = integrate_panel(RIGHT)

    fig = plt.figure(figsize=(13, 5), dpi=300)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.6],
                            wspace=0.30, left=0.07, right=0.97,
                            top=0.82, bottom=0.13)
    ax_L = fig.add_subplot(gs[0])
    ax_R = fig.add_subplot(gs[1])

    draw_panel(ax_L, t_s_L, I_s_L, mwi_L, LEFT,  label="(a)")
    draw_panel(ax_R, t_s_R, I_s_R, mwi_R, RIGHT, label="(b)")

    legend_handles = [
        Line2D([0], [0], color=COLOR_SIRC, linewidth=1.8, label=r'SIRC ($\varepsilon = 0$)')
    ]

    # Collect all unique epsilons across both panels
    all_eps = []
    for res in mwi_L + mwi_R:
        te = res[0]
        if te not in all_eps:
            all_eps.append(te)

    blues = get_blue_gradient(len(all_eps))
    for te, col in zip(all_eps, blues):
        legend_handles.append(
            Line2D([0], [0], color=col, linewidth=1.8,
                   label=rf'SIRCm (prevalence variant) $\tilde{{\varepsilon}} = {te}$')
        )

    fig.legend(handles=legend_handles,
               loc='upper center', ncol=len(legend_handles),
               frameon=True, edgecolor='#e5e5e5',
               fontsize=12, bbox_to_anchor=(0.52, 0.98))

    # Save PNG output
    out_png = SCRIPT_DIR / "sircmwI_vs_sirc_comparison.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved plot to: {out_png}")


if __name__ == "__main__":
    main()
