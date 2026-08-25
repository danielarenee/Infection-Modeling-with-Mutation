#!/usr/bin/env python3
"""
Side-by-side SIRC vs SIRCmw_I comparison — two-panel figure
=============================================================
Left panel  : 20 year window, x-axis in years
Right panel : 20 year window, x-axis in years
Three SIRCmw_I curves (three ε̃ values, blue shades) + one SIRC curve (red).
Shared colour legend centred above both panels.

Edit the CONFIG block below to change parameters.
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

# -- path setup ---------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[1]
SIRC_ROOT = ROOT.parent / "SIRC"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SIRC_ROOT))

from sircmw_I_utils import sircmw
from sirc_utils    import sirc

# =============================================================================
#                           *** CONFIG ***
# =============================================================================

# --- Shared biological parameters (Casagrandi baseline) ----------------------
MU    = 0.02
ALPHA = 365.0 / 3.0
DELTA = 1.0 / 1.61
GAMMA = 0.35
SIGMA = 0.07874


# --- Reference scale for ε̃ → ε conversion -----------------------------------
I_0 = 0.00114321  # eps = tilde_eps / I_0

# --- Initial conditions (shared across all runs) -----------------------------
Y0 = np.array([0.20, 0.001, 0.499, 0.30])   # (S, I, R, C)

# --- SIRCmw_I feedback values (three shades of blue) ------------------------
TILDE_EPS_VALUES = [0.6]   # ε̃  values to compare
#TILDE_EPS_VALUES = [0.3, 0.5, 0.8]   # ε̃  values to compare


# --- Left panel parameters ---------------------------------------------------
LEFT = dict(
    BETA0      = 1200.0,   # contact rate
    ETA        = 0.07,     # seasonal forcing amplitude
    YEARS      = 100,      # total integration time (years)
    PLOT_YEARS = 5,       # years to display
)

# --- Right panel parameters --------------------------------------------------
RIGHT = dict(
    BETA0      = 400.0,    # contact rate
    ETA        = 0.18,     # seasonal forcing amplitude
    YEARS      = 100,      # total integration time (years)
    PLOT_YEARS = 20,       # years to display
)

# --- Solver ------------------------------------------------------------------
# Use smaller max_step for left panel high β0 to capture faster oscillations accurately
SOLVER_KW = dict(method='DOP853', rtol=1e-6, atol=1e-9, max_step=1/365)

# --- Colours -----------------------------------------------------------------
COLOR_SIRC  = '#E64B35'   # coral/orange — SIRC
BLUE_LIGHT  = '#87bce6'   # lightest blue in gradient
BLUE_DARK   = '#114467'   # darkest blue in gradient (dark shade of #1f77b4)

# Derived: auto-generate one colour per entry in TILDE_EPS_VALUES (do not edit)
_n = len(TILDE_EPS_VALUES)
if _n == 1:
    BLUES = ["#1f77b4"]
else:
    _cmap = mcolors.LinearSegmentedColormap.from_list(
        'sircmwi_blues', [BLUE_LIGHT, '#1f77b4', BLUE_DARK])
    BLUES = [mcolors.to_hex(_cmap(i / (_n - 1))) for i in range(_n)]

# =============================================================================


def integrate_all(cfg):
    """Integrate SIRC + all SIRCmw_I variants for one panel config.
    Returns (t_sirc, I_sirc, [(t_mwi, I_mwi), ...]) with transient trimmed.
    """
    β0    = cfg['BETA0']
    eta   = cfg['ETA']
    years = cfg['YEARS']
    pyrs  = cfg['PLOT_YEARS']
    trans = years - pyrs

    p_sirc = dict(beta0=β0, eps=eta,
                  mu=MU, alpha=ALPHA, delta=DELTA, gamma=GAMMA, sigma=SIGMA)

    sol_s = solve_ivp(sirc, (0, years), Y0, args=(p_sirc,), **SOLVER_KW)
    if not sol_s.success:
        raise RuntimeError(f"SIRC failed: {sol_s.message}")

    mask_s = sol_s.t >= trans
    t_s = sol_s.t[mask_s] - trans
    I_s = sol_s.y[1, mask_s]

    mwi_results = []
    for te in TILDE_EPS_VALUES:
        eps   = te / I_0
        p_mwi = dict(beta0=β0, eta=eta, eps=eps,
                     mu=MU, alpha=ALPHA, delta=DELTA, gamma=GAMMA, sigma=SIGMA)
        sol_m = solve_ivp(sircmw, (0, years), Y0, args=(p_mwi,), **SOLVER_KW)
        if not sol_m.success:
            raise RuntimeError(f"SIRCmw_I (ε̃={te}) failed: {sol_m.message}")
        mask_m = sol_m.t >= trans
        t_m = sol_m.t[mask_m] - trans
        I_m = sol_m.y[1, mask_m]
        mwi_results.append((t_m, I_m))

    return t_s, I_s, mwi_results


def get_cached_integration(cfg_name, cfg):
    """Load integration results from a local cache file if parameters match,
    otherwise run integration and save it.
    """
    eps_str = "_".join(f"{v:.4f}" for v in TILDE_EPS_VALUES)
    cache_name = f"cache_{cfg_name}_b{cfg['BETA0']}_e{cfg['ETA']}_y{cfg['YEARS']}_py{cfg['PLOT_YEARS']}_eps_{eps_str}.npz"
    cache_path = Path(__file__).resolve().parent / cache_name

    if cache_path.exists():
        print(f"Loading cached data for {cfg_name} from {cache_name} ...")
        try:
            data = np.load(cache_path)
            t_s = data['t_s']
            I_s = data['I_s']
            mwi_results = []
            for i in range(len(TILDE_EPS_VALUES)):
                mwi_results.append((data[f't_m_{i}'], data[f'I_m_{i}']))
            return t_s, I_s, mwi_results
        except Exception as e:
            print(f"Error loading cache: {e}. Re-integrating ...")

    print(f"No valid cache found for {cfg_name}. Integrating ...")
    t_s, I_s, mwi_results = integrate_all(cfg)

    # Save to cache
    try:
        save_dict = {'t_s': t_s, 'I_s': I_s}
        for i, (t_m, I_m) in enumerate(mwi_results):
            save_dict[f't_m_{i}'] = t_m
            save_dict[f'I_m_{i}'] = I_m
        np.savez(cache_path, **save_dict)
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return t_s, I_s, mwi_results


def draw_panel(ax, t_s, I_s, mwi_results, cfg,
               x_months=False, label=""):
    """Draw a single panel. x_months=True converts x-axis to months."""
    scale = 12 if x_months else 1

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.grid(True, linestyle='-', linewidth=0.5, color='#e5e5e5')

    # SIRC baseline
    ax.plot(t_s * scale, I_s, color=COLOR_SIRC, linewidth=1.6)

    # SIRCmw_I curves
    for (t_m, I_m), col in zip(mwi_results, BLUES):
        ax.plot(t_m * scale, I_m, color=col, linewidth=1.6)

    # axes
    xlabel = 'Time (months)' if x_months else 'Time (years)'
    ax.set_xlabel(xlabel, fontsize=11, labelpad=6)
    ax.set_ylabel('Prevalence $I(t)$', fontsize=11)
    ax.set_xlim(0, cfg['PLOT_YEARS'] * scale)

    i_all = np.concatenate([I_s] + [r[1] for r in mwi_results])
    ax.set_ylim(0, i_all.max() * 1.15)

    # month ticks on left panel
    if x_months:
        ax.set_xticks(range(0, cfg['PLOT_YEARS'] * 12 + 1, 3))

    # panel label + parameter annotation
    ax.text(0.03, 0.97, label, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')
    ax.text(0.97, 0.97,
            rf'$\beta_0={cfg["BETA0"]:.0f}$,  $\eta={cfg["ETA"]}$',
            transform=ax.transAxes, fontsize=9, va='top', ha='right')


def main():
    t_s_L, I_s_L, mwi_L = get_cached_integration("left", LEFT)
    t_s_R, I_s_R, mwi_R = get_cached_integration("right", RIGHT)

    # -- figure: equal-width panels -------------------------------------------
    fig = plt.figure(figsize=(13, 5), dpi=300)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1],
                            wspace=0.28, left=0.07, right=0.97,
                            top=0.82, bottom=0.13)
    ax_L = fig.add_subplot(gs[0])
    ax_R = fig.add_subplot(gs[1])

    draw_panel(ax_L, t_s_L, I_s_L, mwi_L, LEFT,  x_months=False, label="(a)")
    draw_panel(ax_R, t_s_R, I_s_R, mwi_R, RIGHT, x_months=False, label="(b)")

    # -- shared legend centred above both panels ------------------------------
    if len(TILDE_EPS_VALUES) > 1:
        legend_handles = [
            Line2D([0], [0], color=COLOR_SIRC, linewidth=1.8, label='SIRC  ($\\varepsilon = 0$)'),
            Line2D([0], [0], color=BLUE_DARK, linewidth=1.8,
                   label=rf'SIRCmw (prevalence driven-variant)  $\tilde{{\varepsilon}}$ increasing from {TILDE_EPS_VALUES[0]} to {TILDE_EPS_VALUES[-1]} (increasingly dark blue)')
        ]
    else:
        legend_handles = [
            Line2D([0], [0], color=COLOR_SIRC, linewidth=1.8, label='SIRC  ($\\varepsilon = 0$)'),
            Line2D([0], [0], color=BLUES[0], linewidth=1.8,
                   label=rf'SIRCmw (prevalence driven-variant)  $\tilde{{\varepsilon}} = {TILDE_EPS_VALUES[0]}$')
        ]

    fig.legend(handles=legend_handles,
               loc='upper center', ncol=len(legend_handles),
               frameon=True, edgecolor='#e5e5e5',
               fontsize=13, bbox_to_anchor=(0.52, 0.98))

    # -- save -----------------------------------------------------------------
    out_dir = Path(__file__).resolve().parent
    out_png = out_dir / "sircmwI_vs_sirc_comparison_20years.png"
    out_pdf = out_dir / "sircmwI_vs_sirc_comparison_20years.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    print(f"Saved -> {out_png}")
    print(f"Saved -> {out_pdf}")
    plt.close()


if __name__ == "__main__":
    main()
