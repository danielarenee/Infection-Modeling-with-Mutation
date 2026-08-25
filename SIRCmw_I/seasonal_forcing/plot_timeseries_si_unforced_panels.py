#!/usr/bin/env python3
"""
SIRC vs. SIRCmw (SI-feedback version) prevalence time series comparison
=======================================================================
Plots unforced/forced SIRC vs SIRCmw under multiple equal feedback values.
All customizable parameters are located in the CONFIG block below.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add paths to find sircmw_utils from SIRCmw folder
ROOT = Path(__file__).resolve().parents[1]
SIRCMW_ROOT = ROOT.parent / "SIRCmw"
sys.path.insert(0, str(SIRCMW_ROOT))

from sircmw_utils import (
    sircmw,
    integrate_with_reseeding
)

# =============================================================================
#                           *** CONFIG ***
# =============================================================================

# --- Biological model parameters (Casagrandi baseline) ----------------------
MU    = 0.02            # birth / death rate
ALPHA = 365.0 / 3.0     # recovery rate   (~121.7 yr^-1)
DELTA = 1.0 / 1.61      # R -> C waning rate
GAMMA = 0.35            # C -> S loss-of-immunity rate
SIGMA = 0.07874         # cross-immunity factor
BETA0 = 600.0           # baseline contact rate

# --- Seasonal forcing --------------------------------------------------------
ETA = 0.0               # forcing amplitude  (0 = unforced)

# --- SIRCmw (SI version) feedback -------------------------------------------
TILDE_EPS_VALUES = [0.25, 1.57, 1.59]   # ε̃ values to compare (one panel per value)
SI_0             = 0.0002045           # reference prevalence scale for conversion

# --- Simulation & Plotting ---------------------------------------------------
YEARS_LIST = [25.0, 10.0, 10.0]        # simulation/display length for each panel (years)
Y0    = np.array([0.2, 0.001, 0.499, 0.3])   # initial conditions (S, I, R, C)

# --- Solver ------------------------------------------------------------------
# method='Radau' is used to resolve numerical stiffness of SI version
SOLVER_KW = dict(method='Radau', rtol=1e-6, atol=1e-9)

# --- Colours -----------------------------------------------------------------
COLOR_SIRC  = '#E64B35'   # warm coral/orange  — SIRC baseline
COLOR_SIRCM = '#1f77b4'   # standard matplotlib blue  — SIRCm

# =============================================================================


def main():
    # Simulation time grid (simulate baseline up to the maximum required years)
    max_years = max(YEARS_LIST)
    t_span = (0.0, max_years)

    # 1. Simulate the SIRC baseline (tilde_eps = 0.0)
    print("Simulating SIRC baseline...")
    p_sirc = dict(beta0=BETA0, eta=ETA, eps1=0.0, eps2=0.0,
                  mu=MU, alpha=ALPHA, delta=DELTA, gamma=GAMMA, sigma=SIGMA)
    t_sirc, Y_sirc, _ = integrate_with_reseeding(
        sircmw, t_span, Y0, p_sirc,
        threshold=1e-15, I_seed=1e-14,
        **SOLVER_KW
    )

    # 2. Setup the side-by-side subplot panels (1 row, N columns, sharey=False)
    num_panels = len(TILDE_EPS_VALUES)
    fig, axs = plt.subplots(1, num_panels, figsize=(5.0 * num_panels, 5.0),
                            sharex=False, sharey=False, dpi=300)
    
    # If only one panel is plotted, axs must be wrapped in a list
    if num_panels == 1:
        axs = [axs]
        
    sirc_handle = None

    for i, te in enumerate(TILDE_EPS_VALUES):
        ax = axs[i]
        panel_years = YEARS_LIST[i]
        lw = 2.0 if i == 0 else 0.5  # make lines thinner for middle and right panels
        
        # Plot SIRC baseline in the background (trimmed to panel years)
        mask_sirc = t_sirc <= panel_years
        line_sirc, = ax.plot(t_sirc[mask_sirc], Y_sirc[1, mask_sirc], color=COLOR_SIRC, alpha=1.0, linewidth=lw)
        if i == 0:
            sirc_handle = line_sirc
            
        # Simulate and plot SIRCmw for the current epsilon value
        print(f"Simulating SIRCmw (SI) with tilde_eps = {te:.2f} for {panel_years} years...")
        eps = te / SI_0
        p_mod = dict(beta0=BETA0, eta=ETA, eps1=eps, eps2=eps,
                     mu=MU, alpha=ALPHA, delta=DELTA, gamma=GAMMA, sigma=SIGMA)
        t_span_mod = (0.0, panel_years)
        t_mod, Y_mod, _ = integrate_with_reseeding(
            sircmw, t_span_mod, Y0, p_mod,
            threshold=1e-15, I_seed=1e-14,
            **SOLVER_KW
        )
        
        # Plot SIRCmw in the foreground (fully opaque)
        line_sircmw, = ax.plot(t_mod, Y_mod[1, :], color=COLOR_SIRCM, alpha=1.0, linewidth=lw)
        
        # Add local legend inside the subplot with just the value of tilde_eps
        ax.legend(
            handles=[line_sircmw], 
            labels=[f"$\\tilde{{\\varepsilon}} = {te:.2f}$"], 
            loc='upper right', 
            frameon=True, 
            edgecolor='#e5e5e5',
            fontsize=12
        )
        
        # Subplot styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.grid(True, linestyle='-', linewidth=0.5, color='#e5e5e5')
        ax.set_xlim(0, panel_years)
        ax.set_xticks(range(0, int(panel_years) + 1, 5))
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel('Time (years)', fontsize=11, labelpad=8)
        
    axs[0].set_ylabel('Prevalence I(t)', fontsize=11)

    # Global legend for SIRC and SIRCm above the subplots (on top of all panels)
    from matplotlib.lines import Line2D
    sirc_legend_handle = Line2D([0], [0], color=COLOR_SIRC, alpha=1.0, linewidth=2.0)
    sircm_legend_handle = Line2D([0], [0], color=COLOR_SIRCM, alpha=1.0, linewidth=2.0)
    fig.legend(
        handles=[sirc_legend_handle, sircm_legend_handle],
        labels=['SIRC', 'SIRCm (transmission-driven variant)'],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=False,
        fontsize=14
    )

    # Adjust layout to leave room for the global legend at the top
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # Save output plots (both PNG and PDF)
    out_dir = Path(__file__).resolve().parent
    save_path_png = out_dir / "timeseries_si_comparison_three_panels.png"
    plt.savefig(save_path_png, dpi=300)
    print(f"Saved three-panel comparison plot to {save_path_png.resolve()}")

    save_path_pdf = out_dir / "timeseries_si_comparison_three_panels.pdf"
    plt.savefig(save_path_pdf, dpi=300)
    print(f"Saved three-panel comparison plot to {save_path_pdf.resolve()}")
    plt.close()


if __name__ == "__main__":
    main()
