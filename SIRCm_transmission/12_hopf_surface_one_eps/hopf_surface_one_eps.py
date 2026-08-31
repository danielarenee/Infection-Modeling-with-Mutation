#!/usr/bin/env python3
"""
3D Hopf bifurcation surface and 2D parameter slices (Transmission-driven variant)

Generates a 3-panel figure for the transmission-driven SIRCm model:
- Left panel: 3D Hopf bifurcation surface in (tilde_eps, sigma, beta0) space from continuation data.
- Middle panel: 2D stability slice in the (tilde_eps, beta0) plane at FIXED_SIGMA.
- Right panel: 2D stability slice in the (tilde_eps, sigma) plane at FIXED_BETA0.

OUTPUT: Figure 12
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

from sircmw_utils import (
    sircmw_jacobian,
    get_algebraic_equilibria,
    MU as mu,
    ALPHA as alpha,
    DELTA as delta,
    GAMMA as gamma,
    SI_0
)

# =============================================================================
# CONFIGURATION (Fixed parameters for 2D stability slices)
# =============================================================================
FIXED_SIGMA = 0.07874  # for (tilde_eps, beta0) sweep
FIXED_BETA0 = 600.0    # for (tilde_eps, sigma) sweep

TILDE_EPS_MIN, TILDE_EPS_MAX = 0.0, 2.0
BETA0_MIN, BETA0_MAX = 0.0, 2000.0
SIGMA_MIN, SIGMA_MAX = 0.0, 0.3
GRID_RESOLUTION_2D = 200

COLOR_DFE      = '#F3F1F5'
COLOR_STABLE   = '#e46c5c'
COLOR_UNSTABLE = '#fca636'
COLOR_3D_START = '#d6556d'
COLOR_3D_END   = '#fca636'

plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 15,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18,
    'legend.fontsize': 12
})


def check_stability_full(tilde_eps, beta_val, sigma_val):
    """Evaluates local stability of the transmission-driven SIRCm model."""
    if beta_val < (mu + alpha):
        return 0.0

    p = {
        'beta0': beta_val,
        'sigma': sigma_val,
        'mu': mu,
        'alpha': alpha,
        'delta': delta,
        'gamma': gamma,
        'si_0': SI_0
    }
    eqs = get_algebraic_equilibria(tilde_eps, p)
    if not eqs:
        return 0.0

    eq = max(eqs, key=lambda u: u[1])
    eps = tilde_eps / SI_0
    J = sircmw_jacobian(eq, eps, p=p)
    eigs = np.linalg.eigvals(J)
    max_real = np.max(np.real(eigs))
    return 1.0 if max_real < 0.0 else 2.0


def run_2d_sweep(x_grid, y_grid, sweep_type):
    Z = np.zeros((len(y_grid), len(x_grid)))
    for j, y_val in enumerate(y_grid):
        for i, x_val in enumerate(x_grid):
            if sweep_type == 'eps_beta':
                Z[j, i] = check_stability_full(x_val, y_val, FIXED_SIGMA)
            elif sweep_type == 'eps_sigma':
                Z[j, i] = check_stability_full(x_val, FIXED_BETA0, y_val)
    return Z


def main():
    csv_path = SCRIPT_DIR / "hopf_surface_eps_beta_sigma.csv"
    if not csv_path.exists():
        print(f"Error: Bifurcation data CSV not found at {csv_path}.")
        return

    print("Loading bifurcation data...")
    df = pd.read_csv(csv_path)

    # --- 3D Surface Reconstruction ---
    u_lin = np.linspace(0.0, 1.0, 300)
    sigmas = sorted(df["sigma"].unique())

    eps_mesh   = np.full((len(sigmas), len(u_lin)), np.nan)
    sig_mesh   = np.full((len(sigmas), len(u_lin)), np.nan)
    upper_grid = np.full((len(sigmas), len(u_lin)), np.nan)
    lower_grid = np.full((len(sigmas), len(u_lin)), np.nan)

    for i, sig in enumerate(sigmas):
        group = df[df["sigma"] == sig]
        if len(group) < 5:
            continue

        min_row     = group.loc[group["tilde_eps"].idxmin()]
        eps_vertex  = min_row["tilde_eps"]
        beta_vertex = min_row["beta0"]

        forward_pts  = group[group["branch"] == "forward"].sort_values("tilde_eps")
        backward_pts = group[group["branch"] == "backward"]
        b_seq  = backward_pts.reset_index(drop=True)
        v_idx  = b_seq["tilde_eps"].idxmin()
        b_upper = b_seq.iloc[:v_idx + 1]
        b_lower = b_seq.iloc[v_idx:]

        upper_combined = pd.concat([b_upper, forward_pts]).sort_values("tilde_eps").drop_duplicates(subset=["tilde_eps"])
        lower_combined = b_lower.sort_values("tilde_eps").drop_duplicates(subset=["tilde_eps"])

        eps_max = min(upper_combined["tilde_eps"].max(), lower_combined["tilde_eps"].max())

        f_upper = interp1d(upper_combined["tilde_eps"], upper_combined["beta0"], bounds_error=False, fill_value=np.nan)
        f_lower = interp1d(lower_combined["tilde_eps"], lower_combined["beta0"], bounds_error=False, fill_value=np.nan)

        for j, u in enumerate(u_lin):
            te = eps_vertex + u * (eps_max - eps_vertex)
            eps_mesh[i, j] = te
            sig_mesh[i, j] = sig
            if u == 0:
                upper_grid[i, j] = beta_vertex
                lower_grid[i, j] = beta_vertex
            else:
                upper_grid[i, j] = f_upper(te)
                lower_grid[i, j] = f_lower(te)

    # --- 2D Stability Sweeps ---
    eps_sweep = np.linspace(TILDE_EPS_MIN, TILDE_EPS_MAX, GRID_RESOLUTION_2D)
    beta_sweep = np.linspace(BETA0_MIN, BETA0_MAX, GRID_RESOLUTION_2D)
    sigma_sweep = np.linspace(SIGMA_MIN, SIGMA_MAX, GRID_RESOLUTION_2D)

    Z_eps_beta = run_2d_sweep(eps_sweep, beta_sweep, 'eps_beta')
    Z_eps_sigma = run_2d_sweep(eps_sweep, sigma_sweep, 'eps_sigma')

    # --- Plotting ---
    fig = plt.figure(figsize=(18, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 0.9, 0.9])

    levels = [-0.5, 0.5, 1.5, 2.5]
    cmap = mcolors.ListedColormap([COLOR_DFE, COLOR_STABLE, COLOR_UNSTABLE])
    warm_cmap = mcolors.LinearSegmentedColormap.from_list("warm_plasma", [COLOR_3D_START, COLOR_3D_END])

    # Left Panel: 3D Surface
    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax1.plot_surface(eps_mesh, sig_mesh, upper_grid, cmap=warm_cmap, alpha=0.85,
                     shade=True, edgecolor='none', rcount=100, ccount=100)
    ax1.plot_surface(eps_mesh, sig_mesh, lower_grid, cmap=warm_cmap, alpha=0.85,
                     shade=True, edgecolor='none', rcount=100, ccount=100)

    ax1.set_xlabel(r'$\tilde{\epsilon}$', labelpad=12)
    ax1.set_ylabel(r'$\sigma$', labelpad=12)
    ax1.set_zlabel(r'$\beta_0$', labelpad=18)
    ax1.set_xlim(TILDE_EPS_MIN, TILDE_EPS_MAX)
    ax1.set_ylim(SIGMA_MIN, SIGMA_MAX)
    ax1.set_zlim(BETA0_MIN, BETA0_MAX)
    ax1.set_xticks([0.0, 0.5, 1.0, 1.5])
    ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax1.zaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.view_init(elev=15, azim=-55)
    ax1.grid(True, alpha=0.2)

    # Middle Panel: Slice at FIXED_SIGMA
    ax2 = fig.add_subplot(gs[1])
    X_eb, Y_eb = np.meshgrid(eps_sweep, beta_sweep)
    ax2.contourf(X_eb, Y_eb, Z_eps_beta, levels=levels, cmap=cmap, alpha=0.95)
    ax2.contour(X_eb, Y_eb, Z_eps_beta, levels=[1.5], colors='white', linestyles='--', linewidths=2.2)
    ax2.axhline(y=mu + alpha, color='#555555', linestyle=':', linewidth=1.5)
    ax2.set_xlabel(r'$\tilde{\epsilon}$')
    ax2.set_ylabel(r'$\beta_0$')
    ax2.set_xlim(TILDE_EPS_MIN, TILDE_EPS_MAX)
    ax2.set_ylim(BETA0_MIN, BETA0_MAX)
    ax2.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax2.grid(True, alpha=0.25)
    ax2.set_title(rf"Slice at $\sigma = {FIXED_SIGMA}$", pad=10)
    ax2.set_box_aspect(1.0)

    # Right Panel: Slice at FIXED_BETA0
    ax3 = fig.add_subplot(gs[2])
    X_es, Y_es = np.meshgrid(eps_sweep, sigma_sweep)
    ax3.contourf(X_es, Y_es, Z_eps_sigma, levels=levels, cmap=cmap, alpha=0.95)
    ax3.contour(X_es, Y_es, Z_eps_sigma, levels=[1.5], colors='white', linestyles='--', linewidths=2.2)
    ax3.set_xlabel(r'$\tilde{\epsilon}$')
    ax3.set_ylabel(r'$\sigma$')
    ax3.set_xlim(TILDE_EPS_MIN, TILDE_EPS_MAX)
    ax3.set_ylim(SIGMA_MIN, SIGMA_MAX)
    ax3.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax3.grid(True, alpha=0.25)
    ax3.set_title(rf"Slice at $\beta_0 = {FIXED_BETA0:.1f}$", pad=10)
    ax3.set_box_aspect(1.0)

    dfe_patch = mpatches.Patch(color=COLOR_DFE, label="DFE stable ($R_0 < 1$)")
    stable_patch = mpatches.Patch(color=COLOR_STABLE, label="Stable endemic equilibrium")
    unstable_patch = mpatches.Patch(color=COLOR_UNSTABLE, label="Unstable endemic equilibrium")
    hopf_line = plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2.0, label="Hopf bifurcation boundary")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.40, top=0.82, bottom=0.15)
    pos1 = ax1.get_position()
    ax1.set_position([pos1.x0 - 0.01, pos1.y0 - 0.02, pos1.width * 1.15, pos1.height * 1.15])
    fig.legend(handles=[dfe_patch, stable_patch, unstable_patch, hopf_line],
               loc='upper right', bbox_to_anchor=(0.95, 0.99), ncol=4, frameon=True, fontsize=16, handlelength=2.0)

    fig_save_path = SCRIPT_DIR / "12_hopf_surface_one_eps.png"
    plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_save_path}")


if __name__ == "__main__":
    main()
