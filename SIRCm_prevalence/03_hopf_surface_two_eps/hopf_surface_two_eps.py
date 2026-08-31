#!/usr/bin/env python3
"""
Hopf bifurcation surface and 2D stability slices (Prevalence-driven variant)

Generates a 3-panel figure for the prevalence-driven SIRCm model in (eps1, eps2) space:
- Left panel: 3D Hopf bifurcation surface in (eps1, eps2, beta0) space from continuation data.
- Middle panel: 2D stability slice in the (eps1, eps2) plane at SLICE_BETA_1.
- Right panel: 2D stability slice in the (eps1, eps2) plane at SLICE_BETA_2.

OUTPUT: Figure 03
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
from scipy.ndimage import gaussian_filter
from concurrent.futures import ProcessPoolExecutor

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

from sircmw_I_utils import (
    MU, ALPHA, DELTA, GAMMA, SIGMA,
    sircmw_jacobian, get_endemic_roots
)

# =============================================================================
# CONFIGURATION (Edit contact rates for the 2D stability slices)
# =============================================================================
SLICE_BETA_1 = 200.0
SLICE_BETA_2 = 2000.0

EPS1_MIN, EPS1_MAX = 0.0, 3.0
EPS2_MIN, EPS2_MAX = 0.0, 3.0
BETA0_MIN, BETA0_MAX = 100.0, 2000.0
GRID_RESOLUTION_2D = 200

scale_factor = 0.00114321

COLOR_STABLE   = '#e46c5c'
COLOR_UNSTABLE = '#fca636'
COLOR_3D_START = '#d6556d'
COLOR_3D_END   = '#fca636'

plt.rcParams.update({
    'font.size': 15,
    'axes.labelsize': 18,
    'axes.titlesize': 19,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.titlesize': 20,
    'legend.fontsize': 14
})


def get_stability_for_sweep(rel_eps1, rel_eps2, beta):
    if beta < (MU + ALPHA):
        return 1.0
    eps1 = rel_eps1 / scale_factor
    eps2 = rel_eps2 / scale_factor
    roots = get_endemic_roots(eps1, eps2, beta)
    if not roots:
        return 1.0
    for S, I, R, C in roots:
        J = sircmw_jacobian((S, I, R, C), eps1, eps2, p={'beta0': beta})
        if np.max(np.real(np.linalg.eigvals(J))) < 0.0:
            return 1.0
    return 0.0


def eval_stability_single(args):
    return get_stability_for_sweep(*args)


def main():
    csv_path = SCRIPT_DIR / "hopf_slices_eps1_indexed.csv"
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        sys.exit(1)

    print("Loading 3D Hopf continuation data...")
    df = pd.read_csv(csv_path)
    df = df[(df['eps1'] <= 3.0) & (df['eps2'] <= 3.0) | df['eps1'].isna() | df['eps2'].isna()]
    beta0_vals = sorted(df['beta0'].dropna().unique())

    list_eps1 = np.linspace(EPS1_MIN, EPS1_MAX, GRID_RESOLUTION_2D)
    list_eps2 = np.linspace(EPS2_MIN, EPS2_MAX, GRID_RESOLUTION_2D)
    cache_path = SCRIPT_DIR / f"stability_sweep_cache_{GRID_RESOLUTION_2D}.npz"

    if cache_path.exists():
        print(f"Loading cached 2D sweeps from {cache_path}...")
        cache = np.load(cache_path)
        Z_slice1 = cache['Z_slice1']
        Z_slice2 = cache['Z_slice2']
    else:
        print(f"Running 2D stability sweep for Slice 1 (beta = {SLICE_BETA_1})...")
        tasks1 = [(e1, e2, SLICE_BETA_1) for e1 in list_eps1 for e2 in list_eps2]
        with ProcessPoolExecutor(max_workers=4) as ex:
            Z_slice1 = np.array(list(ex.map(eval_stability_single, tasks1))).reshape(GRID_RESOLUTION_2D, GRID_RESOLUTION_2D)

        print(f"Running 2D stability sweep for Slice 2 (beta = {SLICE_BETA_2})...")
        tasks2 = [(e1, e2, SLICE_BETA_2) for e1 in list_eps1 for e2 in list_eps2]
        with ProcessPoolExecutor(max_workers=4) as ex:
            Z_slice2 = np.array(list(ex.map(eval_stability_single, tasks2))).reshape(GRID_RESOLUTION_2D, GRID_RESOLUTION_2D)

        np.savez(cache_path, Z_slice1=Z_slice1, Z_slice2=Z_slice2)
        print(f"Saved cache to {cache_path}")

    fig = plt.figure(figsize=(18, 6.5))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0])
    warm_cmap = mcolors.LinearSegmentedColormap.from_list("warm", [COLOR_3D_START, COLOR_3D_END])

    # --- Left: 3D Surface ---
    ax1 = fig.add_subplot(gs[0], projection='3d')
    N_u = 100
    eps2_lin = np.linspace(0.0, 3.0, N_u)
    eps2_mesh = np.full((len(beta0_vals), N_u), np.nan)
    eps1_mesh = np.full((len(beta0_vals), N_u), np.nan)

    for i, b0_val in enumerate(beta0_vals):
        slice_df = df[df['beta0'] == b0_val].dropna(subset=['eps1', 'eps2'])
        if len(slice_df) < 5:
            continue
        slice_df = slice_df.sort_values('eps2').drop_duplicates(subset=['eps2'])
        if len(slice_df) >= 2:
            f_eps1 = interp1d(slice_df['eps2'], slice_df['eps1'], bounds_error=False, fill_value=np.nan)
            eps2_mesh[i, :] = eps2_lin
            eps1_mesh[i, :] = f_eps1(eps2_lin)

    N_smooth = 100
    beta0_smooth = np.linspace(min(beta0_vals), max(beta0_vals), N_smooth)
    eps2_smooth = np.tile(eps2_lin[np.newaxis, :], (N_smooth, 1))
    eps1_smooth = np.full((N_smooth, N_u), np.nan)
    beta_smooth = np.tile(beta0_smooth[:, np.newaxis], (1, N_u))

    for j in range(N_u):
        valid = ~np.isnan(eps1_mesh[:, j])
        if np.sum(valid) >= 2:
            eps1_smooth[:, j] = np.interp(beta0_smooth, np.array(beta0_vals)[valid], eps1_mesh[valid, j], left=np.nan, right=np.nan)

    ax1.plot_surface(eps2_smooth, beta_smooth, eps1_smooth, cmap=warm_cmap, alpha=0.85,
                     shade=True, edgecolor='none', rcount=100, ccount=100)
    ax1.set_xlabel(r'$\tilde{\varepsilon}_2$', labelpad=12)
    ax1.set_ylabel(r'$\beta_0$', labelpad=12)
    ax1.set_zlabel(r'$\tilde{\varepsilon}_1$', labelpad=18)
    ax1.set_xlim(EPS2_MIN, EPS2_MAX)
    ax1.set_ylim(BETA0_MIN, BETA0_MAX)
    ax1.set_zlim(0.0, 1.0)
    ax1.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax1.set_yticks([500, 1000, 1500, 2000])
    ax1.set_zticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for label in ax1.yaxis.get_majorticklabels():
        label.set_rotation(-15)
        label.set_ha('right')
    ax1.view_init(elev=20, azim=-40)
    ax1.grid(True, alpha=0.2)

    # --- Middle & Right: 2D Slices ---
    cmap_2d = mcolors.ListedColormap([COLOR_UNSTABLE, COLOR_STABLE])
    X, Y = np.meshgrid(list_eps2, list_eps1)
    Z1s = gaussian_filter(Z_slice1, sigma=1.2)
    Z2s = gaussian_filter(Z_slice2, sigma=1.2)

    ax2 = fig.add_subplot(gs[1])
    ax2.contourf(X, Y, Z1s, levels=[-0.5, 0.5, 1.5], cmap=cmap_2d, alpha=0.95)
    ax2.contour(X, Y, Z1s, levels=[0.5], colors='white', linestyles='--', linewidths=2.2)
    ax2.plot([0, 3], [0, 3], color='white', linestyle=':', linewidth=2.0)
    ax2.set_xlabel(r'$\tilde{\varepsilon}_2$')
    ax2.set_ylabel(r'$\tilde{\varepsilon}_1$')
    ax2.grid(True, alpha=0.25)
    ax2.set_title(rf"$\beta_0 = {int(SLICE_BETA_1)}$")
    ax2.set_box_aspect(1.0)

    ax3 = fig.add_subplot(gs[2])
    ax3.contourf(X, Y, Z2s, levels=[-0.5, 0.5, 1.5], cmap=cmap_2d, alpha=0.95)
    ax3.contour(X, Y, Z2s, levels=[0.5], colors='white', linestyles='--', linewidths=2.2)
    ax3.plot([0, 3], [0, 3], color='white', linestyle=':', linewidth=2.0)
    ax3.set_xlabel(r'$\tilde{\varepsilon}_2$')
    ax3.set_ylabel(r'$\tilde{\varepsilon}_1$')
    ax3.grid(True, alpha=0.25)
    ax3.set_title(rf"$\beta_0 = {int(SLICE_BETA_2)}$")
    ax3.set_box_aspect(1.0)

    stable_patch   = mpatches.Patch(color=COLOR_STABLE,   label="Stable equilibrium")
    unstable_patch = mpatches.Patch(color=COLOR_UNSTABLE, label="Unstable equilibrium")
    hopf_line      = plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2.0, label="Hopf bifurcation boundary")
    sym_line       = plt.Line2D([0], [0], color='gray',  linestyle=':',  linewidth=2.0, label=r"$\tilde{\varepsilon}_1 = \tilde{\varepsilon}_2$")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.48, top=0.80, bottom=0.15)
    pos1 = ax1.get_position()
    ax1.set_position([pos1.x0 - 0.04, pos1.y0 - 0.02, pos1.width * 1.12, pos1.height * 1.12])
    fig.legend(handles=[stable_patch, unstable_patch, hopf_line, sym_line],
               loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=4, frameon=True, fontsize=14, handlelength=2.0)

    save_path = SCRIPT_DIR / "03_hopf_surface_two_eps.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
