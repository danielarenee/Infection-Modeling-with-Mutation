#!/usr/bin/env python3
"""
Hopf bifurcation surface and 2D stability slices (Transmission-driven variant)

Generates a 3-panel figure for the transmission-driven SIRCm model in (eps1, eps2) space:
- Left panel: 3D Hopf bifurcation surface in (eps1, eps2, beta0) space from continuation data.
- Middle panel: 2D stability slice in the (eps1, eps2) plane at SLICE_BETA_1.
- Right panel: 2D stability slice in the (eps1, eps2) plane at SLICE_BETA_2.

OUTPUT: Figure 04
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

from sircmw_utils import (
    MU, ALPHA, DELTA, GAMMA, SIGMA, SI_0,
    sircmw_jacobian, get_endemic_roots
)

# =============================================================================
# CONFIGURATION (Edit contact rates for the 2D stability slices)
# =============================================================================
SLICE_BETA_1 = 600.0
SLICE_BETA_2 = 900.0

EPS1_MIN, EPS1_MAX = 0.0, 3.0
EPS2_MIN, EPS2_MAX = 0.0, 3.0
BETA0_MIN, BETA0_MAX = 100.0, 2000.0
GRID_RESOLUTION_2D = 200

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
    eps1 = rel_eps1 / SI_0
    eps2 = rel_eps2 / SI_0
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
    scale_factor = 0.0002045191 / 0.000178
    df['eps1'] = df['eps1'] * scale_factor
    df['eps2'] = df['eps2'] * scale_factor
    beta_vals = sorted(df['beta0'].dropna().unique())

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
        with ProcessPoolExecutor(max_workers=3) as executor:
            Z_slice1 = np.array(list(executor.map(eval_stability_single, tasks1))).reshape(len(list_eps1), len(list_eps2))

        print(f"Running 2D stability sweep for Slice 2 (beta = {SLICE_BETA_2})...")
        tasks2 = [(e1, e2, SLICE_BETA_2) for e1 in list_eps1 for e2 in list_eps2]
        with ProcessPoolExecutor(max_workers=3) as executor:
            Z_slice2 = np.array(list(executor.map(eval_stability_single, tasks2))).reshape(len(list_eps1), len(list_eps2))

        np.savez(cache_path, Z_slice1=Z_slice1, Z_slice2=Z_slice2)
        print(f"Saved cache to {cache_path}")

    fig = plt.figure(figsize=(18, 6.5))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1.5, 0.9, 0.9])
    warm_cmap = mcolors.LinearSegmentedColormap.from_list("warm_plasma", [COLOR_3D_START, COLOR_3D_END])

    # --- Left: 3D Surface ---
    ax1 = fig.add_subplot(gs[0], projection='3d')
    df = df[(df['eps1'] <= 3.15) & (df['eps2'] <= 3.15) | df['eps1'].isna() | df['eps2'].isna()]
    eps1_vals = sorted(df['eps1'].dropna().unique())

    N_u = 100
    u_lin = np.linspace(0.0, 1.0, N_u)
    eps1_mesh = np.full((len(eps1_vals), N_u), np.nan)
    eps2_mesh = np.full((len(eps1_vals), N_u), np.nan)
    upper_grid = np.full((len(eps1_vals), N_u), np.nan)
    lower_grid = np.full((len(eps1_vals), N_u), np.nan)

    for i, e1_val in enumerate(eps1_vals):
        slice_df = df[df['eps1'] == e1_val].dropna(subset=['eps2', 'beta0', 'branch'])
        if len(slice_df) < 5:
            continue

        v_idx_global = slice_df['eps2'].idxmin()
        eps2_vertex = slice_df.loc[v_idx_global, 'eps2']
        beta_vertex = slice_df.loc[v_idx_global, 'beta0']

        forward_pts = slice_df[slice_df['branch'] == 'forward']
        backward_pts = slice_df[slice_df['branch'] == 'backward']

        if len(forward_pts) < 5 or len(backward_pts) < 5 or (forward_pts['eps2'].max() - eps2_vertex < 0.2) or (backward_pts['eps2'].max() - eps2_vertex < 0.2):
            b_upper = slice_df[slice_df['beta0'] >= 300.0].sort_values('eps2').drop_duplicates(subset=['eps2'])
            b_lower = slice_df[slice_df['beta0'] < 300.0].sort_values('eps2').drop_duplicates(subset=['eps2'])
        else:
            slice_df_clean = slice_df.reset_index(drop=True)
            v_pos = slice_df_clean['eps2'].idxmin()
            part1 = slice_df_clean.iloc[:v_pos+1]
            part2 = slice_df_clean.iloc[v_pos:]
            if part1['beta0'].mean() >= part2['beta0'].mean():
                b_upper, b_lower = part1, part2
            else:
                b_upper, b_lower = part2, part1
            b_upper = b_upper.sort_values('eps2').drop_duplicates(subset=['eps2'])
            b_lower = b_lower.sort_values('eps2').drop_duplicates(subset=['eps2'])

        f_upper = interp1d(b_upper['eps2'], b_upper['beta0'], bounds_error=False, fill_value=np.nan) if len(b_upper) >= 2 else lambda x: np.nan
        f_lower = interp1d(b_lower['eps2'], b_lower['beta0'], bounds_error=False, fill_value=np.nan) if len(b_lower) >= 2 else lambda x: np.nan

        eps_max = 2.40 * scale_factor
        tol = 1e-4
        if eps_max > eps2_vertex:
            for j, u in enumerate(u_lin):
                te2 = eps2_vertex + u * (eps_max - eps2_vertex)
                eps2_mesh[i, j] = te2
                eps1_mesh[i, j] = e1_val
                if len(b_lower) >= 2:
                    min_lo, max_lo = b_lower['eps2'].min(), b_lower['eps2'].max()
                    if min_lo - tol <= te2 <= max_lo + tol:
                        lower_grid[i, j] = f_lower(np.clip(te2, min_lo, max_lo))
                if len(b_upper) >= 2:
                    min_up, max_up = b_upper['eps2'].min(), b_upper['eps2'].max()
                    if min_up - tol <= te2 <= max_up + tol:
                        upper_grid[i, j] = f_upper(np.clip(te2, min_up, max_up))

    N_smooth = 100
    eps1_smooth = np.linspace(0.0, 3.0, N_smooth)
    eps1_mesh_smooth = np.tile(eps1_smooth[:, np.newaxis], (1, N_u))
    eps2_mesh_smooth = np.full((N_smooth, N_u), np.nan)
    upper_grid_smooth = np.full((N_smooth, N_u), np.nan)
    lower_grid_smooth = np.full((N_smooth, N_u), np.nan)

    for j in range(N_u):
        valid_e2 = ~np.isnan(eps2_mesh[:, j])
        if np.sum(valid_e2) >= 2:
            eps2_mesh_smooth[:, j] = np.interp(eps1_smooth, np.array(eps1_vals)[valid_e2], eps2_mesh[valid_e2, j], left=np.nan, right=np.nan)
        valid_up = ~np.isnan(upper_grid[:, j])
        if np.sum(valid_up) >= 2:
            upper_grid_smooth[:, j] = np.interp(eps1_smooth, np.array(eps1_vals)[valid_up], upper_grid[valid_up, j], left=np.nan, right=np.nan)
        valid_lo = ~np.isnan(lower_grid[:, j])
        if np.sum(valid_lo) >= 2:
            lower_grid_smooth[:, j] = np.interp(eps1_smooth, np.array(eps1_vals)[valid_lo], lower_grid[valid_lo, j], left=np.nan, right=np.nan)

    ax1.plot_surface(eps1_mesh_smooth, eps2_mesh_smooth, upper_grid_smooth, cmap=warm_cmap, alpha=0.85,
                     shade=True, edgecolor='none', rcount=100, ccount=100)
    ax1.plot_surface(eps1_mesh_smooth, eps2_mesh_smooth, lower_grid_smooth, cmap=warm_cmap, alpha=0.85,
                     shade=True, edgecolor='none', rcount=100, ccount=100)

    ax1.set_xlabel(r'$\tilde{\varepsilon}_1$', labelpad=12)
    ax1.set_ylabel(r'$\tilde{\varepsilon}_2$', labelpad=12)
    ax1.set_zlabel(r'$\beta_0$', labelpad=18)
    ax1.set_xlim(EPS1_MIN, EPS1_MAX)
    ax1.set_ylim(EPS2_MIN, EPS2_MAX)
    ax1.set_zlim(BETA0_MIN, BETA0_MAX)
    ax1.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax1.set_yticks([0.0, 1.0, 2.0, 3.0])
    ax1.zaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.view_init(elev=24, azim=-35)
    ax1.grid(True, alpha=0.2)

    # --- Middle & Right: 2D Slices ---
    cmap_2d = mcolors.ListedColormap([COLOR_UNSTABLE, COLOR_STABLE])
    X, Y = np.meshgrid(list_eps2, list_eps1)
    Z_slice1_smoothed = gaussian_filter(Z_slice1, sigma=1.2)
    Z_slice2_smoothed = gaussian_filter(Z_slice2, sigma=1.2)

    ax2 = fig.add_subplot(gs[1])
    ax2.contourf(X, Y, Z_slice1_smoothed, levels=[-0.5, 0.5, 1.5], cmap=cmap_2d, alpha=0.95)
    ax2.contour(X, Y, Z_slice1_smoothed, levels=[0.5], colors='white', linestyles='--', linewidths=2.2)
    ax2.plot([0, 3], [0, 3], color='white', linestyle=':', linewidth=2.0)
    ax2.set_xlabel(r'$\tilde{\epsilon}_2$')
    ax2.set_ylabel(r'$\tilde{\epsilon}_1$')
    ax2.grid(True, alpha=0.25)
    ax2.set_title(rf"$\beta_0 = {int(SLICE_BETA_1)}$")
    ax2.set_box_aspect(1.0)

    ax3 = fig.add_subplot(gs[2])
    ax3.contourf(X, Y, Z_slice2_smoothed, levels=[-0.5, 0.5, 1.5], cmap=cmap_2d, alpha=0.95)
    ax3.contour(X, Y, Z_slice2_smoothed, levels=[0.5], colors='white', linestyles='--', linewidths=2.2)
    ax3.plot([0, 3], [0, 3], color='white', linestyle=':', linewidth=2.0)
    ax3.set_xlabel(r'$\tilde{\epsilon}_2$')
    ax3.set_ylabel(r'$\tilde{\epsilon}_1$')
    ax3.grid(True, alpha=0.25)
    ax3.set_title(rf"$\beta_0 = {int(SLICE_BETA_2)}$")
    ax3.set_box_aspect(1.0)

    stable_patch = mpatches.Patch(color=COLOR_STABLE, label="Stable equilibrium")
    unstable_patch = mpatches.Patch(color=COLOR_UNSTABLE, label="Unstable equilibrium")
    hopf_line = plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2.0, label="Hopf bifurcation boundary")
    symmetric_line = plt.Line2D([0], [0], color='gray', linestyle=':', linewidth=2.0, label=r"$\tilde{\varepsilon}_1 = \tilde{\varepsilon}_2$")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.40, top=0.82, bottom=0.15)
    pos1 = ax1.get_position()
    ax1.set_position([pos1.x0 + 0.01, pos1.y0 - 0.02, pos1.width * 1.12, pos1.height * 1.12])
    fig.legend(handles=[stable_patch, unstable_patch, hopf_line, symmetric_line],
               loc='upper right', bbox_to_anchor=(0.95, 0.99), ncol=4, frameon=True, fontsize=16, handlelength=2.0)

    fig_save_path = SCRIPT_DIR / "04_hopf_surface_two_eps.png"
    plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_save_path}")


if __name__ == "__main__":
    main()
