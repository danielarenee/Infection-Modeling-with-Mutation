"""
This script generates a 3-panel figure for the SIRCmw model in the (eps1, eps2) parameter space:
- Left panel: 3D Hopf bifurcation surface with eps2 and beta0 on the floor, and eps1 as the vertical axis.
- Middle panel: 2D stability region slice in the (eps1, eps2) plane at beta0 = 200.
- Right panel: 2D stability region slice in the (eps1, eps2) plane at beta0 = 700.
For both 2D slices, it displays the diagonal line representing the symmetric case (eps1 = eps2).
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from concurrent.futures import ProcessPoolExecutor

# Add workspace directory to path to import sircmw_utils
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent.parent.parent))

from SIRCmw_I.sircmw_I_utils import (
    MU, ALPHA, DELTA, GAMMA, SIGMA, SI_0,
    sircmw_jacobian, get_C, get_endemic_roots
)

# Scaling factor: choose between I_0 (0.001) and S_0 * I_0 (0.000178 / SI_0)
scale_factor = 0.001

# ==============================================================================
# USER-EDITABLE CONFIGURATION
# ==============================================================================
# Parameter ranges for eps1 and eps2 (relative values)
EPS1_MIN, EPS1_MAX = 0.0, 3.0
EPS2_MIN, EPS2_MAX = 0.0, 3.0

# Fixed contact rate (beta0) values for the 2D slices
SLICE_BETA_1 = 200.0
SLICE_BETA_2 = 2000.0

# 2D Grid sweeps configuration
GRID_RESOLUTION_2D = 500

# Limits for the 3D surface
BETA0_MIN, BETA0_MAX = 100.0, 2000.0

# Color palette for 2D slices (stable vs unstable regions) matching the paper colors
COLOR_STABLE = '#e46c5c'        # Warm pink/red for stable endemic region
COLOR_UNSTABLE = '#fca636'      # Bright orangey yellow for unstable endemic region

# Color palette for 3D surface gradient (plasma/warm gradient)
COLOR_3D_START = '#d6556d'      # Start color (base height) of the 3D surface
COLOR_3D_END = '#fca636'        # End color (top height) of the 3D surface
# ==============================================================================

# Matplotlib publication settings
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
        return 1.0  # DFE is stable
    eps1 = rel_eps1 / scale_factor
    eps2 = rel_eps2 / scale_factor
    roots = get_endemic_roots(eps1, eps2, beta)
    if not roots:
        return 1.0  # DFE is stable
    
    has_stable = False
    for S, I, R, C in roots:
        J = sircmw_jacobian((S, I, R, C), eps1, eps2, p={'beta0': beta})
        max_real = np.max(np.real(np.linalg.eigvals(J)))
        if max_real < 0.0:
            has_stable = True
            break
    return 1.0 if has_stable else 0.0

def eval_stability_single(args):
    """Helper wrapper for parallel processing map"""
    rel_eps1, rel_eps2, beta = args
    return get_stability_for_sweep(rel_eps1, rel_eps2, beta)

def main():
    # 1. Load the 3D surface data from the CSV file
    csv_path = SCRIPT_DIR.parent / "beta_3d_surface" / "hopf_slices_eps1_indexed.csv"
    if not csv_path.exists():
        print(f"Error: Continuation data file not found at {csv_path}")
        sys.exit(1)
        
    print("Loading 3D Hopf continuation data...")
    df = pd.read_csv(csv_path)
    df = df[(df['eps1'] <= 3.0) & (df['eps2'] <= 3.0) | df['eps1'].isna() | df['eps2'].isna()]
    
    beta0_vals = sorted(df['beta0'].dropna().unique())
    
    # 2. Run 2D stability sweeps for beta_0 slices
    list_eps1 = np.linspace(EPS1_MIN, EPS1_MAX, GRID_RESOLUTION_2D)
    list_eps2 = np.linspace(EPS2_MIN, EPS2_MAX, GRID_RESOLUTION_2D)
    
    # Check for cached sweep results to prevent recalculation during styling edits
    cache_path = SCRIPT_DIR / f"stability_sweep_cache_{GRID_RESOLUTION_2D}.npz"
    if cache_path.exists():
        print(f"Loading cached 2D sweeps from {cache_path}...")
        cache = np.load(cache_path)
        Z_slice1 = cache['Z_slice1']
        Z_slice2 = cache['Z_slice2']
    else:
        print(f"\nRunning 2D stability sweep for Slice 1 (beta = {SLICE_BETA_1})...")
        tasks1 = [(e1, e2, SLICE_BETA_1) for e1 in list_eps1 for e2 in list_eps2]
        results1 = [eval_stability_single(t) for t in tasks1]
        Z_slice1 = np.array(results1).reshape(len(list_eps1), len(list_eps2))
        
        print(f"Running 2D stability sweep for Slice 2 (beta = {SLICE_BETA_2})...")
        tasks2 = [(e1, e2, SLICE_BETA_2) for e1 in list_eps1 for e2 in list_eps2]
        results2 = [eval_stability_single(t) for t in tasks2]
        Z_slice2 = np.array(results2).reshape(len(list_eps1), len(list_eps2))
        
        np.savez(cache_path, Z_slice1=Z_slice1, Z_slice2=Z_slice2)
        print(f"Saved computed sweeps to cache at {cache_path}")

    # 3-panel figure layout
    fig = plt.figure(figsize=(18, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0])
    
    # Custom warm gradient for the 3D surface
    warm_cmap = mcolors.LinearSegmentedColormap.from_list("warm_plasma", [COLOR_3D_START, COLOR_3D_END])
    
    # 1. Left Panel (3D surface: x=eps2, y=beta0, z=eps1)
    ax1 = fig.add_subplot(gs[0], projection='3d')
    
    N_u = 100
    eps2_lin = np.linspace(0.0, 3.0, N_u)
    
    eps2_mesh = np.full((len(beta0_vals), N_u), np.nan)
    eps1_mesh = np.full((len(beta0_vals), N_u), np.nan)
    beta_mesh = np.full((len(beta0_vals), N_u), np.nan)
    
    # Interpolate eps1 as a function of eps2 for each beta0 slice
    for i, b0_val in enumerate(beta0_vals):
        slice_df = df[df['beta0'] == b0_val].dropna(subset=['eps1', 'eps2'])
        if len(slice_df) < 5:
            continue
            
        slice_df = slice_df.sort_values('eps2').drop_duplicates(subset=['eps2'])
        
        if len(slice_df) >= 2:
            f_eps1 = interp1d(slice_df['eps2'], slice_df['eps1'], bounds_error=False, fill_value=np.nan)
            eps2_mesh[i, :] = eps2_lin
            eps1_mesh[i, :] = f_eps1(eps2_lin)
            beta_mesh[i, :] = b0_val
            
    # Smooth along the beta0 axis
    N_smooth = 100
    beta0_smooth = np.linspace(min(beta0_vals), max(beta0_vals), N_smooth)
    
    eps2_mesh_smooth = np.tile(eps2_lin[np.newaxis, :], (N_smooth, 1))
    eps1_mesh_smooth = np.full((N_smooth, N_u), np.nan)
    beta_mesh_smooth = np.tile(beta0_smooth[:, np.newaxis], (1, N_u))
    
    for j in range(N_u):
        valid_b = ~np.isnan(eps1_mesh[:, j])
        if np.sum(valid_b) >= 2:
            eps1_mesh_smooth[:, j] = np.interp(beta0_smooth, np.array(beta0_vals)[valid_b], eps1_mesh[valid_b, j], left=np.nan, right=np.nan)
            
    # Plot the smooth surface (x=eps2, y=beta0, z=eps1)
    # Using the single smooth sheet, colored by height eps1
    ax1.plot_surface(eps2_mesh_smooth, beta_mesh_smooth, eps1_mesh_smooth, cmap=warm_cmap, alpha=0.85,
                     shade=True, edgecolor='none', rcount=100, ccount=100)
    
    ax1.set_xlabel(r'$\epsilon_2$', labelpad=12)
    ax1.set_ylabel(r'$\beta_0$', labelpad=12)
    ax1.set_zlabel(r'$\epsilon_1$', labelpad=18)
    
    ax1.set_xlim(EPS2_MIN, EPS2_MAX)
    ax1.set_ylim(BETA0_MIN, BETA0_MAX)
    ax1.set_zlim(0.0, 1.0)
    
    ax1.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax1.set_yticks([500, 1000, 1500, 2000])
    ax1.set_zticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Rotate the y-tick labels (beta0 values) slightly to prevent overlap
    for label in ax1.yaxis.get_majorticklabels():
        label.set_rotation(-15)
        label.set_ha('right')
    
    ax1.view_init(elev=20, azim=-40) #CHANGE HERE 
    ax1.grid(True, alpha=0.2)
    # Remove the 3D Hopf Bifurcation Surface title as requested
    
    # Define custom ListedColormap for the 2D slices
    cmap_2d = mcolors.ListedColormap([COLOR_UNSTABLE, COLOR_STABLE])
    X, Y = np.meshgrid(list_eps2, list_eps1)
 
    # Apply Gaussian smoothing to stability grids to soften the boundary lines
    Z_slice1_smoothed = gaussian_filter(Z_slice1, sigma=1.2)
    Z_slice2_smoothed = gaussian_filter(Z_slice2, sigma=1.2)

    # 2. Middle Panel (Slice 1: beta = 200)
    ax2 = fig.add_subplot(gs[1])
    ax2.contourf(X, Y, Z_slice1_smoothed, levels=[-0.5, 0.5, 1.5], cmap=cmap_2d, alpha=0.95)
    ax2.contour(X, Y, Z_slice1_smoothed, levels=[0.5], colors='white', linestyles='--', linewidths=2.2)
    # Add dotted diagonal line e1=e2 (white)
    ax2.plot([0, 3], [0, 3], color='white', linestyle=':', linewidth=2.0)
    ax2.set_xlabel(r'$\tilde{\epsilon}_2$')
    ax2.set_ylabel(r'$\tilde{\epsilon}_1$')
    ax2.grid(True, alpha=0.25)
    ax2.set_title(rf"$\beta_0 = {int(SLICE_BETA_1)}$")
    ax2.set_box_aspect(1.0)
    
    # 3. Right Panel (Slice 2: beta = 700)
    ax3 = fig.add_subplot(gs[2])
    ax3.contourf(X, Y, Z_slice2_smoothed, levels=[-0.5, 0.5, 1.5], cmap=cmap_2d, alpha=0.95)
    ax3.contour(X, Y, Z_slice2_smoothed, levels=[0.5], colors='white', linestyles='--', linewidths=2.2)
    # Add dotted diagonal line e1=e2 (white)
    ax3.plot([0, 3], [0, 3], color='white', linestyle=':', linewidth=2.0)
    ax3.set_xlabel(r'$\tilde{\epsilon}_2$')
    ax3.set_ylabel(r'$\tilde{\epsilon}_1$')
    ax3.grid(True, alpha=0.25)
    ax3.set_title(rf"$\beta_0 = {int(SLICE_BETA_2)}$")
    ax3.set_box_aspect(1.0)
    
    import matplotlib.patches as mpatches
    stable_patch = mpatches.Patch(color=COLOR_STABLE, label="Stable equilibrium")
    unstable_patch = mpatches.Patch(color=COLOR_UNSTABLE, label="Unstable equilibrium")
    hopf_line = plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2.0, label="Hopf bifurcation boundary")
    symmetric_line = plt.Line2D([0], [0], color='gray', linestyle=':', linewidth=2.0, label=r"$\epsilon_1 = \epsilon_2$")
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.48, top=0.80, bottom=0.15)
    
    pos1 = ax1.get_position()
    ax1.set_position([pos1.x0 - 0.04, pos1.y0 - 0.02, pos1.width * 1.12, pos1.height * 1.12])
    
    fig.legend(handles=[stable_patch, unstable_patch, hopf_line, symmetric_line],
               loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=4, frameon=True, fontsize=14, handlelength=2.0)
    
    fig_save_path = SCRIPT_DIR.parent / "sircmw_eps1_eps2_3d_hopf_and_slices.png"
    plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSuccessfully generated and saved plot to: {fig_save_path}")

if __name__ == "__main__":
    main()
