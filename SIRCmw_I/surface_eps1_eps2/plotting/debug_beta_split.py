import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
csv_path = SCRIPT_DIR.parent / "beta_3d_surface" / "hopf_slices_eps1_indexed.csv"
df = pd.read_csv(csv_path)

df = df[(df['eps1'] <= 3.0) & (df['eps2'] <= 3.0) | df['eps1'].isna() | df['eps2'].isna()]
eps1_vals = sorted(df['eps1'].dropna().unique())

N_u = 100
u_lin = np.linspace(0.0, 1.0, N_u)

eps1_mesh = np.full((len(eps1_vals), N_u), np.nan)
eps2_mesh = np.full((len(eps1_vals), N_u), np.nan)
upper_grid = np.full((len(eps1_vals), N_u), np.nan)
lower_grid = np.full((len(eps1_vals), N_u), np.nan)

tol = 1e-4

for i, e1_val in enumerate(eps1_vals):
    slice_df = df[df['eps1'] == e1_val].dropna(subset=['eps2', 'beta0'])
    if len(slice_df) < 5:
        continue
    v_idx_global = slice_df['eps2'].idxmin()
    eps2_vertex = slice_df.loc[v_idx_global, 'eps2']
    beta_vertex = slice_df.loc[v_idx_global, 'beta0']
    
    # Check monotonic case
    # If the curve doesn't go below beta_vertex + 10.0 (or leaf range is small), it's monotonic
    forward_pts = slice_df[slice_df['branch'] == 'forward']
    backward_pts = slice_df[slice_df['branch'] == 'backward']
    
    # We can detect monotonic slices if the max eps2 of the short leaf is close to vertex
    if len(forward_pts) < 5 or len(backward_pts) < 5 or (forward_pts['eps2'].max() - eps2_vertex < 0.2) or (backward_pts['eps2'].max() - eps2_vertex < 0.2):
        # Monotonic case: split at beta = 300.0
        b_upper = slice_df[slice_df['beta0'] >= 300.0].sort_values('eps2').drop_duplicates(subset=['eps2'])
        b_lower = slice_df[slice_df['beta0'] < 300.0].sort_values('eps2').drop_duplicates(subset=['eps2'])
    else:
        # Folding case: split at beta_vertex!
        b_upper = slice_df[slice_df['beta0'] >= beta_vertex].sort_values('eps2').drop_duplicates(subset=['eps2'])
        b_lower = slice_df[slice_df['beta0'] <= beta_vertex].sort_values('eps2').drop_duplicates(subset=['eps2'])
        
    f_upper = interp1d(b_upper['eps2'], b_upper['beta0'], bounds_error=False, fill_value=np.nan) if len(b_upper) >= 2 else lambda x: np.nan
    f_lower = interp1d(b_lower['eps2'], b_lower['beta0'], bounds_error=False, fill_value=np.nan) if len(b_lower) >= 2 else lambda x: np.nan
    
    eps_max = 2.40
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

# Plot directly
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(eps1_mesh, eps2_mesh, upper_grid, cmap='plasma', alpha=0.8)
ax.plot_surface(eps1_mesh, eps2_mesh, lower_grid, cmap='plasma', alpha=0.8)
plt.savefig(SCRIPT_DIR / "test_beta_split.png")
print("Saved beta-split test plot successfully.")
