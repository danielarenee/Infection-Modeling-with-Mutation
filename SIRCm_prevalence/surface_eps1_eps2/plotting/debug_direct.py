import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    slice_df_branch = df[df['eps1'] == e1_val].dropna(subset=['eps2', 'beta0', 'branch'])
    if len(slice_df_branch) < 5:
        continue
    v_idx_global = slice_df_branch['eps2'].idxmin()
    eps2_vertex = slice_df_branch.loc[v_idx_global, 'eps2']
    beta_vertex = slice_df_branch.loc[v_idx_global, 'beta0']
    
    forward_pts = slice_df_branch[slice_df_branch['branch'] == 'forward']
    backward_pts = slice_df_branch[slice_df_branch['branch'] == 'backward']
    
    if v_idx_global in forward_pts.index:
        cross_pts = forward_pts.reset_index(drop=True)
        away_pts = backward_pts
    else:
        cross_pts = backward_pts.reset_index(drop=True)
        away_pts = forward_pts
        
    v_idx_cross = cross_pts['eps2'].idxmin()
    part1 = cross_pts.iloc[:v_idx_cross+1]
    part2 = cross_pts.iloc[v_idx_cross:]
    
    leafA = pd.concat([away_pts, part1]).sort_values('eps2').drop_duplicates(subset=['eps2'])
    leafB = part2.sort_values('eps2').drop_duplicates(subset=['eps2'])
    
    # Standard splitting case (no monotonic branch splitting, just direct assignment)
    if len(leafA) < 5 or len(leafB) < 5 or (leafA['eps2'].max() - eps2_vertex < 0.2) or (leafB['eps2'].max() - eps2_vertex < 0.2):
        b_upper = slice_df_branch.sort_values('eps2').drop_duplicates(subset=['eps2'])
        b_lower = pd.DataFrame(columns=slice_df_branch.columns)
    else:
        if leafA['beta0'].mean() >= leafB['beta0'].mean():
            b_upper, b_lower = leafA, leafB
        else:
            b_upper, b_lower = leafB, leafA
            
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

# Plot directly without Step 2 smoothing
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(eps1_mesh, eps2_mesh, upper_grid, cmap='plasma', alpha=0.8)
ax.plot_surface(eps1_mesh, eps2_mesh, lower_grid, cmap='plasma', alpha=0.8)
plt.savefig(SCRIPT_DIR / "test_direct.png")
print("Saved direct test plot successfully.")
