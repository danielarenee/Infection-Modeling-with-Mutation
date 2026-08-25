"""
Reads the output from sircmw_beta_3d_continuation_eps1_slices.jl, constructs
a parameterized split-sheet (upper and lower leaves) surface using bilinear interpolation
with hybrid monotonic branch support, and plots both the surface sheets and the stack of curves in 3D Plotly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.colors
from scipy.interpolate import interp1d

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
csv_path = SCRIPT_DIR.parent / "beta_3d_surface" / "hopf_slices_eps1_indexed.csv"

df = pd.read_csv(csv_path)
# Limit both epsilon values to 3.0, preserving NaNs for line breaks
df = df[(df['eps1'] <= 3.0) & (df['eps2'] <= 3.0) | df['eps1'].isna() | df['eps2'].isna()]

# find unique eps1 values
eps1_vals = sorted(df['eps1'].dropna().unique())

N_u = 100
u_lin = np.linspace(0.0, 1.0, N_u)

eps1_mesh = np.full((len(eps1_vals), N_u), np.nan)
eps2_mesh = np.full((len(eps1_vals), N_u), np.nan)
upper_grid = np.full((len(eps1_vals), N_u), np.nan)
lower_grid = np.full((len(eps1_vals), N_u), np.nan)

# Step 1: Interpolate the dense eps1 slices over regular u coordinates
for i, e1_val in enumerate(eps1_vals):
    slice_df = df[df['eps1'] == e1_val].dropna(subset=['eps2', 'beta0', 'branch'])
    if len(slice_df) < 5:
        continue
        
    # Find global minimum eps2 (the nose vertex)
    v_idx_global = slice_df['eps2'].idxmin()
    eps2_vertex = slice_df.loc[v_idx_global, 'eps2']
    beta_vertex = slice_df.loc[v_idx_global, 'beta0']
    
    # Split using the branch identifiers forward/backward
    forward_pts = slice_df[slice_df['branch'] == 'forward']
    backward_pts = slice_df[slice_df['branch'] == 'backward']
    
    # Split sequentially along the path at the global minimum eps2 (the nose vertex)
    if len(forward_pts) < 5 or len(backward_pts) < 5 or (forward_pts['eps2'].max() - eps2_vertex < 0.2) or (backward_pts['eps2'].max() - eps2_vertex < 0.2):
        # Monotonic case: split at beta0 = 300.0
        b_upper = slice_df[slice_df['beta0'] >= 300.0].sort_values('eps2').drop_duplicates(subset=['eps2'])
        b_lower = slice_df[slice_df['beta0'] < 300.0].sort_values('eps2').drop_duplicates(subset=['eps2'])
    else:
        # Folding case
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
    
    eps_max = 2.40
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

# Step 2: Smoothly interpolate coordinates and height-fields along the eps1 axis
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

# Step 3: Build the Plotly figure
fig = go.Figure()

# Plot the smooth upper and lower surface sheets in a transparent Blues colormap
fig.add_trace(go.Surface(
    x=eps2_mesh_smooth,
    y=eps1_mesh_smooth,
    z=upper_grid_smooth,
    colorscale="Blues",
    opacity=0.6,
    showscale=False,
    name="Upper Surface",
    legendgroup="surface"
))

fig.add_trace(go.Surface(
    x=eps2_mesh_smooth,
    y=eps1_mesh_smooth,
    z=lower_grid_smooth,
    colorscale="Blues",
    opacity=0.6,
    showscale=False,
    name="Lower Surface",
    legendgroup="surface"
))

# Plot all continuation curves
b_min, b_max = min(eps1_vals), max(eps1_vals)
b_range = b_max - b_min if b_max > b_min else 1.0

for k in range(len(eps1_vals)):
    e1_val = eps1_vals[k]
    slice_df = df[df['eps1'] == e1_val]
    x_coords = slice_df['eps2'].values
    y_coords = slice_df['eps1'].values
    z_coords = slice_df['beta0'].values
    
    val_norm = (e1_val - b_min) / b_range
    color_hex = plotly.colors.sample_colorscale('turbo', [val_norm])[0]
    
    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='lines',
        line=dict(color=color_hex, width=6),
        name=f'eps1 = {e1_val:.2f}',
        legendgroup="curves"
    ))

fig.update_layout(
    title=dict(
        text='SIRCmw Hopf Bifurcation Surface & Curves (eps1 Slices)',
        x=0.5,
        y=0.95
    ),
    scene=dict(
        xaxis=dict(title='Tilde epsilon 2', range=[0, 3.0]),
        yaxis=dict(title='Tilde epsilon 1', range=[0, 3.0]),
        zaxis=dict(title='Beta0', range=[100, 2000]),
        camera=dict(
            eye=dict(x=1.8, y=-1.8, z=1.4)
        )
    ),
    template='plotly_white',  # light mode
    margin=dict(l=0, r=0, b=0, t=80),
    showlegend=True
)

out_file = SCRIPT_DIR.parent / "sircmw_eps1_eps2_interactive.html"
fig.write_html(str(out_file))
print(f"Interactive 3D visualization saved to: {out_file}")
