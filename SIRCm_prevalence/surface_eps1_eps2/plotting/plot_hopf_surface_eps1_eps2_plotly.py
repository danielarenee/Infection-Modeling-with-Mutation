"""
Reads the output from sircmw_beta_3d_continuation_eps1_slices.jl (sliced along beta0),
constructs a single-sheet Hopf bifurcation surface where eps1 is the vertical axis
and eps2 and beta0 are on the floor, and plots it in 3D Plotly.
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

if not csv_path.exists():
    print(f"Error: csv data not found at {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)
# Limit both epsilon values to 3.0
df = df[(df['eps1'] <= 3.0) & (df['eps2'] <= 3.0) | df['eps1'].isna() | df['eps2'].isna()]

# Find unique beta0 slice values
beta0_vals = sorted(df['beta0'].dropna().unique())

N_u = 100
eps2_lin = np.linspace(0.0, 3.0, N_u)

eps1_mesh = np.full((len(beta0_vals), N_u), np.nan)
eps2_mesh = np.full((len(beta0_vals), N_u), np.nan)
beta_mesh = np.full((len(beta0_vals), N_u), np.nan)

# Step 1: Interpolate eps1 as a function of eps2 for each beta0 slice
for i, b0_val in enumerate(beta0_vals):
    slice_df = df[df['beta0'] == b0_val].dropna(subset=['eps1', 'eps2'])
    if len(slice_df) < 5:
        continue
        
    # Sort by eps2 and drop duplicate values to make it single-valued: eps1 = f(eps2)
    slice_df = slice_df.sort_values('eps2').drop_duplicates(subset=['eps2'])
    
    if len(slice_df) >= 2:
        f_eps1 = interp1d(slice_df['eps2'], slice_df['eps1'], bounds_error=False, fill_value=np.nan)
        eps2_mesh[i, :] = eps2_lin
        eps1_mesh[i, :] = f_eps1(eps2_lin)
        beta_mesh[i, :] = b0_val

# Step 2: Smoothly interpolate along the beta0 axis
N_smooth = 100
beta0_smooth = np.linspace(min(beta0_vals), max(beta0_vals), N_smooth)

eps2_mesh_smooth = np.tile(eps2_lin[np.newaxis, :], (N_smooth, 1))
eps1_mesh_smooth = np.full((N_smooth, N_u), np.nan)
beta_mesh_smooth = np.tile(beta0_smooth[:, np.newaxis], (1, N_u))

for j in range(N_u):
    valid_b = ~np.isnan(eps1_mesh[:, j])
    if np.sum(valid_b) >= 2:
        eps1_mesh_smooth[:, j] = np.interp(beta0_smooth, np.array(beta0_vals)[valid_b], eps1_mesh[valid_b, j], left=np.nan, right=np.nan)

# Step 3: Build the Plotly figure
fig = go.Figure()

# Plot the single smooth Hopf surface sheet (x=eps2, y=beta0, z=eps1)
fig.add_trace(go.Surface(
    x=eps2_mesh_smooth,
    y=beta_mesh_smooth,
    z=eps1_mesh_smooth,
    colorscale="Blues",
    opacity=0.75,
    showscale=True,
    name="Hopf Surface",
    colorbar=dict(title="Tilde eps1", thickness=15, len=0.6)
))

# Plot the raw continuation curves (rotated: x=eps2, y=beta0, z=eps1)
b_min, b_max = min(beta0_vals), max(beta0_vals)
b_range = b_max - b_min if b_max > b_min else 1.0

for k in range(0, len(beta0_vals), 2): # Plot every second curve to keep rendering lightweight
    b0_val = beta0_vals[k]
    slice_df = df[df['beta0'] == b0_val]
    x_coords = slice_df['eps2'].values
    y_coords = slice_df['beta0'].values
    z_coords = slice_df['eps1'].values
    
    val_norm = (b0_val - b_min) / b_range
    color_hex = plotly.colors.sample_colorscale('turbo', [val_norm])[0]
    
    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='lines',
        line=dict(color=color_hex, width=4),
        name=f'beta0 = {b0_val:.1f}',
        showlegend=False
    ))

fig.update_layout(
    title=dict(
        text='SIRCmw Asymmetric Hopf Bifurcation Surface (eps1 vs eps2 vs beta0)',
        x=0.5,
        y=0.95
    ),
    scene=dict(
        xaxis=dict(title='Tilde epsilon 2 (Floor)', range=[0, 3.0]),
        yaxis=dict(title='Beta0 (Floor)', range=[100, 2000]),
        zaxis=dict(title='Tilde epsilon 1 (Vertical)', range=[0, 1.0]),
        camera=dict(
            eye=dict(x=1.8, y=-1.8, z=1.4)
        )
    ),
    template='plotly_white',
    margin=dict(l=0, r=0, b=0, t=80)
)

out_file = SCRIPT_DIR.parent / "sircmw_eps1_eps2_interactive.html"
fig.write_html(str(out_file))
print(f"Interactive 3D visualization saved to: {out_file}")
