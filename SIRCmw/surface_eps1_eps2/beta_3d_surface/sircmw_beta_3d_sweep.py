"""
Reads the output from sircmw_beta_3d_continuation.jl, groups the coordinates by beta0 and generates a 3D plot
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.colors

SCRIPT_DIR = Path(__file__).parent
csv_path = SCRIPT_DIR / "hopf_slices_eps1_eps2_beta.csv"

df = pd.read_csv(csv_path)

# find unique beta values
beta_vals = sorted(df['beta0'].unique())

fig = go.Figure()

b_min, b_max = min(beta_vals), max(beta_vals)
b_range = b_max - b_min if b_max > b_min else 1.0

for b_val in beta_vals:
    # filter points for this beta slice
    slice_df = df[df['beta0'] == b_val]
    
    # extract coordinates
    x_coords = slice_df['eps2'].values
    y_coords = slice_df['eps1'].values
    z_coords = slice_df['beta0'].values
    
    # color-code 
    val_norm = (b_val - b_min) / b_range
    color_hex = plotly.colors.sample_colorscale('turbo', [val_norm])[0]
    
    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='lines',
        line=dict(color=color_hex, width=6),
        name=f'Beta = {b_val:.0f}'
    ))

fig.update_layout(
    title=dict(
        text='SIRCmw Hopf Bifurcation Boundary Curves',
        x=0.5,
        y=0.95
    ),
    scene=dict(
        xaxis=dict(title='Tilde epsilon 2', range=[0, 3]),
        yaxis=dict(title='Tilde epsilon 1', range=[0, 3]),
        zaxis=dict(title='Beta0', range=[100, 2000]),
        camera=dict(
            eye=dict(x=1.8, y=-1.8, z=1.4)
        )
    ),
    template='plotly_white',  # light mode
    margin=dict(l=0, r=0, b=0, t=80)
)

out_file = SCRIPT_DIR / "beta_3d_prevalence.html"
fig.write_html(str(out_file))
print(f"Interactive 3D visualization saved to: {out_file}")
