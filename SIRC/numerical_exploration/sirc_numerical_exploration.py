"""
sweeps a 2d parameter grid across all combinations of parameters of SIRC
to analyze the endemic equilibrium stability and plots a figure with heatmaps
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations

sys.path.append(str(Path(__file__).parent.parent))

from sirc_utils import stability_analysis

def sweep(param1_name, param1_vals, param2_name, param2_vals, fixed_params):
    """
    2D parameter sweep: for each (p1, p2) pair in the grid, calls
    stability_analysis with all other parameters held at their reference values.
    """
    n1, n2 = len(param1_vals), len(param2_vals)
    max_real = np.full((n2, n1), np.nan)

    for i, p2 in enumerate(param2_vals):
        for j, p1 in enumerate(param1_vals):
            params = dict(fixed_params)
            params[param1_name] = p1
            params[param2_name] = p2

            mr, eigs = stability_analysis(**params)
            max_real[i, j] = mr

    return max_real

N = 100 # resolution of the grid

# fixed reference values (for when we dont sweep)
ref = dict(beta=600, mu=0.02, alpha=365/3, delta=1/1.61, gamma=0.35, sigma=0.07874)

# parameter names, labels, and ranges from table 1 in the Casagrandi paper
param_info = {
    'beta':  (np.linspace(2*(ref['mu']+ref['alpha']), 10*(ref['mu']+ref['alpha']), N), r'$\beta$'),
    'sigma': (np.linspace(0.05, 0.2, N), r'$\sigma$'),
    'delta': (np.linspace(0.5, 1.0, N), r'$\delta$'),
    'gamma': (np.linspace(1/5, 1/2, N), r'$\gamma$'),
    'mu': (np.linspace(1/80, 1/40, N), r'$\mu$'),
    'alpha': (np.linspace(365/7, 365/2, N), r'$\alpha$'),
}

# get all 15 pairs
sweep_configs = []
for p1_name, p2_name in combinations(param_info.keys(), 2):
    p1_vals, p1_label = param_info[p1_name]
    p2_vals, p2_label = param_info[p2_name]
    sweep_configs.append({
        'param1_name': p1_name,
        'param1_vals': p1_vals,
        'param1_label': p1_label,
        'param2_name': p2_name,
        'param2_vals': p2_vals,
        'param2_label': p2_label,
        'title': f'{p1_label} vs {p2_label}'
    })

fig, axes = plt.subplots(3, 5, figsize=(30, 16))
axes_flat = axes.flatten()
all_max_real = []

for k, cfg in enumerate(sweep_configs):
    fixed = {key: val for key, val in ref.items()
             if key != cfg['param1_name'] and key != cfg['param2_name']}

    max_real = sweep(
        cfg['param1_name'], cfg['param1_vals'],
        cfg['param2_name'], cfg['param2_vals'],
        fixed
    )
    all_max_real.append(np.nanmax(max_real))

    ax = axes_flat[k]
    vmax = np.nanmax(np.abs(max_real))
    vmin = -vmax

    im = ax.pcolormesh(cfg['param1_vals'], cfg['param2_vals'], max_real,
                        cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')

    # gray out region where no endemic equilibrium exists (R0 <= 1)
    no_endemic = np.where(np.isnan(max_real), 1.0, np.nan)
    ax.pcolormesh(cfg['param1_vals'], cfg['param2_vals'], no_endemic,
                  cmap='Greys', vmin=0, vmax=1, shading='auto', alpha=0.5)

    ax.set_xlabel(cfg['param1_label'], fontsize=10)
    ax.set_ylabel(cfg['param2_label'], fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'sirc_numerical_exploration.png')

print("\nSummary across all sweeps:")
for k, cfg in enumerate(sweep_configs):
    print(f"\n{cfg['title']}:")
    print(f"  Max real part in the grid: {all_max_real[k]:.6f}")
