"""
Comparison plot between prevalence from simulation and from polynomial roots
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from sircmw_utils import (
    MU as mu,
    ALPHA as alpha,
    GAMMA as gamma,
    DELTA as delta,
    SIGMA as sigma,
    BETA0 as beta,
    SI_0,
    step,
    solve_rk4,
    _rk4_mean_endemic,
    get_algebraic_equilibria
)

#%%

# USING SIMULATION
# this is the mean behavior of all compartments of the SIRCmw after 200 years of simulation
# Results of simulation are saved to sircmw_eps_sweep.npz

_sim = np.load(Path(__file__).parent.parent / "sircmw_eps_sweep.npz")
#_sim     = np.load("sircmw_eps_sweep.npz")
nsim     = int(_sim['nsim'])
tsim     = float(_sim['tsim'])
avgyrs   = float(_sim['avgyrs'])
eps_vals = _sim['eps_vals']
mean_eps = _sim['mean_eps']
amp_eps  = _sim['amp_eps']


fig1, ax1 = plt.subplots(figsize=(9, 5))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
labels = ['S', 'I', 'R', 'C']
for k, (col, lbl) in enumerate(zip(colors, labels)):
    ax1.plot(eps_vals, mean_eps[k], color=col, linewidth=1.5, label=lbl)
ax1.set_xlabel(r'Common $\tilde{\varepsilon}$', fontsize=12)
ax1.set_ylabel('Mean prevalence (last 10 yrs)', fontsize=12)
ax1.set_title(f'SIRCmw mean prevalence vs tilde eps (β₀=600, {tsim:.0f} yrs)', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "sircmw_mean_prevalence_sim.png", dpi=150)
plt.show()


# USING POLYNOMIAL ROOTS 
# sweep tilde_eps and collect equilibrium points
tilde_eps_vals = np.linspace(0, 2, 1000)
rows = [] # each entry: (tilde_eps, S*, I*, R*, C*)

for te in tilde_eps_vals:
    eqs = get_algebraic_equilibria(te)
    for eq in eqs:
        rows.append((te, *eq))

data = np.array(rows) 

# plot
fig, ax = plt.subplots(figsize=(9, 5))

for col_idx, (color, label) in enumerate(zip(
        ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'],
        ['S*', 'I*', 'R*', 'C*'])):
    ax.scatter(data[:, 0], data[:, 1 + col_idx], s=4, color=color, label=label)

ax.set_xlabel(r'$\tilde{\varepsilon}$', fontsize=12)
ax.set_ylabel('Equilibrium prevalence', fontsize=12)
ax.set_title('SIRCmw equilibrium (S*, I*, R*, C*) from polynomial roots', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "sircmw_equilibrium_poly.png", dpi=150)
plt.show()

#%%

# combined: polynomial equilibria (transparent scatter) beneath simulation means (solid lines)
fig2, ax2 = plt.subplots(figsize=(9, 5))

for col_idx, (color, label) in enumerate(zip(
        ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'],
        ['S', 'I', 'R', 'C'])):
    ax2.scatter(data[:, 0], data[:, 1 + col_idx],
                s=4, color=color, alpha=0.08, label=f'{label}* (pol)')
    ax2.plot(eps_vals, mean_eps[col_idx], color=color, linewidth=1.5,
             label=f'{label} (sim)')

ax2.set_xlabel(r'$\tilde{\varepsilon}$', fontsize=12)
ax2.set_ylabel('Prevalence', fontsize=12)
ax2.set_title('SIRCmw: polynomial equilibria vs simulation mean prevalence', fontsize=13)
ax2.legend(fontsize=9, ncol=2)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "sircmw_prevalence_comparison.png", dpi=150)
plt.show()
