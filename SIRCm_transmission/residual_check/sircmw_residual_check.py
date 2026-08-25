"""
 checks how close the time averaged simulation equilibrium is to a fixed point 
 by plugging each averaged state into the RHS and plotting the residual components
 (should be zero)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from sircmw_utils import sircmw, MU as mu, ALPHA as alpha, DELTA as delta, GAMMA as gamma, SIGMA as sigma, BETA0 as beta0

y0_ts  = np.array([0.2, 0.001, 0.499, 0.3])
SI_0   = y0_ts[0] * y0_ts[1]  


def sircmw_rhs(y, tilde_eps):
    eps = tilde_eps / SI_0       
    p = {'beta0': beta0, 'sigma': sigma, 'eps': eps, 'mu': mu, 'alpha': alpha, 'delta': delta, 'gamma': gamma}
    return sircmw(0.0, y, p)

#  load sweep data
npz_path = Path(__file__).parent.parent / "sircmw_eps_sweep.npz"
d = np.load(npz_path)
eps_vals = d["eps_vals"]    
mean_eps = d["mean_eps"] 

#  compute residuals 
N = len(eps_vals)
residuals = np.empty((4, N))  # each component of f(y*)

for i, te in enumerate(eps_vals):
    y_star = mean_eps[:, i]  # [S*, I*, R*, C*]
    f = sircmw_rhs(y_star, te)
    residuals[:, i] = f

residual_norm = np.linalg.norm(residuals, axis=0)   # L2 norm per epsilon


# plot
labels = ["dS/dt", "dI/dt", "dR/dt", "dC/dt"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

fig, ax = plt.subplots(figsize=(9, 5))

for k, (lbl, col) in enumerate(zip(labels, colors)):
    ax.plot(eps_vals, residuals[k], color=col, linewidth=1.2, label=lbl)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel(r"Common $\tilde{\varepsilon}$", fontsize=12)
ax.set_ylabel(r"$f_k(y^*)$", fontsize=12)
ax.set_title(r"SIRCmw residual: long term time avg substituted into RHS",
             fontsize=12)
ax.legend(fontsize=10, loc="best")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / "sircmw_residual_check.png", dpi=150)
plt.show()
print("Plot saved.")
