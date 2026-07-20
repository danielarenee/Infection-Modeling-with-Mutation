"""
Verifies exponential decay between disease outbreaks
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from sircm_utils import (
    sirc_modified,
    solve,
    MU,
    SIGMA,
    Y0,
    DELTA_PRIME,
    GAMMA_PRIME
)

h = 1/365

mod_params = {
    'mu': MU, 'alpha': 365.0/3.0, 'sigma': SIGMA,
    'beta0': 800, 'eps': 0.0,
    'delta_prime': DELTA_PRIME, 'gamma_prime': GAMMA_PRIME,
}

t_mod, y_mod = solve(sirc_modified, Y0, (0, 200), h, mod_params)

# extract quiet period (between two spikes)
t_quiet_start = 16
t_quiet_end   = 20
mask_quiet = (t_mod > t_quiet_start) & (t_mod < t_quiet_end)

R_quiet = y_mod[mask_quiet, 2]
t_quiet = t_mod[mask_quiet]

# derivative
dR = np.diff(R_quiet) / h
t_dR = t_quiet[:-1]
R_for_comparison = R_quiet[:-1]

# plot actual derivative vs theoretical decay rate -mu * R
fig1, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].plot(t_quiet, R_quiet, 'g-', linewidth=1.5)
axes[0].set_ylabel('R(t)', fontsize=12)
axes[0].set_title('R during one quiet period', fontsize=13)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_dR, dR, 'r-', linewidth=1.5, label='numerical dR/dt')
axes[1].plot(t_dR, -MU * R_for_comparison, 'b--', linewidth=1.5, label='-μR')
axes[1].set_xlabel('Time (years)', fontsize=12)
axes[1].set_ylabel('dR/dt', fontsize=12)
axes[1].set_title('Derivative of R: actual vs -μR', fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / "R_decay_derivative.png", dpi=150)
plt.show()
plt.close()
