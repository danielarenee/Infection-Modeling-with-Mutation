"""
Comparison of BDF vs RK4 solver to investigate why solve_ivp was failing
(drops many orders of magnitude)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp

sys.path.append(str(Path(__file__).parent.parent))

from sircm_utils import (
    sirc_modified,
    solve,
    MU,
    ALPHA,
    SIGMA,
    Y0,
    DELTA_PRIME,
    GAMMA_PRIME
)

h_rk4 = 1/730

pm_bdf = {
    'mu': MU, 'alpha': ALPHA, 'sigma': SIGMA,
    'beta0': 3100, 'eps': 0.0,
    'delta_prime': DELTA_PRIME, 'gamma_prime': GAMMA_PRIME
}

t_mod, y_mod = solve(sirc_modified, Y0, (0, 10), h_rk4, pm_bdf)

sol_bdf = solve_ivp(lambda t, y: sirc_modified(t, y, pm_bdf),
                    (0, 10), Y0, method='BDF',
                    rtol=1e-8, atol=1e-10, max_step=0.01)

print(f"BDF status: {sol_bdf.status}")
print(f"BDF reached t = {sol_bdf.t[-1]:.4f}")

# frozen phase check
mask = (t_mod > 1) & (t_mod < 6)
I_frozen = y_mod[mask, 1]
print(f"\nRK4 (h=1/730) frozen phase (t=1 to 6):")
print(f"  min(I) = {I_frozen.min():.2e}")
print(f"  max(I) = {I_frozen.max():.2e}")
print(f"  mean(I) = {I_frozen.mean():.2e}")

# plot linear scale comparison
fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(t_mod, y_mod[:, 1], 'r-', linewidth=1, label='RK4 (h=1/730)')
if sol_bdf.status == 0:
    ax1.plot(sol_bdf.t, sol_bdf.y[1], 'g--', linewidth=1.5, label='BDF')
else:
    ax1.plot(sol_bdf.t, sol_bdf.y[1], 'g--', linewidth=1.5,
             label=f'BDF (failed at t={sol_bdf.t[-1]:.2f})')
ax1.set_xlabel('Time (years)', fontsize=12)
ax1.set_ylabel('Prevalence I(t)', fontsize=12)
ax1.set_title('β = 3100: RK4 vs BDF', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "rk4_vs_bdf_stiffness.png", dpi=150)
plt.show()
plt.close()

# plot log scale comparison
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.semilogy(t_mod[y_mod[:, 1] > 0], y_mod[y_mod[:, 1] > 0, 1],
             'r-', linewidth=1, label='RK4 (h=1/730)')
if sol_bdf.status == 0:
    pos = sol_bdf.y[1] > 0
    ax2.semilogy(sol_bdf.t[pos], sol_bdf.y[1, pos],
                 'g--', linewidth=1.5, label='BDF')
ax2.set_xlabel('Time (years)', fontsize=12)
ax2.set_ylabel('I(t) — log scale', fontsize=12)
ax2.set_title('β = 3100: frozen phase on log scale', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "rk4_vs_bdf_stiffness_log.png", dpi=150)
plt.show()
plt.close()

