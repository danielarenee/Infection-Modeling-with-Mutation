import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import fsolve

sys.path.append(str(Path(__file__).parent.parent))

from sircm_utils import (
    sirc,
    sirc_modified,
    solve,
    MU,
    ALPHA,
    DELTA,
    GAMMA,
    SIGMA,
    Y0,
    DELTA_PRIME,
    GAMMA_PRIME
)

h = 1/365

orig_params = {
    'mu': MU, 'alpha': ALPHA, 'delta': DELTA, 'gamma': GAMMA,
    'sigma': SIGMA, 'beta0': 800, 'eps': 0.0,
}

mod_params = {
    'mu': MU, 'alpha': ALPHA, 'sigma': SIGMA,
    'beta0': 800, 'eps': 0.0,
    'delta_prime': DELTA_PRIME, 'gamma_prime': GAMMA_PRIME,
}

# 50-year transient simulation
t_orig, y_orig = solve(sirc, Y0, (0, 50), h, orig_params)
t_mod,  y_mod  = solve(sirc_modified, Y0, (0, 50), h, mod_params)

# Plot I(t) and C(t) comparison
fig1, axes1 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes1[0].plot(t_orig, y_orig[:, 1], 'b-',  linewidth=1, label='Original SIRC')
axes1[0].plot(t_mod,  y_mod[:, 1],  'r--', linewidth=1, label='Modified SIRC')
axes1[0].set_ylabel('Prevalence I(t)', fontsize=12)
axes1[0].set_title('Original vs Modified SIRC: I(t) and C(t)', fontsize=13)
axes1[0].legend(fontsize=11)
axes1[0].grid(True, alpha=0.3)

axes1[1].plot(t_orig, y_orig[:, 3], 'b-',  linewidth=1, label='C (original)')
axes1[1].plot(t_mod,  y_mod[:, 3],  'r--', linewidth=1, label='C (modified)')
axes1[1].set_xlabel('Time (years)', fontsize=12)
axes1[1].set_ylabel('Cross-immune fraction', fontsize=12)
axes1[1].legend(fontsize=10)
axes1[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "sirc_sircm_IC_50yrs.png", dpi=150)
plt.show()
plt.close()

# Plot S(t) and R(t) comparison
fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes2[0].plot(t_orig, y_orig[:, 0], 'b-',  linewidth=1, label='Original SIRC')
axes2[0].plot(t_mod,  y_mod[:, 0],  'r--', linewidth=1, label='Modified SIRC')
axes2[0].set_ylabel('Susceptible fraction', fontsize=12)
axes2[0].set_title('Original vs Modified SIRC: S(t) and R(t)', fontsize=13)
axes2[0].legend(fontsize=11)
axes2[0].grid(True, alpha=0.3)

axes2[1].plot(t_orig, y_orig[:, 2], 'b-',  linewidth=1, label='R (original)')
axes2[1].plot(t_mod,  y_mod[:, 2],  'r--', linewidth=1, label='R (modified)')
axes2[1].set_xlabel('Time (years)', fontsize=12)
axes2[1].set_ylabel('Recovered fraction', fontsize=12)
axes2[1].legend(fontsize=10)
axes2[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "sirc_sircm_SR_50yrs.png", dpi=150)
plt.show()
plt.close()

# 200-year SIRCm all compartments plot
t_mod_200, y_mod_200 = solve(sirc_modified, Y0, (0, 200), h, mod_params)

fig3, ax3 = plt.subplots(figsize=(14, 6))
ax3.plot(t_mod_200, y_mod_200[:, 0], 'b-',  linewidth=1, label='S (susceptible)')
ax3.plot(t_mod_200, y_mod_200[:, 1], 'r-',  linewidth=1, label='I (infected)')
ax3.plot(t_mod_200, y_mod_200[:, 2], 'g-',  linewidth=1, label='R (recovered)')
ax3.plot(t_mod_200, y_mod_200[:, 3], '-',   linewidth=1, color='orange', label='C (cross-immune)')
ax3.set_xlabel('Time (years)', fontsize=12)
ax3.set_ylabel('Fraction of population', fontsize=12)
ax3.set_title('SIRCm hand coded RK4 (β₀ = 800, no seasonality, 200 yrs)', fontsize=13)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "sircm_all_compartments_200yrs.png", dpi=150)
plt.show()
plt.close()

# Sum check
total_200 = y_mod_200[:, 0] + y_mod_200[:, 1] + y_mod_200[:, 2] + y_mod_200[:, 3]
print(f"Sum check (200 yrs): min = {total_200.min():.10f}, max = {total_200.max():.10f}")

# 500-year long-term trajectories
t_long, y_long = solve(sirc_modified, Y0, (0, 500), h, mod_params)
t_long_orig, y_long_orig = solve(sirc, Y0, (0, 500), h, orig_params)

# Combined 500-year plot
fig4, ax4 = plt.subplots(figsize=(14, 6))
ax4.plot(t_long_orig, y_long_orig[:, 1], 'b-', linewidth=0.5, label='Original SIRC')
ax4.plot(t_long, y_long[:, 1], 'r-', linewidth=0.5, label='Modified SIRC')
ax4.set_xlabel('Time (years)', fontsize=12)
ax4.set_ylabel('Prevalence I(t)', fontsize=12)
ax4.set_title('Original vs Modified SIRC — 500 years', fontsize=13)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "sirc_sircm_long_500yrs.png", dpi=150)
plt.show()
plt.close()

# Prints and Comparisons
print("\nFinal State Comparison at t=500:")
print(f"Original SIRC: S={y_long_orig[-1, 0]:.6f}, I={y_long_orig[-1, 1]:.10f}, R={y_long_orig[-1, 2]:.6f}, C={y_long_orig[-1, 3]:.6f}")
print(f"Modified SIRC: S={y_long[-1, 0]:.6f}, I={y_long[-1, 1]:.10f}, R={y_long[-1, 2]:.6f}, C={y_long[-1, 3]:.6f}")

# Analytical endemic equilibrium of SIRC at beta=800
def eq_800(x):
    return sirc(0.0, x, orig_params)
eq_orig = fsolve(eq_800, [0.2, 0.001, 0.499, 0.3])
print("\nOriginal SIRC Endemic Equilibrium (fsolve at beta=800):")
print(f"  S={eq_orig[0]:.6f}  I={eq_orig[1]:.10f}  R={eq_orig[2]:.6f}  C={eq_orig[3]:.6f}")

print("\nDisease checking at checks:")
for t_check in [300, 350, 400, 450, 500]:
    idx = np.argmin(np.abs(t_long - t_check))
    print(f"  t={t_check}: I = {y_long[idx, 1]:.10f}")
