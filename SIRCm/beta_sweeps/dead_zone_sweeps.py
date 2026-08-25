
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add SIRCm parent folder to sys.path
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

h = 1/365

# 1. Sweep beta0 from 100 to 1000
beta_test = np.linspace(100, 1000, 50)
I_final = np.zeros(len(beta_test))

print("Running broad beta0 sweep...")
for i, b in enumerate(beta_test):
    pm = {'mu': MU, 'alpha': ALPHA, 'sigma': SIGMA,
          'beta0': b, 'eps': 0.0,
          'delta_prime': DELTA_PRIME, 'gamma_prime': GAMMA_PRIME}
    _, y = solve(sirc_modified, Y0, (0, 500), h, pm)
    I_final[i] = np.mean(y[-3650:, 1])

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(beta_test, I_final, 'r-o', markersize=3)
ax1.axvline(x=MU+ALPHA, color='k', linestyle=':', label='R₀ = 1')
ax1.set_xlabel('β₀', fontsize=12)
ax1.set_ylabel('Long-run prevalence I', fontsize=12)
ax1.set_title('Endemic prevalence: SIRCm dead zone', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "dead_zone_sweeps.png", dpi=150)
plt.show()
plt.close()

# 2. Super zoom into threshold region (178 to 190)
beta_zoom = np.linspace(178, 190, 100)
I_zoom = np.zeros(len(beta_zoom))

print("Running threshold zoom sweep...")
for i, b in enumerate(beta_zoom):
    pm = {'mu': MU, 'alpha': ALPHA, 'sigma': SIGMA,
          'beta0': b, 'eps': 0.0,
          'delta_prime': DELTA_PRIME, 'gamma_prime': GAMMA_PRIME}
    _, y = solve(sirc_modified, Y0, (0, 1000), h, pm)
    I_zoom[i] = np.mean(y[-3650:, 1])

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(beta_zoom, I_zoom, 'r-o', markersize=4)
ax2.axvline(x=MU+ALPHA, color='k', linestyle=':', label='R₀ = 1')
ax2.set_xlabel('β₀', fontsize=12)
ax2.set_ylabel('Long-run prevalence I', fontsize=12)
ax2.set_title('Threshold region: where does disease persist?', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "threshold_zoom.png", dpi=150)
plt.show()
plt.close()

# 3. Simulate three regimes
y0_small = np.array([0.999, 0.000001, 0.0, 0.0])

pm_below = {'mu': MU, 'alpha': ALPHA, 'sigma': SIGMA,
            'beta0': 100, 'eps': 0.0,
            'delta_prime': DELTA_PRIME, 'gamma_prime': GAMMA_PRIME}

pm_dead = {'mu': MU, 'alpha': ALPHA, 'sigma': SIGMA,
           'beta0': 150, 'eps': 0.0,
           'delta_prime': DELTA_PRIME, 'gamma_prime': GAMMA_PRIME}

pm_above = {'mu': MU, 'alpha': ALPHA, 'sigma': SIGMA,
            'beta0': 200, 'eps': 0.0,
            'delta_prime': DELTA_PRIME, 'gamma_prime': GAMMA_PRIME}

t1, y1 = solve(sirc_modified, y0_small, (0, 10), h, pm_below)
t2, y2 = solve(sirc_modified, y0_small, (0, 10), h, pm_dead)
t3, y3 = solve(sirc_modified, y0_small, (0, 10), h, pm_above)

fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(t1, y1[:, 1], 'b-', linewidth=1.5, label='β₀ = 100 (R₀ = 0.82, below R₀=1)')
ax3.plot(t2, y2[:, 1], 'r-', linewidth=1.5, label='β₀ = 150 (R₀ = 1.23, dead zone)')
ax3.plot(t3, y3[:, 1], 'g-', linewidth=1.5, label='β₀ = 200 (R₀ = 1.64, above R₀*)')
ax3.set_xlabel('Time (years)', fontsize=12)
ax3.set_ylabel('Prevalence I(t)', fontsize=12)
ax3.set_title('Three regimes: below R₀=1, dead zone, above R₀*', fontsize=13)
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "regimes_comparison.png", dpi=150)
plt.show()
plt.close()

fig4, ax4 = plt.subplots(figsize=(12, 5))
mask = t1 > 2
ax4.plot(t1[mask], y1[mask, 1], 'b--', linewidth=1, label='β₀ = 100 (R₀ = 0.82, below R₀=1)')
ax4.plot(t2[mask], y2[mask, 1], 'r--', linewidth=1, label='β₀ = 150 (R₀ = 1.23, dead zone)')
ax4.plot(t3[mask], y3[mask, 1], 'g--', linewidth=1, label='β₀ = 200 (R₀ = 1.64, above R₀*)')
ax4.set_xlabel('Time (years)', fontsize=12)
ax4.set_ylabel('Prevalence I(t)', fontsize=12)
ax4.set_title('Three regimes: zoomed into settling behavior', fontsize=13)
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "regimes_zoomed.png", dpi=150)
plt.show()
plt.close()

print(f"I at t=10:")
print(f"  beta=100: {y1[-1, 1]:.15f}")
print(f"  beta=150: {y2[-1, 1]:.15f}")
print(f"  beta=200: {y3[-1, 1]:.15f}")
