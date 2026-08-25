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
beta_zoom = np.linspace(122, 300, 40)

fig, ax = plt.subplots(figsize=(10, 6))

print("Running threshold feedback sweep...")
for factor in [0.5, 1, 2]:
    I_test = np.zeros(len(beta_zoom))
    for i, b in enumerate(beta_zoom):
        pm = {'mu': MU, 'alpha': ALPHA, 'sigma': SIGMA,
              'beta0': b, 'eps': 0.0,
              'delta_prime': DELTA_PRIME * factor,
              'gamma_prime': GAMMA_PRIME * factor}
        _, y = solve(sirc_modified, Y0, (0, 500), h, pm)
        I_test[i] = np.mean(y[-3650:, 1])
    peak = I_test.max()
    if peak > 0:
        ax.plot(beta_zoom, I_test / peak, '-o', markersize=3,
                label=f'{factor}x (peak I = {peak:.4f})')

ax.axvline(x=MU+ALPHA, color='k', linestyle=':', label='R₀ = 1')
ax.set_xlabel('β₀', fontsize=12)
ax.set_ylabel('Prevalence (normalized to peak)', fontsize=12)
ax.set_title('Threshold location vs feedback strength', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / "threshold_feedback_analysis.png", dpi=150)
plt.show()
plt.close()
