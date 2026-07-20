"""
 Analyzes compartment trends near infectious outbreaks
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from sircm_utils import (
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

mod_params = {
    'mu': MU, 'alpha': ALPHA, 'sigma': SIGMA,
    'beta0': 800, 'eps': 0.0,
    'delta_prime': DELTA_PRIME, 'gamma_prime': GAMMA_PRIME,
}

t_mod, y_mod = solve(sirc_modified, Y0, (0, 200), h, mod_params)

# effective Rates Calculation
beta_val = 800
drift_arr = beta_val * y_mod[:, 0] * y_mod[:, 1]
delta_eff_arr = DELTA_PRIME * drift_arr
gamma_eff_arr = GAMMA_PRIME * drift_arr

# dataFrame compilation and print
simulation_data = {
    'Time (Years)': t_mod,
    'S(t)': y_mod[:, 0],
    'I(t)': y_mod[:, 1],
    'R(t)': y_mod[:, 2],
    'C(t)': y_mod[:, 3],
    'Beta': np.full_like(t_mod, beta_val),
    'Delta_eff': delta_eff_arr,
    'Gamma_eff': gamma_eff_arr,
    'Original_Delta': np.full_like(t_mod, DELTA),
    'Original_Gamma': np.full_like(t_mod, GAMMA),
}
df_sircm = pd.DataFrame(simulation_data)
print("\nFirst 10 time steps:")
print(df_sircm.head(10).to_string(index=False))

# spike Analysis
I_mod_arr = y_mod[:, 1]
threshold = 0.001

spike_times = []
spike_I = []
spike_S_before = []
spike_R_before = []
spike_C_before = []

for n in range(1, len(I_mod_arr) - 1):
    if (I_mod_arr[n] > I_mod_arr[n - 1] and
            I_mod_arr[n] > I_mod_arr[n + 1] and
            I_mod_arr[n] > threshold):
        spike_times.append(t_mod[n])
        spike_I.append(I_mod_arr[n])

        # find values 0.1 years before the spike
        pre_idx = np.argmin(np.abs(t_mod - (t_mod[n] - 0.1)))
        spike_S_before.append(y_mod[pre_idx, 0])
        spike_R_before.append(y_mod[pre_idx, 2])
        spike_C_before.append(y_mod[pre_idx, 3])

spike_times = np.array(spike_times)
intervals = np.diff(spike_times)

print("\nSpike analysis:")
print(f"{'Spike':>5} {'Time':>8} {'Interval':>10} {'Peak I':>10} {'S before':>10} {'R before':>10} {'C before':>10}")
for i in range(len(spike_times)):
    intv = f"{intervals[i]:.2f}" if i < len(intervals) else "—"
    print(f"{i + 1:>5} {spike_times[i]:>8.2f} {intv:>10} {spike_I[i]:>10.6f} {spike_S_before[i]:>10.4f} {spike_R_before[i]:>10.4f} {spike_C_before[i]:>10.4f}")

# plot spike trends
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
axes2[0,0].plot(range(1, len(spike_I)+1), spike_I, 'ro-')
axes2[0,0].set_ylabel('Peak I')
axes2[0,0].set_title('Spike amplitude')
axes2[0,0].grid(True, alpha=0.3)

axes2[0,1].plot(range(1, len(intervals)+1), intervals, 'ko-')
axes2[0,1].set_ylabel('Years')
axes2[0,1].set_title('Inter-spike interval')
axes2[0,1].grid(True, alpha=0.3)

axes2[1,0].plot(range(1, len(spike_S_before)+1), spike_S_before, 'bo-', label='S')
axes2[1,0].plot(range(1, len(spike_R_before)+1), spike_R_before, 'go-', label='R')
axes2[1,0].set_ylabel('Fraction')
axes2[1,0].set_title('Pre-spike S and R')
axes2[1,0].legend()
axes2[1,0].grid(True, alpha=0.3)

axes2[1,1].plot(range(1, len(spike_C_before)+1), spike_C_before, 'o-', color='orange')
axes2[1,1].set_ylabel('C fraction')
axes2[1,1].set_title('Pre-spike C')
axes2[1,1].grid(True, alpha=0.3)

for ax in axes2.flat:
    ax.set_xlabel('Spike number')

plt.tight_layout()
plt.savefig(Path(__file__).parent / "spike_trends.png", dpi=150)
plt.show()
plt.close()
