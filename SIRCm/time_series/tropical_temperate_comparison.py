"""
Recreation of temperate/tropical regimes time series plots from Casagrandi's paper (SIRC)
and comparison with SIRCm
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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

# TROPICAL SIMULATIONS

tropical_nonseasonal_params = {
    'mu': MU, 'alpha': ALPHA, 'delta': DELTA,
    'gamma': GAMMA, 'sigma': SIGMA,
    'beta0': 1200, 'eps': 0.0,
}

tropical_params = {
    'mu': MU, 'alpha': ALPHA, 'delta': DELTA,
    'gamma': GAMMA, 'sigma': SIGMA,
    'beta0': 1200, 'eps': 0.07,
}

tropical_mod_nonseasonal_params = {
    'mu': MU, 'alpha': ALPHA, 'sigma': SIGMA,
    'beta0': 1200, 'eps': 0.0,
    'delta_prime': DELTA_PRIME, 'gamma_prime': GAMMA_PRIME,
}

tropical_mod_params = {
    'mu': MU, 'alpha': ALPHA, 'sigma': SIGMA,
    'beta0': 1200, 'eps': 0.07,
    'delta_prime': DELTA_PRIME, 'gamma_prime': GAMMA_PRIME,
}

t_trop, y_trop = solve(sirc, Y0, (0, 300), 1/365, tropical_params)
t_trop_nonseasonal, y_trop_nonseasonal = solve(sirc, Y0, (0, 300), 1/365, tropical_nonseasonal_params)
t_trop_m, y_trop_m = solve(sirc_modified, Y0, (0, 300), 1/365, tropical_mod_params)
t_trop_m_nonseasonal, y_trop_m_nonseasonal = solve(sirc_modified, Y0, (0, 300), 1/365, tropical_mod_nonseasonal_params)

mask_trop = t_trop > 290  # last 50 years of the 300-year run

fig1, ax1 = plt.subplots(figsize=(10, 9))
ax1.plot((t_trop_nonseasonal[mask_trop] - 250) * 12, y_trop_nonseasonal[mask_trop, 1], 'b--', linewidth=1.5, label='Original SIRC')
ax1.plot((t_trop[mask_trop] - 250) * 12, y_trop[mask_trop, 1], 'b-',  linewidth=1.5, label='Original SIRC (seasonally adjusted)')
ax1.plot((t_trop_m_nonseasonal[mask_trop] - 250) * 12, y_trop_m_nonseasonal[mask_trop, 1], 'r--', linewidth=1.5, label='Modified SIRC')
ax1.plot((t_trop_m[mask_trop] - 250) * 12, y_trop_m[mask_trop, 1], 'r-',  linewidth=1.5, label='Modified SIRC (seasonally adjusted)')
ax1.set_xlabel('Time (months)', fontsize=12)
ax1.set_ylabel('Prevalence I(t)', fontsize=12)
ax1.set_title('Tropical Regime (ε = 0.07, β₀ = 1200)', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "tropical_comparison.png", dpi=150)
plt.show()
plt.close()

# TEMPERATE SIMULATIONS

temperate_nonseasonal_params = {
    'mu': MU, 'alpha': ALPHA, 'delta': DELTA,
    'gamma': GAMMA, 'sigma': SIGMA,
    'beta0': 400, 'eps': 0.0,
}

temperate_params = {
    'mu': MU, 'alpha': ALPHA, 'delta': DELTA,
    'gamma': GAMMA, 'sigma': SIGMA,
    'beta0': 400, 'eps': 0.18,
}

temperate_mod_params = {
    'mu': MU, 'alpha': ALPHA, 'sigma': SIGMA,
    'beta0': 400, 'eps': 0.18,
    'delta_prime': DELTA_PRIME, 'gamma_prime': GAMMA_PRIME,
}

t_temp_nonseasonal, y_temp_nonseasonal = solve(sirc, Y0, (0, 300), 1/365, temperate_nonseasonal_params)
t_temp, y_temp = solve(sirc, Y0, (0, 300), 1/365, temperate_params)
t_temp_m, y_temp_m = solve(sirc_modified, Y0, (0, 300), 1/365, temperate_mod_params)

mask_temp = t_temp > 250  # last 50 years

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(t_temp[mask_temp] - 250, y_temp[mask_temp, 1], 'k-',  linewidth=1, label='Original SIRC (seasonally adjusted)')
ax2.plot(t_temp_m[mask_temp] - 250, y_temp_m[mask_temp, 1], 'r--', linewidth=1, label='Modified SIRC (seasonally adjusted)')
ax2.set_xlabel('Time (years)', fontsize=12)
ax2.set_ylabel('Prevalence I(t)', fontsize=12)
ax2.set_title('Temperate Regime (ε = 0.18, β₀ = 400)', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "temperate_comparison.png", dpi=150)
plt.show()
plt.close()
