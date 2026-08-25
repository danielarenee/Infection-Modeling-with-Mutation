"""
This script generates the second plot from algebraic_version/surface_epsilon_sircmw_prevalence_roots.ipynb.
It plots a 2-panel figure showing the endemic prevalence surface and a diagonal slice profile,
and exports it as a 300 DPI PNG file.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

# Path resolution to import utility or load data correctly
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent.parent.parent))

from SIRCmw_I.sircmw_I_utils import MU, ALPHA

NPZ_PATH = SCRIPT_DIR.parent / "algebraic_version" / "prevalence_roots_results.npz"
SAVE_PATH = SCRIPT_DIR.parent / "algebraic_version" / "eps_prevalence_2.png"

# Load the grid sweep results
if not NPZ_PATH.exists():
    print(f"Error: Data file not found at {NPZ_PATH}")
    sys.exit(1)

prevalence_results = np.load(NPZ_PATH)
list_eps1 = prevalence_results['list_eps1']
list_eps2 = prevalence_results['list_eps2']
prevalences = prevalence_results['prevalences']
stabilities = prevalence_results['stabilities']

# Keep only values of tilde epsilon >= 0 to match notebook logic
mask1 = list_eps1 >= 0
mask2 = list_eps2 >= 0
list_eps1 = list_eps1[mask1]
list_eps2 = list_eps2[mask2]
prevalences = prevalences[mask1][:, mask2]
stabilities = stabilities[mask1][:, mask2]

# Create regular grid interpolator matching notebook cell 6
interp = RegularGridInterpolator(
    (list_eps1, list_eps2),
    prevalences,
    method='linear',        
    bounds_error=False,
    fill_value=np.nan, 
)

eps1_a, eps2_a = 0.0, 0.0
eps1_b, eps2_b = 2.5, 2.5

# Define the line by two endpoints in (eps1, eps2) space
p_start = np.array([eps1_a, eps2_a])
p_end   = np.array([eps1_b, eps2_b])

n = 300
t = np.linspace(0, 1, n)
line = p_start + t[:, None] * (p_end - p_start)   # shape (n, 2): col 0 = eps1, col 1 = eps2

# Query order must match the grid: (eps1, eps2)
profile = interp(line)

# Create the figure with the exact styling of the second plot in the notebook
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left panel: pcolormesh of the prevalence surface with the diagonal slice line
im = ax1.pcolormesh(list_eps1, list_eps2, prevalences, cmap='turbo', shading='auto')
ax1.plot(line[:, 1], line[:, 0], color='gray', label='exact diagonal slice')
ax1.set_xlabel(r'$\tilde{\epsilon}_2$', fontsize=14)
ax1.set_ylabel(r'$\tilde{\epsilon}_1$', fontsize=14)
ax1.set_title('Analytical Roots prevalence $I^*$', fontsize=14)
ax1.legend(fontsize=10)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_aspect('equal', adjustable='box')

# Right panel: 1D profile of the diagonal slice
ax2.plot(np.linspace(eps1_a, eps1_b, n), profile, color='gray', label='exact diagonal slice')

# Calculate I_inf* = 1 - R_0^-1
R0 = 500.0 / (MU + ALPHA)
I_inf = 1.0 - 1.0 / R0

# Plot upper limit horizontal line (black, dot-dash)
ax2.axhline(I_inf, color='black', linestyle='-.',
           label=r'Upper limit $I^*_{\infty}$')

# Plot lower limit horizontal line (black, dotted)
ax2.axhline(profile[0], color='black', linestyle=':',
           label=r'Lower limit $I^*_0$')

ax2.set_xlabel(r'$\tilde{\epsilon}_2$', fontsize=14)
ax2.set_ylabel('Analytical Prevalence $I^*$', fontsize=14)
ax2.legend(fontsize=10)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.grid(True)

plt.tight_layout()

# Save figure in 300 dpi matching the prompt requirements
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
plt.close()

print(f"Successfully generated and saved plot to: {SAVE_PATH}")
