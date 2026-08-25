"""
This script generates a single-panel figure showing the endemic prevalence surface
with a dual colormap and dual colorbars. The low-prevalence region (10^-3.5 to 0.1)
uses logarithmic contour spacing and a custom light-to-strong purple colormap that matches
the exact starting color and hue of the standard 'turbo' colormap. The high-prevalence 
region (0.1 to 0.62) uses linear contour spacing and the standard 'turbo' colormap. 
The transition boundary is set at I* = 0.1, corresponding to the inflection point of the
endemic prevalence along the diagonal. It uses cubic spline interpolation for smooth level 
curves.
It exports the plot as a 300 DPI PNG file.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.interpolate import RectBivariateSpline
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Path resolution to load data correctly
SCRIPT_DIR = Path(__file__).resolve().parent
NPZ_PATH = SCRIPT_DIR.parent / "algebraic_version" / "prevalence_roots_results.npz"
SAVE_PATH = SCRIPT_DIR / "eps_prevalence_zoomed.png"

# Load the grid sweep results
if not NPZ_PATH.exists():
    print(f"Error: Data file not found at {NPZ_PATH}")
    sys.exit(1)

prevalence_results = np.load(NPZ_PATH)
list_eps1 = prevalence_results['list_eps1']
list_eps2 = prevalence_results['list_eps2']
prevalences = prevalence_results['prevalences']

# Keep only values of tilde epsilon >= 0 to match notebook logic
mask1 = list_eps1 >= 0
mask2 = list_eps2 >= 0
list_eps1 = list_eps1[mask1]
list_eps2 = list_eps2[mask2]
prevalences = prevalences[mask1][:, mask2]

# Perform cubic spline interpolation to smooth the grid and eliminate squiggly lines
# original grid: 101 x 101 -> interpolated grid: 300 x 300
spline = RectBivariateSpline(list_eps1, list_eps2, prevalences)
list_eps1_fine = np.linspace(list_eps1[0], list_eps1[-1], 300)
list_eps2_fine = np.linspace(list_eps2[0], list_eps2[-1], 300)
prevalences_fine = spline(list_eps1_fine, list_eps2_fine)
prevalences_fine = np.clip(prevalences_fine, 0, None)  # Clip to prevent spline undershoot below 0

# Define two separate sets of levels for the two regimes, splitting at I* = 0.1
# The maximum high prevalence in the dataset is ~0.614, so we end levels_high at 0.62
levels_low = np.logspace(-3.5, -1.0, 15)  # 15 levels -> 14 color bands (low-prevalence, log-spaced)
levels_high = np.linspace(0.1, 0.62, 15)   # 15 levels -> 14 color bands (high-prevalence, linearly-spaced)

# ==============================================================================
# COLOR & HUE CONFIGURATION FOR MATCHING BOUNDARIES
# ==============================================================================
# Get the standard turbo colormap and sample 14 colors for the high prevalence region
cmap_turbo = plt.colormaps['turbo']
colors_high = [cmap_turbo(val) for val in np.linspace(0.0, 1.0, len(levels_high) - 1)]

# The first color band of the high prevalence plot (represents [0.1, 0.137])
target_purple = colors_high[0]

# --- CUSTOM STARTING PURPLE (LIGHT PURPLE) ---
# To ensure the low prevalence colors belong to the exact same purple hue scheme:
# we blend target_purple with white.
# We use a 20% target_purple / 80% white blend here to make the start of the gradient
# whiter/lighter than before.
light_purple = 0.20 * np.array(target_purple) + 0.80 * np.array([1.0, 1.0, 1.0, 1.0])

# Generate 14 colors for the low prevalence bands by interpolating between light and target purple
colors_low = [
    tuple(t * np.array(target_purple) + (1 - t) * light_purple)
    for t in np.linspace(0.0, 1.0, len(levels_low) - 1)
]
# ==============================================================================

# Create the figure with a single panel
fig, ax = plt.subplots(figsize=(8.5, 7.5))

# Set facecolor for the axis to handle the 0-prevalence (disease-free) region
ax.set_facecolor('#eaeaea') # light gray for extinction region

# 1. Plot low prevalence (using the custom matched purple colors list)
pcm_low = ax.contourf(
    list_eps1_fine, list_eps2_fine, prevalences_fine, 
    levels=levels_low, 
    colors=colors_low
)
ax.contour(
    list_eps1_fine, list_eps2_fine, prevalences_fine, 
    levels=levels_low, 
    colors='black', 
    linewidths=0.3, 
    alpha=0.3
)

# 2. Plot high prevalence (using the turbo colors list)
pcm_high = ax.contourf(
    list_eps1_fine, list_eps2_fine, prevalences_fine, 
    levels=levels_high, 
    colors=colors_high
)
ax.contour(
    list_eps1_fine, list_eps2_fine, prevalences_fine, 
    levels=levels_high, 
    colors='black', 
    linewidths=0.3, 
    alpha=0.3
)

# Colorbar setup using a divider to place both colorbars side-by-side on the right
divider = make_axes_locatable(ax)

# High prevalence colorbar (closest to plot)
cax_high = divider.append_axes("right", size="5%", pad=0.15)
ticks_high = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
cbar_high = fig.colorbar(pcm_high, cax=cax_high, ticks=ticks_high)
cbar_high.ax.set_yticklabels([f"{t:.2f}" for t in ticks_high])
cbar_high.ax.tick_params(labelsize=10)
cbar_high.ax.set_title('High\nPrevalence', fontsize=10, pad=10, loc='center')

# Low prevalence colorbar (further to the right, with pad=0.9 for extra spacing)
cax_low = divider.append_axes("right", size="5%", pad=0.9)
ticks_low = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
cbar_low = fig.colorbar(pcm_low, cax=cax_low, ticks=ticks_low)
cbar_low.ax.set_yticklabels([f"{t:.4f}" if t < 0.01 else f"{t:.2f}" for t in ticks_low])
cbar_low.ax.tick_params(labelsize=10)
cbar_low.ax.set_title('Low\nPrevalence', fontsize=10, pad=10, loc='center')

# Axis labels and titles
ax.set_xlabel(r'$\tilde{\epsilon}_2$', fontsize=14)
ax.set_ylabel(r'$\tilde{\epsilon}_1$', fontsize=14)
ax.set_title(r'Endemic Prevalence $I^*$', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()

# Save the figure as a high-resolution 300 DPI image
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
plt.close()

print(f"Successfully generated and saved single dual-scale plot to: {SAVE_PATH}")
