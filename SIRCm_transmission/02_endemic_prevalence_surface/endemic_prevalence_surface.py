#!/usr/bin/env python3
"""
Endemic prevalence surface and asymptotic approximations (Transmission-driven variant)

Plots the 2D endemic prevalence surface in the (eps1, eps2) parameter space alongside 
1D diagonal slices (eps1 = eps2) with small-epsilon Taylor and large-epsilon perturbation 
series expansions for the transmission-driven SIRCm model.

OUTPUT: Figure 02
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

SCRIPT_DIR = Path(__file__).resolve().parent
NPZ_PATH = SCRIPT_DIR / "prevalence_roots_results.npz"
SAVE_PATH = SCRIPT_DIR / "02_endemic_prevalence_surface.png"

if not NPZ_PATH.exists():
    print(f"Error: Data file not found at {NPZ_PATH}")
    sys.exit(1)

prevalence_results = np.load(NPZ_PATH)
list_eps1 = prevalence_results['list_eps1']
list_eps2 = prevalence_results['list_eps2']
prevalences = prevalence_results['prevalences']

spline = RectBivariateSpline(list_eps1, list_eps2, prevalences)
list_eps1_fine = np.concatenate(([0], np.logspace(-3, 4, 2000)))
list_eps2_fine = np.concatenate(([0], np.logspace(-3, 4, 2000)))
prevalences_fine = spline(list_eps1_fine, list_eps2_fine)
prevalences_fine = np.clip(prevalences_fine, 0, None)

transition_boundary = 0.091887
max_prev_val = np.max(prevalences)
max_prev_rounded = np.ceil(max_prev_val * 100) / 100.0

levels_low = np.logspace(-3.5, np.log10(transition_boundary), 15)
levels_high = np.linspace(transition_boundary, max_prev_rounded, 15)

# Color palettes
cmap_turbo = plt.colormaps['turbo']
colors_high = [cmap_turbo(val) for val in np.linspace(0.0, 1.0, len(levels_high) - 1)]
target_purple = colors_high[0]
light_purple = 0.20 * np.array(target_purple) + 0.80 * np.array([1.0, 1.0, 1.0, 1.0])
colors_low = [
    tuple(t * np.array(target_purple) + (1 - t) * light_purple)
    for t in np.linspace(0.0, 1.0, len(levels_low) - 1)
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8), gridspec_kw={'width_ratios': [1, 1.5], 'wspace': 0.8})
ax1.set_facecolor('#eaeaea')

# --- Left Panel: 2D Contour Plot ---
pcm_low = ax1.contourf(list_eps1_fine, list_eps2_fine, prevalences_fine, levels=levels_low, colors=colors_low)
ax1.contour(list_eps1_fine, list_eps2_fine, prevalences_fine, levels=levels_low, colors='black', linewidths=0.3, alpha=0.3)

pcm_high = ax1.contourf(list_eps1_fine, list_eps2_fine, prevalences_fine, levels=levels_high, colors=colors_high)
ax1.contour(list_eps1_fine, list_eps2_fine, prevalences_fine, levels=levels_high, colors='black', linewidths=0.3, alpha=0.3)

ax1.set_xlabel(r'$\tilde{\epsilon}_2$', fontsize=16)
ax1.set_ylabel(r'$\tilde{\epsilon}_1$', fontsize=16)
ax1.set_title(r'Endemic Prevalence $I^*$', fontsize=16)
ax1.set_xscale('symlog', linthresh=1e-3)
ax1.set_yscale('symlog', linthresh=1e-3)

ticks = [0, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)
ax1.set_xticklabels(['0', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$'])
ax1.set_yticklabels(['0', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$'])
ax1.set_xlim(0, 10000.0)
ax1.set_ylim(0, 10000.0)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_box_aspect(1)

# --- Right Panel: Diagonal Profile with Series Expansions ---
profile_diag_fine = spline(list_eps1_fine, list_eps1_fine, grid=False)

SI_0 = 0.000178
MU_val = 0.02
ALPHA_val = 365.0 / 3.0
BETA0_val = 600.0

# Small-epsilon Taylor series
I0_low = 0.00114320801603584
I1_low = 2.13626659e-07
I2_low = 3.96519742e-11
I3_low = 7.35471861e-15

eps_fine = list_eps1_fine / SI_0
approx_low1 = I0_low + I1_low * eps_fine
approx_low2 = approx_low1 + I2_low * eps_fine**2
approx_low3 = approx_low2 + I3_low * eps_fine**3

# Large-epsilon perturbation series
R0 = BETA0_val / (MU_val + ALPHA_val)
I_inf = 1.0 - 1.0 / R0
h1_large = -2544.88451996
h2_large = -8146807.41597
h3_large = -5257082910.28

with np.errstate(divide='ignore', invalid='ignore'):
    approx_high1 = I_inf + h1_large / eps_fine
    approx_high2 = approx_high1 + h2_large / eps_fine**2
    approx_high3 = approx_high2 + h3_large / eps_fine**3

mask_high1 = (list_eps1_fine >= 0.4) & (approx_high1 > 1e-4) & (approx_high1 < 1.0)
mask_high2 = (list_eps1_fine >= 0.4) & (approx_high2 > 1e-4) & (approx_high2 < 1.0)
mask_high3 = (list_eps1_fine >= 0.4) & (approx_high3 > 1e-4) & (approx_high3 < 1.0)

approx_high1_plot = np.where(mask_high1, approx_high1, np.nan)
approx_high2_plot = np.where(mask_high2, approx_high2, np.nan)
approx_high3_plot = np.where(mask_high3, approx_high3, np.nan)

for approx_plot in [approx_high1_plot, approx_high2_plot, approx_high3_plot]:
    valid_indices = np.where(~np.isnan(approx_plot))[0]
    if len(valid_indices) > 0:
        first_valid_idx = valid_indices[0]
        if first_valid_idx > 0 and list_eps1_fine[first_valid_idx - 1] >= 0.4:
            approx_plot[first_valid_idx - 1] = 1e-6

ax2.plot(list_eps1_fine, profile_diag_fine, color='gray', lw=2.2, label='exact diagonal slice')
ax2.axhline(I_inf, color='black', linestyle='-.', lw=1.5, label=r'Upper limit $I^*_{\infty}$')
ax2.axhline(I0_low, color='black', linestyle=':', lw=1.5, label=r'Lower limit $I^*_0$')

ax2.plot(list_eps1_fine, approx_low1, color='lightskyblue', ls='--', lw=1.3, alpha=0.8, label=r'Low-$\tilde{\epsilon}$ expansion, order 1')
ax2.plot(list_eps1_fine, approx_low2, color='dodgerblue', ls='--', lw=1.3, alpha=0.8, label=r'Low-$\tilde{\epsilon}$ expansion, order 2')
ax2.plot(list_eps1_fine, approx_low3, color='navy', ls='--', lw=1.3, alpha=0.8, label=r'Low-$\tilde{\epsilon}$ expansion, order 3')

ax2.plot(list_eps1_fine, approx_high1_plot, color='orange', ls='--', lw=1.3, alpha=0.8, label=r'High-$\tilde{\epsilon}$ expansion, order 1')
ax2.plot(list_eps1_fine, approx_high2_plot, color='darkorange', ls='--', lw=1.3, alpha=0.8, label=r'High-$\tilde{\epsilon}$ expansion, order 2')
ax2.plot(list_eps1_fine, approx_high3_plot, color='orangered', ls='--', lw=1.3, alpha=0.8, label=r'High-$\tilde{\epsilon}$ expansion, order 3')

ax2.set_xlabel(r'$\tilde{\epsilon}$', fontsize=16)
ax2.set_ylabel(r'Endemic Prevalence $I^*$', fontsize=16)
ax2.set_title(r'Prevalence along diagonal $\tilde{\epsilon}_1=\tilde{\epsilon}_2$', fontsize=16)
ax2.set_xscale('symlog', linthresh=1e-3)
ax2.set_yscale('log')

y1 = np.log10(I0_low)
y2 = np.log10(I_inf)
y_mid = (y1 + y2) / 2.0
W = (y2 - y1) / 2.0 + 0.4
y_min = 10**(y_mid - W)
y_max = 10**(y_mid + W)

ax2.set_xticks(ticks)
ax2.set_xticklabels(['0', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^0$', r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$'])
ax2.set_xlim(0, 10000.0)
ax2.set_ylim(y_min, y_max)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.grid(True, which='both', linestyle='--', alpha=0.5)
ax2.legend(loc='lower right', fontsize=9.5, labelspacing=0.3, borderpad=0.3)

fig.canvas.draw()
pos1 = ax1.get_position()
pos2 = ax2.get_position()
ax2.set_position([pos2.x0, pos1.y0, pos2.width, pos1.height])
fig.canvas.draw()
pos2 = ax2.get_position()

# --- Colorbars ---
cax_high = fig.add_axes([pos1.x1 + 0.04, pos1.y0, 0.015, pos1.height])
ticks_high = [round(transition_boundary, 2)] + [t for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] if t <= max_prev_rounded]
cbar_high = fig.colorbar(pcm_high, cax=cax_high, ticks=ticks_high)
cbar_high.ax.set_yticklabels([f"{t:.2f}" for t in ticks_high])
cbar_high.ax.tick_params(labelsize=12)
cbar_high.ax.set_title('High\nPrevalence', fontsize=12, pad=10, loc='center')

cax_low = fig.add_axes([pos1.x1 + 0.10, pos1.y0, 0.015, pos1.height])
ticks_low = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, round(transition_boundary, 2)]
cbar_low = fig.colorbar(pcm_low, cax=cax_low, ticks=ticks_low)
cbar_low.ax.set_yticklabels([f"{t:.4f}" if t < 0.01 else f"{t:.2f}" for t in ticks_low])
cbar_low.ax.tick_params(labelsize=12)
cbar_low.ax.set_title('Low\nPrevalence', fontsize=12, pad=10, loc='center')

plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {SAVE_PATH}")
