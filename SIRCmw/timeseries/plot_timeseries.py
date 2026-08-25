#!/usr/bin/env python3
"""
SIRCmw prevalence time series plot
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sircmw_utils import (
    sircmw,
    integrate_with_reseeding
)

def main():
    # simulation parameters
    y0 = np.array([0.2, 0.001, 0.499, 0.3])
    
    # S*I scaling factor 
    # si_0 = y0[0] * y0[1] # 0.0002
    si_0 = y0[1]
    print(si_0)
    
    # Transmission rate and integration duration (years)
    beta = 600.0
    years = 50.0

    # --- 2. Color System ---
    # Define a clean blue gradient plus a contrasting warm orange for the SIRC baseline
    blues = ['#5D9CEC', '#3B5998', '#24426E'] # Light, medium, and soft dark blue
    sirc_orange = '#E64B35'                   # Warm orange/coral

    # Setup the side-by-side subplot panels (1 row, 3 columns)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=False, dpi=300)

    # =========================================================================
    # Panel 1: Low feedback (tilde_eps = 0.0, 0.2, 0.3)
    # =========================================================================
    left_eps_vals = [0.0, 0.2, 0.3]
    left_colors = [sirc_orange, blues[0], blues[1]]
    ax_left = axs[0]
    
    print("Simulating Low Feedback Panel...")
    for te, col in zip(left_eps_vals, left_colors):
        # Convert scaled feedback (tilde_eps) to physical feedback (eps)
        eps = te / si_0
        p_mod = {'beta0': beta, 'eps': eps}
        
        # Integrate SIRCmw ODE system with numerical re-seeding controls
        t, Y, _ = integrate_with_reseeding(
            sircmw, (0, years), y0, p_mod,
            threshold=1e-15, I_seed=1e-14,
            method='DOP853', rtol=1e-6, atol=1e-9
        )
        
        # Format SIRC baseline distinctively in the legend
        label = f"$\\tilde{{\\varepsilon}} = {te:.1f}$ (SIRC)" if te == 0.0 else f"$\\tilde{{\\varepsilon}} = {te:.1f}$"
        ax_left.plot(t, Y[1, :], label=label, color=col, linewidth=1.75)

    ax_left.set_title("Low feedback", fontsize=12, pad=10)
    ax_left.set_ylabel('Prevalence I(t)', fontsize=12)
    ax_left.legend(frameon=True, edgecolor='#e5e5e5', loc='upper right')

    # =========================================================================
    # Panel 2: Moderate feedback (tilde_eps = 1.54)
    # =========================================================================
    middle_eps_vals = [1.54]
    middle_colors = [blues[1]] # Medium blue
    ax_middle = axs[1]
    
    print("Simulating Moderate Feedback Panel...")
    for te, col in zip(middle_eps_vals, middle_colors):
        eps = te / si_0
        p_mod = {'beta0': beta, 'eps': eps}
        t, Y, _ = integrate_with_reseeding(
            sircmw, (0, years), y0, p_mod,
            threshold=1e-15, I_seed=1e-14,
            method='DOP853', rtol=1e-6, atol=1e-9
        )
        ax_middle.plot(t, Y[1, :], label=f"$\\tilde{{\\varepsilon}} = {te:.2f}$", color=col, linewidth=1.5)

    ax_middle.set_title("Moderate feedback", fontsize=12, pad=10)
    ax_middle.legend(frameon=True, edgecolor='#e5e5e5', loc='upper right')

    # =========================================================================
    # Panel 3: High feedback (tilde_eps = 1.55, 1.60)
    # =========================================================================
    right_eps_vals = [1.55, 1.60]
    right_colors = [blues[0], blues[1]] # Light and medium blue
    ax_right = axs[2]
    
    print("Simulating High Feedback Panel...")
    for te, col in zip(right_eps_vals, right_colors):
        eps = te / si_0
        p_mod = {'beta0': beta, 'eps': eps}
        t, Y, _ = integrate_with_reseeding(
            sircmw, (0, years), y0, p_mod,
            threshold=1e-15, I_seed=1e-14,
            method='DOP853', rtol=1e-6, atol=1e-9
        )
        ax_right.plot(t, Y[1, :], label=f"$\\tilde{{\\varepsilon}} = {te:.2f}$", color=col, linewidth=1.75)

    ax_right.set_title("High feedback", fontsize=12, pad=10)
    ax_right.legend(frameon=True, edgecolor='#e5e5e5', loc='upper right')

    # --- 3. Subplot Styling and Layout ---
    for ax in axs:
        ax.set_xlabel('Time (years)', fontsize=12, labelpad=8)
        # Remove top/right frame borders for cleaner appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        # Add light gray background grids
        ax.grid(True, linestyle='-', linewidth=0.5, color='#e5e5e5')
        ax.set_xlim(0, years)
        ax.set_ylim(bottom=0.0)

    plt.tight_layout()

    # Save output plot
    save_path = Path(__file__).resolve().parent / "timeseries_three_panels.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved three-panel plot to {save_path.resolve()}")
    plt.close()

if __name__ == "__main__":
    main()
