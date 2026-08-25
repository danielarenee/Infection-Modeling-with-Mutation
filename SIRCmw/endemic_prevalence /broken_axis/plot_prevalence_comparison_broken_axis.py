import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent.parent.parent
sys.path.append(str(WORKSPACE_DIR / "SIRCmw"))

import sircmw_utils

# Model parameters (aligned with plot_prevalence_comparison.jl)
MU = 0.02
ALPHA = 365.0 / 3.0
BETA0_MAX = 1500.0
TRANSCRITICAL_BETA0 = MU + ALPHA  # ~121.6867

# Values of tilde_eps to analyze (SIRC baseline, 0.1, 0.2, 0.3, 0.4, and 0.5 to 2.0)
tilde_eps_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

def plot_branch_segments(ax, beta_vals, I_vals, stability, color, label):
    """Plots stable (solid) and unstable (dashed) segments of a branch seamlessly."""
    # Filter points that are within or close to the subplot's y-limits to avoid rendering issues
    ymin, ymax = ax.get_ylim()
    # We extend the check slightly to make sure lines cross borders nicely
    margin = (ymax - ymin) * 0.1
    
    n = len(beta_vals)
    if n == 0:
        return
        
    current_segment_beta = [beta_vals[0]]
    current_segment_I = [I_vals[0]]
    current_stability = stability[0]
    
    first_stable_label = False
    
    for i in range(1, n):
        if stability[i] != current_stability:
            current_segment_beta.append(beta_vals[i])
            current_segment_I.append(I_vals[i])
            
            linestyle = '-' if current_stability else '--'
            lbl = label if (current_stability and not first_stable_label) else None
            if current_stability:
                first_stable_label = True
                
            ax.plot(current_segment_beta, current_segment_I, color=color, linestyle=linestyle, lw=2.2, label=lbl)
            
            # Start new segment
            current_segment_beta = [beta_vals[i]]
            current_segment_I = [I_vals[i]]
            current_stability = stability[i]
        else:
            current_segment_beta.append(beta_vals[i])
            current_segment_I.append(I_vals[i])
            
    # Plot remaining trailing segment
    linestyle = '-' if current_stability else '--'
    lbl = label if (current_stability and not first_stable_label) or (not current_stability and not first_stable_label) else None
    ax.plot(current_segment_beta, current_segment_I, color=color, linestyle=linestyle, lw=2.2, label=lbl)

def main():
    cache_path = SCRIPT_DIR.parent / "log_scale" / "prevalence_comparison_data.pkl"
    
    if not cache_path.exists():
        print(f"Error: Cached data file not found at {cache_path}")
        print("Please run plot_prevalence_comparison_log.py first to compute the cache.")
        sys.exit(1)
        
    print(f"Loading cached bifurcation data from: {cache_path}")
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
        
    # --- PLOTTING ---
    print("Generating broken-axis plot...")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Arial'],
        'mathtext.fontset': 'dejavusans'
    })
    
    # Vertically stacked subplots: height ratio 2:1 for top (high) and bottom (low)
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(11, 8.5), gridspec_kw={'height_ratios': [2, 1]})
    fig.subplots_adjust(hspace=0.06)  # small vertical gap
    
    ax_top.set_facecolor('#fafafa')
    ax_bottom.set_facecolor('#fafafa')
    
    # Set y-limits for both axes to split the view
    ax_top.set_ylim(0.0022, 0.6)      # high prevalence panel
    ax_bottom.set_ylim(0.0, 0.0022)   # low prevalence panel (zoomed in)
    
    # Use the tab20 color palette to get 12 distinct colors
    colors = [plt.cm.tab20(i) for i in range(len(tilde_eps_vals))]
    
    # Add vertical indicator lines for R0 = 1
    ax_top.axvline(TRANSCRITICAL_BETA0, color='gray', linestyle=':', lw=1.5, alpha=0.7, label=r'$R_0 = 1$')
    ax_bottom.axvline(TRANSCRITICAL_BETA0, color='gray', linestyle=':', lw=1.5, alpha=0.7)
    
    # Plot curves and Hopf points on both subplots
    for te, color in zip(tilde_eps_vals, colors):
        branch = data[te]
        label = r"$\tilde{\epsilon} = 0.0$ (SIRC)" if te == 0.0 else rf"$\tilde{{\epsilon}} = {te}$"
        
        # Plot curves on both subplots
        plot_branch_segments(ax_top, branch['beta_vals'], branch['I_vals'], branch['stability'], color, label)
        plot_branch_segments(ax_bottom, branch['beta_vals'], branch['I_vals'], branch['stability'], color, label)
        
        # Plot Hopf points on both
        for hx, hy in branch['hopf_points']:
            ax_top.scatter(hx, hy, color='green', marker='D', s=35, edgecolor='black', zorder=5)
            ax_bottom.scatter(hx, hy, color='green', marker='D', s=35, edgecolor='black', zorder=5)
            
    # Add dummy entries for legend classifications (only on ax_top to avoid duplicates)
    ax_top.plot([], [], color='black', linestyle='-', lw=2.2, label='Stable equilibrium')
    ax_top.plot([], [], color='black', linestyle='--', lw=2.2, label='Unstable equilibrium')
    ax_top.scatter([], [], color='green', marker='D', s=35, edgecolor='black', label='Hopf point')
    
    # Formatting
    ax_top.set_xlim(0.0, BETA0_MAX)
    ax_bottom.set_xlim(0.0, BETA0_MAX)
    
    ax_bottom.set_xlabel(r"Contact rate $\beta_0$", fontsize=14, labelpad=8)
    
    # Add shared y-label
    fig.text(0.04, 0.5, r"Infected fraction $I^*$", va='center', rotation='vertical', fontsize=14)
    
    # Enable grid on both subplots
    ax_top.grid(True, which="both", linestyle="--", color="gray", alpha=0.25)
    ax_bottom.grid(True, which="both", linestyle="--", color="gray", alpha=0.25)
    
    ax_top.tick_params(axis='both', which='major', labelsize=12)
    ax_bottom.tick_params(axis='both', which='major', labelsize=12)
    
    # Hide the spines between ax_top and ax_bottom
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    
    ax_top.tick_params(labelbottom=False, bottom=False)  # hide ticks on the bottom of the top panel
    ax_bottom.tick_params(top=False)  # hide ticks on the top of the bottom panel
    
    # Place legend on the top subplot
    ax_top.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, frameon=True, facecolor='white', edgecolor='#ddd')
    
    # --- BROKEN AXIS SLASH MARKS ---
    d = .015  # how big to make the diagonal lines in axes coordinates
    
    # top axes: bottom-left and bottom-right break marks
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, lw=1.2)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)        # bottom-left diagonal
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # bottom-right diagonal

    # bottom axes: top-left and top-right break marks
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # top-left diagonal
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # top-right diagonal
    
    # Save plots
    png_out = SCRIPT_DIR / "prevalence_comparison_broken_axis.png"
    pdf_out = SCRIPT_DIR / "prevalence_comparison_broken_axis.pdf"
    
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_out, bbox_inches='tight')
    
    print(f"Plots saved to: {png_out.name} and {pdf_out.name}")
    plt.close()

if __name__ == '__main__':
    main()
