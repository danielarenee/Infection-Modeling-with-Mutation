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

# Constants matching plot_prevalence_comparison_log.py
MU = 0.02
ALPHA = 365.0 / 3.0
BETA0_MAX = 1500.0
TRANSCRITICAL_BETA0 = MU + ALPHA  # ~121.6867

# Values of tilde_eps (excluding 0.0 since it is the baseline I_0)
tilde_eps_vals = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

def plot_branch_segments(ax, beta_vals, y_vals, stability, color, label):
    """Plots stable (solid) and unstable (dashed) segments of a branch seamlessly."""
    n = len(beta_vals)
    if n == 0:
        return
        
    current_segment_beta = [beta_vals[0]]
    current_segment_y = [y_vals[0]]
    current_stability = stability[0]
    
    first_stable_label = False
    
    for i in range(1, n):
        # Skip NaNs in plotting (division by zero regions)
        if np.isnan(y_vals[i]) or np.isnan(current_segment_y[-1]):
            # If we hit a NaN, draw what we have and reset
            if len(current_segment_beta) > 1 and not np.isnan(current_segment_y[0]):
                linestyle = '-' if current_stability else '--'
                lbl = label if (current_stability and not first_stable_label) else None
                if current_stability:
                    first_stable_label = True
                ax.plot(current_segment_beta, current_segment_y, color=color, linestyle=linestyle, lw=2.2, label=lbl)
            
            current_segment_beta = [beta_vals[i]]
            current_segment_y = [y_vals[i]]
            current_stability = stability[i]
            continue
            
        if stability[i] != current_stability:
            # Add boundary transition point to prevent gaps between segments
            current_segment_beta.append(beta_vals[i])
            current_segment_y.append(y_vals[i])
            
            linestyle = '-' if current_stability else '--'
            lbl = label if (current_stability and not first_stable_label) else None
            if current_stability:
                first_stable_label = True
                
            ax.plot(current_segment_beta, current_segment_y, color=color, linestyle=linestyle, lw=2.2, label=lbl)
            
            # Start new segment
            current_segment_beta = [beta_vals[i]]
            current_segment_y = [y_vals[i]]
            current_stability = stability[i]
        else:
            current_segment_beta.append(beta_vals[i])
            current_segment_y.append(y_vals[i])
            
    # Plot remaining trailing segment
    if len(current_segment_beta) > 1 and not np.isnan(current_segment_y[0]):
        linestyle = '-' if current_stability else '--'
        lbl = label if (current_stability and not first_stable_label) or (not current_stability and not first_stable_label) else None
        ax.plot(current_segment_beta, current_segment_y, color=color, linestyle=linestyle, lw=2.2, label=lbl)

def main():
    cache_path = SCRIPT_DIR.parent / "log_scale" / "prevalence_comparison_data.pkl"
    
    if not cache_path.exists():
        print(f"Error: Cached data file not found at {cache_path}")
        print("Please run plot_prevalence_comparison_log.py first to compute the cache.")
        sys.exit(1)
        
    print(f"Loading cached bifurcation data from: {cache_path}")
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
        
    # Baseline curve I_0
    I0_branch = data[0.0]
    beta_vals = np.array(I0_branch['beta_vals'])
    I0_vals = np.array(I0_branch['I_vals'])
    
    # Matplotlib styling
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Arial'],
        'mathtext.fontset': 'dejavusans'
    })
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7.5))
    ax1.set_facecolor('#fafafa')
    ax2.set_facecolor('#fafafa')
    ax3.set_facecolor('#fafafa')
    
    # Map colors for all 9 epsilons using tab10 colormap
    all_tilde_eps_vals = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    colors = [plt.cm.tab10(i) for i in range(len(all_tilde_eps_vals))]
    
    # Add vertical R0 = 1 indicator lines
    for ax in (ax1, ax2, ax3):
        ax.axvline(TRANSCRITICAL_BETA0, color='gray', linestyle=':', lw=1.5, alpha=0.7, label=r'$R_0 = 1$')
        
    # Process and plot each epsilon curve
    for i, te in enumerate(all_tilde_eps_vals):
        branch = data[te]
        I_vals = np.array(branch['I_vals'])
        stability = branch['stability']
        color = colors[i]
        label = r"$\tilde{\epsilon} = 0.0$ (SIRC)" if te == 0.0 else rf"$\tilde{{\epsilon}} = {te}$"
        
        # 1. Left Panel: Prevalence I*
        plot_branch_segments(ax1, beta_vals, I_vals, stability, color, label)
        
        # Plot Hopf points on ax1
        for hx, hy in branch['hopf_points']:
            ax1.scatter(hx, hy, color='green', marker='D', s=55, edgecolor='black', zorder=5)
            
        # For te > 0.0, plot difference and relative change
        if te > 0.0:
            # 2. Middle Panel: Absolute Difference: I_eps - I_0
            abs_diff = I_vals - I0_vals
            plot_branch_segments(ax2, beta_vals, abs_diff, stability, color, label)
            
            # 3. Right Panel: Relative Change: (I_eps - I_0) / I_0
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_change = np.where(I0_vals > 0.0, (I_vals - I0_vals) / I0_vals, np.nan)
            plot_branch_segments(ax3, beta_vals, rel_change, stability, color, label)
            
            # Plot Hopf points on ax2 and ax3
            for hx, hy in branch['hopf_points']:
                # Interpolate baseline I_0 at the exact hx parameter value
                I0_at_hx = np.interp(hx, beta_vals, I0_vals)
                
                # Hopf point absolute diff
                hy_abs = hy - I0_at_hx
                ax2.scatter(hx, hy_abs, color='green', marker='D', s=55, edgecolor='black', zorder=5)
                
                # Hopf point relative change
                if I0_at_hx > 0.0:
                    hy_rel = (hy - I0_at_hx) / I0_at_hx
                    ax3.scatter(hx, hy_rel, color='green', marker='D', s=55, edgecolor='black', zorder=5)
                    
    # Add legend helper classification lines
    for ax in (ax1, ax2, ax3):
        ax.plot([], [], color='black', linestyle='-', lw=2.2, label='Stable equilibrium')
        ax.plot([], [], color='black', linestyle='--', lw=2.2, label='Unstable equilibrium')
        ax.scatter([], [], color='green', marker='D', s=55, edgecolor='black', label='Hopf point')
        ax.set_xscale('linear')
        ax.set_xlim(0, BETA0_MAX)
        ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.25)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlabel(r"Contact rate $\beta_0$", fontsize=14, labelpad=8)
        ax.legend(loc='best', fontsize=10, frameon=True, facecolor='white', edgecolor='#ddd')
        
    ax1.set_ylabel(r"Infected fraction $I^*$", fontsize=14, labelpad=8)
    ax1.set_title(r"Prevalence $I^*$", fontsize=14, pad=12)
    
    ax2.set_ylabel(r"Absolute difference $I^*_{\tilde{\epsilon}} - I^*_0$", fontsize=14, labelpad=8)
    ax2.set_title(r"Absolute Prevalence Difference", fontsize=14, pad=12)
    
    ax3.set_ylabel(r"Relative change $(I^*_{\tilde{\epsilon}} - I^*_0) / I^*_0$", fontsize=14, labelpad=8)
    ax3.set_title(r"Relative Prevalence Change", fontsize=14, pad=12)
    
    plt.tight_layout()
    
    png_out = SCRIPT_DIR / "relative_prevalence_comparison.png"
    pdf_out = SCRIPT_DIR / "relative_prevalence_comparison.pdf"
    
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_out, bbox_inches='tight')
    
    print(f"Plots saved to: {png_out.name} and {pdf_out.name}")
    plt.close()

if __name__ == '__main__':
    main()
