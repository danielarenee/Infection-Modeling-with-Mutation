import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add SIRCmw directory to path to import sircmw_utils
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent.parent))

from sircmw_I_utils import (
    sircmw_jacobian,
    get_algebraic_equilibria
)

# Contact rate
BETA = 600

# Scaling factor: choose between I_0 (0.001) and S_0 * I_0 (0.0002)
# scale_factor = 0.0002 # original SI_0 scaling
scale_factor = 0.001 # new I_0 scaling

TILDE_EPS_LIMITS = (0.0, 2.0)
NUM_POINTS = 2000

# Output filename
OUTPUT_FILENAME = "sircmw_eigenvalues_stacked.png"

# Matplotlib publication settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10
})

def main():
    p = {'beta0': BETA, 'si_0': scale_factor}
    tilde_eps_vals = np.linspace(TILDE_EPS_LIMITS[0], TILDE_EPS_LIMITS[1], NUM_POINTS)
    
    print(f"Sweeping {NUM_POINTS} points for tilde_eps ∈ {TILDE_EPS_LIMITS}...")
    rows = []
    for te in tilde_eps_vals:
        eqs = get_algebraic_equilibria(te, p)
        for eq in eqs:
            rows.append((te, *eq))
            
    data = np.array(rows)
    te_plot = data[:, 0]
    n_eq = len(data)
    
    print("Computing Jacobian eigenvalues at each equilibrium...")
    eigvals = np.empty((n_eq, 4), dtype=complex)
    for i, row in enumerate(data):
        te = row[0]
        eq = row[1:]
        eps = te / scale_factor
        J = sircmw_jacobian(eq, eps, p=p)
        eigs = np.linalg.eigvals(J)
        # Sort eigenvalues: by real component first, then imaginary component
        idx = np.lexsort((-eigs.imag, -eigs.real))
        eigvals[i] = eigs[idx]

    # Create figure with 2 subplots side-by-side (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 4.2))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    
    # Plot components on both panels
    for ax in (ax1, ax2):
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5, zorder=0)
        for k in range(4):
            ax.plot(te_plot, eigvals[:, k].real, color=colors[k], linestyle='-', linewidth=1.5,
                    label=fr'Re($\lambda_{k+1}$)' if ax == ax1 else "")
            ax.plot(te_plot, eigvals[:, k].imag, color=colors[k], linestyle=':', linewidth=1.5,
                    label=fr'Im($\lambda_{k+1}$)' if ax == ax1 else "")
            
    # Set labels, limits, and grids for side-by-side panels
    
    # Left panel (Full spectrum): cut below -800 but let the top auto-scale
    ax1.set_ylim(bottom=-800)
    ax1.set_ylabel('Eigenvalue component', labelpad=10)
    ax1.set_xlabel(r'Common Mutation Feedback $\tilde{\epsilon}$', labelpad=10)
    ax1.set_title(fr'Full Jacobian Spectrum ($\beta_0 = {BETA}$)', pad=10)
    ax1.grid(True, alpha=0.25)
    
    # Place legend in the lower left corner of the left plot (ncol=2 is cleaner for side-by-side)
    ax1.legend(loc='lower left', ncol=2, framealpha=0.9)
    
    # Right panel (Zoomed spectrum): zoom in on tilde_eps and eigenvalue components
    ax2.set_xlim(0.0, 2.0)
    ax2.set_ylim(-25, 15)
    ax2.set_ylabel('Eigenvalue component', labelpad=10)
    ax2.set_xlabel(r'Common Mutation Feedback $\tilde{\epsilon}$', labelpad=10)
    ax2.set_title('Zoomed Spectrum ($\\tilde{\\epsilon} \\in [0.0, 2.0]$, $\\lambda \\in [-25, 15]$)', pad=10)
    ax2.grid(True, alpha=0.25)
    
    # Adjust layout to make both panels slim and elegant
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)
    
    fig_save_path = SCRIPT_DIR.parent / OUTPUT_FILENAME
    plt.savefig(fig_save_path, dpi=250, bbox_inches='tight')
    
    # Also save as PDF
    pdf_save_path = fig_save_path.with_suffix('.pdf')
    plt.savefig(pdf_save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated and saved stacked eigenvalues plot to: {fig_save_path} and {pdf_save_path}")

if __name__ == "__main__":
    main()
