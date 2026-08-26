#!/usr/bin/env python3
"""
SIRC vs. SIRCmw endemic prevalence comparison for tilde_eps from 0.0 to 0.5
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib

# Use Agg backend for non-interactive saving
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to sys.path to find sircmw_I_utils
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

from sircmw_I_utils import (
    sircmw_jacobian,
    get_algebraic_equilibria,
    MU as mu,
    ALPHA as alpha,
    DELTA as delta,
    GAMMA as gamma,
    SIGMA as sigma,
    BETA0 as beta0
)

scale_factor = 0.001
transcritical_beta0 = mu + alpha

# Plot limits
BETA0_MIN, BETA0_MAX = 0.0, 1500.0

def run_stability_sweep(tilde_eps):
    beta_grid = np.linspace(BETA0_MIN, BETA0_MAX, 1000)
    
    beta_stable, I_stable = [], []
    beta_unstable, I_unstable = [], []
    hopf_points = []
    
    was_stable = None
    
    for b in beta_grid:
        if b <= transcritical_beta0:
            beta_stable.append(b)
            I_stable.append(0.0)
            beta_unstable.append(b)
            I_unstable.append(np.nan)
            was_stable = True
        else:
            p = {
                'beta0': b,
                'sigma': sigma,
                'mu': mu,
                'alpha': alpha,
                'delta': delta,
                'gamma': gamma,
                'si_0': scale_factor
            }
            eqs = get_algebraic_equilibria(tilde_eps, p)
            if not eqs:
                beta_stable.append(b)
                I_stable.append(0.0)
                beta_unstable.append(b)
                I_unstable.append(np.nan)
                was_stable = True
            else:
                eq = max(eqs, key=lambda u: u[1])
                I_star = eq[1]
                
                # Check stability
                eps = tilde_eps / scale_factor
                J = sircmw_jacobian(eq, eps, p=p)
                eigs = np.linalg.eigvals(J)
                max_real = np.max(np.real(eigs))
                is_stable = (max_real < 0.0)
                
                if was_stable is not None and is_stable != was_stable:
                    hopf_points.append((b, I_star))
                
                was_stable = is_stable
                
                if is_stable:
                    beta_stable.append(b)
                    I_stable.append(I_star)
                    beta_unstable.append(b)
                    I_unstable.append(np.nan)
                else:
                    beta_stable.append(b)
                    I_stable.append(np.nan)
                    beta_unstable.append(b)
                    I_unstable.append(I_star)
                    
    return (np.array(beta_stable), np.array(I_stable), 
            np.array(beta_unstable), np.array(I_unstable), 
            hopf_points)

def main():
    print("Generating SIRCmw Prevalence Comparison Sweep Plot (tilde_eps = 0.0 to 0.5)...")
    tilde_eps_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Clean color palette
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(tilde_eps_vals))]
    
    plt.figure(figsize=(9, 6), dpi=300)
    plt.axvline(transcritical_beta0, color='gray', linestyle='--', linewidth=1.2, label="$R_0 = 1$ floor")
    
    for i, te in enumerate(tilde_eps_vals):
        print(f"  · Running for ε̃ = {te:.2f}")
        b_s, I_s, b_u, I_u, hopf_pts = run_stability_sweep(te)
        col = colors[i]
        
        lbl = f"$\\tilde{{\\varepsilon}} = {te:.1f}$ (SIRC)" if te == 0.0 else f"$\\tilde{{\\varepsilon}} = {te:.1f}$"
        plt.plot(b_s, I_s, color=col, linewidth=1.3, label=lbl)
        plt.plot(b_u, I_u, color=col, linewidth=1.3, linestyle='--', label=None)
        
        for hx, hy in hopf_pts:
            plt.scatter(hx, hy, color='green', marker='D', s=55, zorder=5)

    plt.xlabel("Contact rate $\\beta_0$", fontsize=12)
    plt.ylabel("Infected fraction $I^*$", fontsize=12)
    plt.xlim(BETA0_MIN, BETA0_MAX)
    plt.ylim(bottom=0.0)
    plt.grid(True, linestyle='-', linewidth=0.5, color='#e5e5e5')
    plt.legend(loc="lower right", fontsize=10, frameon=True, edgecolor='#e5e5e5')
    
    plt.tight_layout()
    
    save_comp_png = SCRIPT_DIR / "sircmw_prevalence_comparison.png"
    plt.savefig(save_comp_png, dpi=300)
    print(f"Saved: {save_comp_png.resolve()}")
    
    save_comp_pdf = SCRIPT_DIR / "sircmw_prevalence_comparison.pdf"
    plt.savefig(save_comp_pdf, dpi=300)
    print(f"Saved: {save_comp_pdf.resolve()}")
    plt.close()

if __name__ == "__main__":
    main()
