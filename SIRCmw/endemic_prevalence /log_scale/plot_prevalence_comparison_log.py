import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import brentq

# Resolve paths and import shared utilities
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent.parent.parent
sys.path.append(str(WORKSPACE_DIR / "SIRCmw"))

import sircmw_utils

# Model parameters (aligned with plot_prevalence_comparison.jl)
MU = 0.02
ALPHA = 365.0 / 3.0
DELTA = 1.0 / 1.61
GAMMA = 0.35
SIGMA = 0.07874
SI_0 = 0.0002045  # Correct scaling factor defined in sircmw_utils

p_base = {
    'mu': MU,
    'alpha': ALPHA,
    'delta': DELTA,
    'gamma': GAMMA,
    'sigma': SIGMA,
    'si_0': SI_0
}

# Values of tilde_eps to analyze (SIRC baseline, 0.1, 0.2, 0.3, 0.4, and 0.5 to 2.0)
tilde_eps_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

# Continuation parameters
BETA0_MIN = 0.0
BETA0_MAX = 1500.0
TRANSCRITICAL_BETA0 = MU + ALPHA  # ~121.6867

def get_max_real_eigenvalue(beta0, tilde_eps):
    """Computes the maximum real part of the eigenvalues of the Jacobian at the endemic equilibrium."""
    eps = tilde_eps / SI_0
    coeffs = sircmw_utils.poly_coeffs(beta0, MU, ALPHA, GAMMA, DELTA, eps, SIGMA)
    roots = np.polynomial.polynomial.polyroots(coeffs)
    
    # Filter real roots in (0, 1]
    valid_I = [r.real for r in roots if abs(r.imag) < 1e-6 and 0.0 < r.real <= 1.0]
    if not valid_I:
        return None
        
    # Reconstruct the equilibrium
    eq = sircmw_utils.recover_equilibrium(valid_I[0], eps, p_base)
    if eq is None:
        return None
        
    # Evaluate Jacobian and eigenvalues
    J = sircmw_utils.sircmw_jacobian(eq, eps, eps, p_base)
    eigs = np.linalg.eigvals(J)
    return np.max(np.real(eigs))

def find_hopf_point(tilde_eps, beta_start, beta_end):
    """Finds the precise Hopf bifurcation point between beta_start and beta_end."""
    def objective(b):
        val = get_max_real_eigenvalue(b, tilde_eps)
        return val if val is not None else 0.0
        
    try:
        beta_hopf = brentq(objective, beta_start, beta_end)
        eps = tilde_eps / SI_0
        coeffs = sircmw_utils.poly_coeffs(beta_hopf, MU, ALPHA, GAMMA, DELTA, eps, SIGMA)
        roots = np.polynomial.polynomial.polyroots(coeffs)
        valid_I = [r.real for r in roots if abs(r.imag) < 1e-6 and 0.0 < r.real <= 1.0]
        if valid_I:
            I_star = valid_I[0]
            eq = sircmw_utils.recover_equilibrium(I_star, eps, p_base)
            if eq is not None:
                J = sircmw_utils.sircmw_jacobian(eq, eps, eps, p_base)
                eigs = np.linalg.eigvals(J)
                max_idx = np.argmax(np.real(eigs))
                # Check that the eigenvalue crossing the axis has non-zero imaginary part
                if abs(eigs[max_idx].imag) > 1e-4:
                    return beta_hopf, I_star
    except ValueError:
        pass
    return None

def compute_data():
    """Computes equilibrium branches and stability for all tilde_eps values."""
    print("Computing endemic equilibrium branches...")
    data = {}
    
    # 1500 points for a smooth curves plot (step size of 1.0)
    beta_grid = np.linspace(BETA0_MIN, BETA0_MAX, 1500)
    
    for te in tilde_eps_vals:
        print(f"  · Processing ε̃ = {te}...")
        eps = te / SI_0
        
        beta_vals = []
        I_vals = []
        stability = []
        hopf_candidates = []
        
        # Traverse the beta0 grid
        for b in beta_grid:
            if b < TRANSCRITICAL_BETA0:
                # Below transcritical bifurcation, state is the stable DFE (I* = 0)
                beta_vals.append(b)
                I_vals.append(0.0)
                stability.append(True)
            else:
                coeffs = sircmw_utils.poly_coeffs(b, MU, ALPHA, GAMMA, DELTA, eps, SIGMA)
                roots = np.polynomial.polynomial.polyroots(coeffs)
                valid_I = [r.real for r in roots if abs(r.imag) < 1e-6 and 0.0 < r.real <= 1.0]
                
                if not valid_I:
                    # Fallback to stable DFE if no endemic root is found
                    beta_vals.append(b)
                    I_vals.append(0.0)
                    stability.append(True)
                else:
                    # Reconstruct and check stability
                    I_star = valid_I[0]
                    eq = sircmw_utils.recover_equilibrium(I_star, eps, p_base)
                    if eq is None:
                        beta_vals.append(b)
                        I_vals.append(0.0)
                        stability.append(True)
                    else:
                        J = sircmw_utils.sircmw_jacobian(eq, eps, eps, p_base)
                        eigs = np.linalg.eigvals(J)
                        max_real = np.max(np.real(eigs))
                        is_stable = max_real < 0.0
                        
                        beta_vals.append(b)
                        I_vals.append(I_star)
                        stability.append(is_stable)
                        
                        # Store max_real to detect sign change crossings
                        hopf_candidates.append((b, max_real))
        
        # Detect exact Hopf bifurcation points where max_real changes sign
        hopf_points = []
        for i in range(len(hopf_candidates) - 1):
            b1, r1 = hopf_candidates[i]
            b2, r2 = hopf_candidates[i+1]
            if r1 * r2 < 0.0:
                # Stability switch detected, search for exact crossing
                hp = find_hopf_point(te, b1, b2)
                if hp is not None:
                    hopf_points.append(hp)
                    
        data[te] = {
            'beta_vals': beta_vals,
            'I_vals': I_vals,
            'stability': stability,
            'hopf_points': hopf_points
        }
        
    return data

def plot_branch_segments(ax, beta_vals, I_vals, stability, color, label):
    """Plots stable (solid) and unstable (dashed) segments of a branch seamlessly."""
    n = len(beta_vals)
    if n == 0:
        return
        
    current_segment_beta = [beta_vals[0]]
    current_segment_I = [I_vals[0]]
    current_stability = stability[0]
    
    first_stable_label = False
    
    for i in range(1, n):
        if stability[i] != current_stability:
            # Add boundary transition point to prevent gaps between segments
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
    cache_path = SCRIPT_DIR / "prevalence_comparison_data.pkl"
    
    # Load from cache if it exists and matches our parameters, otherwise compute
    if cache_path.exists():
        print(f"Loading cached bifurcation data from: {cache_path.name}")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if not all(te in data for te in tilde_eps_vals):
            print("Cached data is missing requested tilde_eps values. Recomputing...")
            data = compute_data()
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            print(f"Bifurcation data cached to: {cache_path.name}")
    else:
        data = compute_data()
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Bifurcation data cached to: {cache_path.name}")
        
    # --- PLOTTING ---
    print("Generating plot...")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Arial'],
        'mathtext.fontset': 'dejavusans'
    })
    
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_facecolor('#fafafa')
    
    # Use the tab20 color palette to get 12 distinct colors
    colors = [plt.cm.tab20(i) for i in range(len(tilde_eps_vals))]
    
    # R0 = 1 indicator line
    ax.axvline(TRANSCRITICAL_BETA0, color='gray', linestyle=':', lw=1.5, alpha=0.7, label=r'$R_0 = 1$')
    
    # Plot curves
    for te, color in zip(tilde_eps_vals, colors):
        branch = data[te]
        label = r"$\tilde{\epsilon} = 0.0$ (SIRC)" if te == 0.0 else rf"$\tilde{{\epsilon}} = {te}$"
        
        # Plot branch segments
        plot_branch_segments(ax, branch['beta_vals'], branch['I_vals'], branch['stability'], color, label)
        
        # Plot Hopf bifurcation points
        for hx, hy in branch['hopf_points']:
            ax.scatter(hx, hy, color='green', marker='D', s=35, edgecolor='black', zorder=5)
            
    # Add dummy entries for legend classifications
    ax.plot([], [], color='black', linestyle='-', lw=2.2, label='Stable equilibrium')
    ax.plot([], [], color='black', linestyle='--', lw=2.2, label='Unstable equilibrium')
    ax.scatter([], [], color='green', marker='D', s=35, edgecolor='black', label='Hopf point')
    
    # Formatting
    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlim(BETA0_MIN, BETA0_MAX)
    ax.set_ylim(1e-6, 1.0)
    
    ax.set_xlabel(r"Contact rate $\beta_0$", fontsize=14, labelpad=8)
    ax.set_ylabel(r"Infected fraction $I^*$", fontsize=14, labelpad=8)
    # Title removed per user request
    
    ax.grid(True, which="both", linestyle="--", color="gray", alpha=0.25)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Place legend outside or neatly inside
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, frameon=True, facecolor='white', edgecolor='#ddd')
    
    plt.tight_layout()
    
    # Save plots
    png_out = SCRIPT_DIR / "prevalence_comparison_log.png"
    pdf_out = SCRIPT_DIR / "prevalence_comparison_log.pdf"
    
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_out, bbox_inches='tight')
    
    print(f"Plot saved to: {png_out.name} and {pdf_out.name}")
    plt.close()

if __name__ == '__main__':
    main()
