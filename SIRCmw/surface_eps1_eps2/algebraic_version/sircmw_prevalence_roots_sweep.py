"""
Computes the endemic equilibrium and stability of the SIRC model for a grid of parameters eps1 and eps2
solving for the reduced equation in I and evaluating stability using the Jacobian
The output is used by surface_epsilon_sircmw_prevalence_roots.ipynb 
"""

import sys
import numpy as np
from pathlib import Path
from scipy.optimize import brentq

sys.path.append(str(Path(__file__).parent.parent.parent))

from sircmw_utils import (
    MU as mu,
    ALPHA as alpha,
    DELTA as delta,
    GAMMA as gamma,
    SIGMA as sigma,
    SI_0,
    sircmw_jacobian
)

# contact rate 
beta0 = 600

def get_C(I, eps2, S0):
    """solves the quadratic equation for C given I"""
    A = -eps2 * gamma * sigma * I
    B = gamma * (1.0 + eps2 * I * S0) + sigma * (beta0 * I + mu)
    D = mu * (1.0 - S0) - beta0 * I * S0
    
    if abs(A) < 1e-14:
        C = -D / B
        return C if (0.0 <= C <= S0 / sigma) else None
    
    disc = B**2 - 4.0 * A * D
    if disc < 0.0:
        return None
    
    C1 = (-B + np.sqrt(disc)) / (2.0 * A)
    C2 = (-B - np.sqrt(disc)) / (2.0 * A)
    
    # check physical constraints (C in [0, S0/sigma] ) to get valid C
    valid_C = []
    if 0.0 <= C1 <= S0 / sigma + 1e-12:
        valid_C.append(C1)
    if 0.0 <= C2 <= S0 / sigma + 1e-12:
        valid_C.append(C2)
        
    if not valid_C:
        return None
    
    # Return the physical root (S >= 0)
    return valid_C[0]

def get_endemic_roots(eps1, eps2):
    """Finds all physical endemic equilibria using 1D algebraic reduction"""
    S0 = (mu + alpha) / beta0
    
    # derivative dC/dt
    def residual(I): 
        C = get_C(I, eps2, S0)
        if C is None:
            return np.nan
        S = S0 - sigma * C  # we calculate S 
        eps1_SI_p1 = eps1 * S * I + 1.0
        # we calculate R
        R = ((1.0 - sigma) * beta0 * C * I + alpha * I) / (mu + eps1_SI_p1 * delta)
        eps2_SI_p1 = eps2 * S * I + 1.0
        dC = eps1_SI_p1 * delta * R - beta0 * C * I - (mu + eps2_SI_p1 * gamma) * C
        return dC # dC must be zero in equilibrium

    # we define a grid of 1000 values to evaluate the residual at each,
    # then scan for sign changes to get roots
    I_grid = np.linspace(1e-10, 1.0, 1000)
    res_vals = [residual(i) for i in I_grid]
    
    roots = []
    for k in range(len(I_grid) - 1):
        i1, i2 = I_grid[k], I_grid[k+1]
        r1, r2 = res_vals[k], res_vals[k+1]
        if np.isnan(r1) or np.isnan(r2):
            continue
        if r1 * r2 <= 0.0:
            try:
                root_I = brentq(residual, i1, i2)
                C_star = get_C(root_I, eps2, S0)
                if C_star is not None:
                    S_star = S0 - sigma * C_star
                    eps1_SI_p1 = eps1 * S_star * root_I + 1.0
                    R_star = ((1.0 - sigma) * beta0 * C_star * root_I + alpha * root_I) / (mu + eps1_SI_p1 * delta)
                    if S_star >= -1e-12 and R_star >= -1e-12 and C_star >= -1e-12:
                        # deduplicate roots
                        if not any(abs(root_I - r[1]) < 1e-6 for r in roots):
                            roots.append((S_star, root_I, R_star, C_star))
            except ValueError:
                pass
    return roots

def run_one_root(relative_eps1, relative_eps2):
    """Computes endemic prevalence and stability for a single parameter pair"""
    eps1 = relative_eps1 / SI_0
    eps2 = relative_eps2 / SI_0
    
    roots = get_endemic_roots(eps1, eps2)
    if not roots:
        return 0.0, 1.0  # DFE (prevalence = 0, stable = 1.0)
    
    # evaluate stability for each root
    stable_roots = []
    unstable_roots = []
    
    # classify stable / unstable
    for S, I, R, C in roots:
        J = sircmw_jacobian((S, I, R, C), eps1, eps2, p={'beta0': beta0})
        max_real = np.max(np.real(np.linalg.eigvals(J)))
        if max_real < 0.0:
            stable_roots.append((I, 1.0))  # 1.0 = stable
        else:
            unstable_roots.append((I, 0.0))  # 0.0 = unstable
            
    # prefer stable root if it exists
    if stable_roots:
        return max(stable_roots, key=lambda x: x[0])
    else:
        return max(unstable_roots, key=lambda x: x[0])

print("Running algebraic endemic equilibrium sweep...")
lower_relative_eps = -1.0
upper_relative_eps = 3.0
n_simulations_per_eps = 101

list_eps1 = np.linspace(lower_relative_eps, upper_relative_eps, n_simulations_per_eps)
list_eps2 = np.linspace(lower_relative_eps, upper_relative_eps, n_simulations_per_eps)

# sweeeeeeep
results = [run_one_root(e1, e2) for e1 in list_eps1 for e2 in list_eps2]

prevalences = np.array([r[0] for r in results]).reshape(len(list_eps1), len(list_eps2))
stabilities = np.array([r[1] for r in results]).reshape(len(list_eps1), len(list_eps2))

# save root results to file
out_path = Path(__file__).parent / "prevalence_roots_results.npz"
np.savez(out_path, list_eps1=list_eps1, list_eps2=list_eps2,
         prevalences=prevalences, stabilities=stabilities)
print(f"Results saved to {out_path.name}")

