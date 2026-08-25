"""
Computes the endemic equilibrium and stability of the SIRC model with I-feedback for a grid of parameters eps1 and eps2 
using a logarithmic scale + 0 to cover several orders of magnitude.
Solving for the reduced equation in I and evaluating stability using the Jacobian.
"""

import sys
import numpy as np
from pathlib import Path

# Add SIRCmw_I root directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sircmw_I_utils import (
    SI_0,
    sircmw_jacobian,
    get_endemic_roots
)

# contact rate (consistent with log_scale sweep = 600)
beta0 = 700
scale_factor = SI_0 # = 0.001 for I-feedback model

def run_one_root(relative_eps1, relative_eps2):
    """Computes endemic prevalence and stability for a single parameter pair"""
    eps1 = relative_eps1 / scale_factor
    eps2 = relative_eps2 / scale_factor
    
    roots = get_endemic_roots(eps1, eps2, beta0)
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

print("Running algebraic endemic equilibrium sweep for I-feedback model on log scale centered at 0...")

# Generate log-spaced grid prepended with 0
list_eps1 = np.concatenate(([0], np.logspace(-3, 4, 100)))
list_eps2 = np.concatenate(([0], np.logspace(-3, 4, 100)))

# sweeeeeeep
results = [run_one_root(e1, e2) for e1 in list_eps1 for e2 in list_eps2]

prevalences = np.array([r[0] for r in results]).reshape(len(list_eps1), len(list_eps2))
stabilities = np.array([r[1] for r in results]).reshape(len(list_eps1), len(list_eps2))

# save results to file
out_path = Path(__file__).parent / "prevalence_roots_results.npz"
np.savez(out_path, list_eps1=list_eps1, list_eps2=list_eps2,
         prevalences=prevalences, stabilities=stabilities)
print(f"Results saved to {out_path.name}")
