"""
Script to run SIRCmw simulation at a specific coordinate to test whether
(relative_eps1 = 0.2, relative_eps2 = 1.0) is stable or unstable,
using the project's native plot_sircmw_timeseries function.
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import numpy as np

# Add workspace directory to path
WORKSPACE_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(WORKSPACE_DIR))

from SIRCmw.sircmw_utils import plot_sircmw_timeseries, get_endemic_roots, SI_0, SIGMA, MU, ALPHA, DELTA, GAMMA

beta0 = 600.0
relative_eps1 = 1
relative_eps2 = 0.5

# Compute physical parameters
eps1 = relative_eps1 / SI_0
eps2 = relative_eps2 / SI_0

y0 = np.array([0.2, 0.001, 0.499, 0.3])

p = {
    'beta0': beta0,
    'sigma': SIGMA,
    'mu': MU,
    'alpha': ALPHA,
    'delta': DELTA,
    'gamma': GAMMA,
    'eps1': eps1,
    'eps2': eps2
}

# Run native plot timeseries function
save_path = Path(__file__).parent / "diagnostic_timeseries.png"
plot_sircmw_timeseries(y0=y0, p=p, years=100, save_path=save_path, show=False)
print(f"Timeseries plot saved to {save_path}")
