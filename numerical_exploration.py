"""
Numerical exploration of the endemic equilibrium of the original SIRC model

For every pair of parameters (15 pairs total from 6 parameters), it sweeps a 2D
grid, finds the endemic equilibrium at each point, builds the Jacobian, and
computes the maximum real part of its eigenvalues.

The result is a 3x5 grid of heatmaps where:
  - Red  -> max real part > 0: equilibrium is unstable
  - Blue -> max real part < 0: equilibrium is stable
  - Gray -> no endemic equilibrium exists (R0 <= 1)
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import fsolve

# FUNCTIONS

# gets the basic reproduction number
def compute_r0(beta, mu, alpha):
    return beta / (mu + alpha)

# solves the system when the derivatives equal 0
def compute_endemic_equilibrium(beta, mu, alpha, delta, gamma, sigma):
    """
    Finds the endemic equilibrium (S*, I*, R*, C*) via Newton's method (fsolve).

    Returns None if:
      - R0 <= 1 (no endemic equilibrium exists by theory)
      - fsolve converges to the disease-free state (I ~ 0)
      - any compartment is negative (no physical solution)
      - compartments don't sum to 1 (conservation violated)
    """
    R0 = compute_r0(beta, mu, alpha)
    if R0 <= 1:
        return None # no endemic equilibrium

    def sirc_rhs(X):
        S, I, R, C = X
        dS = mu*(1-S) - beta*S*I + gamma*C
        dI = beta*S*I + sigma*beta*C*I - (mu+alpha)*I
        dR = (1-sigma)*beta*C*I + alpha*I - (mu+delta)*R
        dC = delta*R - beta*C*I - (mu+gamma)*C
        return [dS, dI, dR, dC]

    guess = [0.2, 0.001, 0.5, 0.3]
    sol = fsolve(sirc_rhs, guess, full_output=True)
    X = sol[0]
    S, I, R, C = X

    # sanity check
    if I <= 1e-10:
        return None
    if S < 0 or R < 0 or C < 0:
        return None
    if abs(S + I + R + C - 1.0) > 1e-6:
        return None

    return S, I, R, C

def compute_jacobian(S, I, R, C, beta, mu, alpha, delta, gamma, sigma):
    """
    Analytical Jacobian (from mathematica) of the SIRC system evaluated at a given point (S,I,R,C)
    """
    J = np.array([
        [-mu - beta * I, -beta * S, 0, gamma],
        [beta * I, beta * S + sigma * beta * C - (mu + alpha), 0, sigma * beta * I],
        [0, (1 - sigma) * beta * C + alpha, -(mu + delta), (1 - sigma) * beta * I],
        [0, -beta * C, delta, -beta * I - (mu + gamma)]
    ])
    return J

def stability_analysis(beta, mu, alpha, delta, gamma, sigma):
    """
    for a given parameter set:
    - computes the endemic equilibrium
    - builds the Jacobian there
    - return the maximum real part of the eigenvalues
    this because if max is negative, all are and its stable. if max is positive its unstable
    (at least one is positive). Returns np.nan if no endemic equilibrium exists.
    """
    eq = compute_endemic_equilibrium(beta, mu, alpha, delta, gamma, sigma)
    if eq is None:
        return np.nan, None # no endemic eq

    S, I, R, C = eq
    J = compute_jacobian(S, I, R, C, beta, mu, alpha, delta, gamma, sigma)
    eigenvalues = np.linalg.eigvals(J)
    max_real = np.max(np.real(eigenvalues))

    return max_real, eigenvalues

#%%
# HEATMAPS
# Let's make some heatmaps from all combinations of two parameters

# sweeps over 2 parameters and computes maximun real part of eigenvalues
def sweep(param1_name, param1_vals, param2_name, param2_vals, fixed_params):
    """
    2D parameter sweep: for each (p1, p2) pair in the grid, calls
    stability_analysis with all other parameters held at their reference values.

    Returns:
        max_real : 2D array of shape (n2, n1) with the max real eigenvalue
                   at each grid point. NaN where no endemic equilibrium exists.
    """
    n1, n2 = len(param1_vals), len(param2_vals)
    max_real = np.full((n2, n1), np.nan)

    for i, p2 in enumerate(param2_vals):
        for j, p1 in enumerate(param1_vals):
            params = dict(fixed_params)
            params[param1_name] = p1
            params[param2_name] = p2

            mr, eigs = stability_analysis(**params)
            max_real[i, j] = mr

    return max_real #returns array for max real part

N = 300 # resolution of the grid

# fixed reference values (for when we dont sweep)
ref = dict(beta=600, mu=0.02, alpha=365/3, delta=1/1.61, gamma=0.35, sigma=0.07874)

# here are all parameter names, labels, and ranges from table 1 in the Casagrandi paper
param_info = {
    'beta':  (np.linspace(2*(ref['mu']+ref['alpha']), 10*(ref['mu']+ref['alpha']), N), r'$\beta$'),
    # when beta starts at 50, R_0 is aprox. 0.4, which is less than 1
    #'beta': (np.linspace(50, 10 * (ref['mu'] + ref['alpha']), N), r'$\beta$'),
    'sigma': (np.linspace(0.05, 0.2, N), r'$\sigma$'),
    'delta': (np.linspace(0.5, 1.0, N), r'$\delta$'),
    'gamma': (np.linspace(1/5, 1/2, N), r'$\gamma$'),
    'mu': (np.linspace(1/80, 1/40, N), r'$\mu$'),
    'alpha': (np.linspace(365/7, 365/2, N), r'$\alpha$'),
}

# get all 15 pairs
sweep_configs = []
for p1_name, p2_name in combinations(param_info.keys(), 2):
    p1_vals, p1_label = param_info[p1_name]
    p2_vals, p2_label = param_info[p2_name]
    sweep_configs.append({
        'param1_name': p1_name,
        'param1_vals': p1_vals,
        'param1_label': p1_label,
        'param2_name': p2_name,
        'param2_vals': p2_vals,
        'param2_label': p2_label,
        'title': f'{p1_label} vs {p2_label}'
    })

# PLOT
# the plot shows the maximum real part of the eigenvalues
# colormap is diverging (RdBu_r): symmetric around 0, vmin/vmax set per panel

fig, axes = plt.subplots(3, 5, figsize=(30, 16))
axes_flat = axes.flatten()
all_max_real = [] # for the summary

for k, cfg in enumerate(sweep_configs):
    fixed = {key: val for key, val in ref.items()
             if key != cfg['param1_name'] and key != cfg['param2_name']}

    max_real = sweep(
        cfg['param1_name'], cfg['param1_vals'],
        cfg['param2_name'], cfg['param2_vals'],
        fixed
    )
    all_max_real.append(np.nanmax(max_real))

    ax = axes_flat[k]
    vmax = np.nanmax(np.abs(max_real))
    vmin = -vmax

    im = ax.pcolormesh(cfg['param1_vals'], cfg['param2_vals'], max_real,
                        cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')

    # gray out region where no endemic equilibrium exists (R0 <= 1)
    no_endemic = np.where(np.isnan(max_real), 1.0, np.nan)
    ax.pcolormesh(cfg['param1_vals'], cfg['param2_vals'], no_endemic,
                  cmap='Greys', vmin=0, vmax=1, shading='auto', alpha=0.5)

    ax.set_xlabel(cfg['param1_label'], fontsize=10)
    ax.set_ylabel(cfg['param2_label'], fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# SUMMARY (across all sweeps)

print("\nSummary across all sweeps:")
for k, cfg in enumerate(sweep_configs):
    print(f"\n{cfg['title']}:")
    print(f"  Max real part in the grid: {all_max_real[k]:.6f}")