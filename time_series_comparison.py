"""
Compares the original SIRC model against SIRCm where the cross-immunity
erosion rates (delta, gamma) are driven by current infection incidence
rather than fixed constants.

Both models are solved via the custom RK4 solver over 300 years. Comparisons
are made under tropical (Singapore) and temperate (England) parameter regimes
(example from Casagrandi's paper with and without seasonal forcing.

Notes to self:
- solve_ivp (RK45, Radau) failed
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

#%%

# ─────────────────────────────────────────────
# RK4 SOLVER
# ─────────────────────────────────────────────

def step(f, t_n, y_n, h, params):
    """
    Single RK4 step

    Args:
        f      : ODE right-hand side
        t_n    : current time
        y_n    : current state vector [S, I, R, C]
        h      : step size (in years)
        params : dict of model parameters

    Returns:
        y_{n+1} : updated state vector
    """
    k1 = h * f(t_n,       y_n,          params)
    k2 = h * f(t_n + h/2, y_n + k1/2,  params)
    k3 = h * f(t_n + h/2, y_n + k2/2,  params)
    k4 = h * f(t_n + h,   y_n + k3,    params)
    return y_n + (1/6) * (k1 + 2*k2 + 2*k3 + k4)


def solve(f, y0, t_span, h, params):
    """
    Integrates the ODE system using fixed-step RK4

    Args:
        f      : ODE right-hand side,
        y0     : initial state vector [S0, I0, R0, C0]
        t_span : (t_start, t_end) in years
        h      : step size (1/365 = daily resolution)
        params : dict of model parameters

    Returns:
        t_arr : 1D array of time points
        y_arr : 2D array of shape (N_steps+1, 4), columns are S,I,R,C
    """
    t_start, t_end = t_span
    N_steps = int(np.round((t_end - t_start) / h))
    t_arr = np.linspace(t_start, t_end, N_steps + 1)
    y_arr = np.empty((N_steps + 1, len(y0)))
    y_arr[0] = y0
    for n in range(N_steps):
        y_arr[n+1] = step(f, t_arr[n], y_arr[n], h, params)
    return t_arr, y_arr


# ─────────────────────────────────────────────
# SIRC MODELS
# ─────────────────────────────────────────────

def sirc(t, y, params):
    """
    Original SIRC model from Casagrandi

    Compartments:
        S : fully susceptible
        I : infected
        R : fully immune (recovered)
        C : cross-immune (partially susceptible to a new strain)

    Parameters:
        mu    : birth/death rate
        alpha : recovery rate from infection
        delta : rate at which fully immune (R) become cross-immune (C)
                (fixed)
        gamma : rate at which cross-immune (C) return to susceptible (S)
                (fixed)
        sigma : reduced susceptibility of cross-immunes
        beta0 : baseline transmission rate
        eps   : amplitude of seasonal forcing (0 = no seasonality)
        (degree of seasonality)
    """
    S, I, R, C = y
    mu    = params['mu']
    alpha = params['alpha']
    delta = params['delta']
    gamma = params['gamma']
    sigma = params['sigma']
    beta0 = params['beta0']
    eps   = params.get('eps', 0)

    beta = beta0 * (1 + eps * np.cos(2 * np.pi * t))

    dSdt = mu*(1 - S) - beta*S*I + gamma*C
    dIdt = beta*S*I + sigma*beta*C*I - (mu + alpha)*I
    dRdt = (1 - sigma)*beta*C*I + alpha*I - (mu + delta)*R
    dCdt = delta*R - beta*C*I - (mu + gamma)*C
    return np.array([dSdt, dIdt, dRdt, dCdt])


def sirc_modified(t, y, params):
    """
    SIRCm with infection-driven cross-immunity erosion

    Delta and gamma are no longer fixed rates. Instead,
    they scale with current infection incidence (beta * S * I).

    The idea is that each new infection event is a potential mutation,
    so the rate at which the virus escapes immunity (R -> C) and
    the rate of re-susceptibility (C -> S) should increase when more
    infections are occurring.

        delta_final = delta_prime * beta * S * I
        gamma_final = gamma_prime * beta * S * I

    where delta_prime and gamma_prime are calibrated so that at the
    endemic equilibrium of the original model, the effective rates
    match the original delta and gamma.

    """
    S, I, R, C = y
    mu          = params['mu']
    alpha       = params['alpha']
    sigma       = params['sigma']
    beta0       = params['beta0']
    eps         = params.get('eps', 0)
    delta_prime = params['delta_prime']
    gamma_prime = params['gamma_prime']

    beta = beta0 * (1 + eps * np.cos(2 * np.pi * t))

    delta_final = delta_prime * beta * S * I
    gamma_final = gamma_prime * beta * S * I

    dSdt = mu*(1 - S) - beta*S*I + gamma_final*C
    dIdt = beta*S*I + sigma*beta*C*I - (mu + alpha)*I
    dRdt = (1 - sigma)*beta*C*I + alpha*I - (mu + delta_final)*R
    dCdt = delta_final*R - beta*C*I - (mu + gamma_final)*C
    return np.array([dSdt, dIdt, dRdt, dCdt])

#%%
# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────
# biological parameters from Casagrandi's paper

mu    = 0.02        # birth/death rate (1/year)
alpha = 365/3       # recovery rate (~3 days infectious period)
delta = 1/1.61      # R -> C transition rate
gamma = 0.35        # C -> S transition rate
sigma = 0.07874     # cross-immunity factor (C individuals ~8x less susceptible)

# Tropical regime (Singapore): high beta, weak seasonality
beta0_trop = 1200
eps_trop   = 0.07

# Temperate regime (England): lower beta, stronger seasonality
beta0_temp = 400
eps_temp   = 0.18

epsnon = 0 # no seasonality (for baseline comparison)

# ─────────────────────────────────────────────
# CALIBRATION OF delta_prime AND gamma_prime
# ─────────────────────────────────────────────
# We want the modified model to match the original at the endemic equilibrium.
# so we find the endemic equilibrium of the original model at beta0=600,
# compute the incidence there, then set delta_prime = delta / incidence_eq
# so that delta_prime * incidence_eq = delta exactly at equilibrium.

calib_params = {'mu': mu, 'alpha': alpha, 'delta': delta, 'gamma': gamma,
                'sigma': sigma, 'beta0': 600, 'eps': 0}

def equilibrium_equations(x):
    return sirc(0, x, calib_params)

# Solve for the endemic equilibrium using Newton's method (fsolve)
eq = fsolve(equilibrium_equations, np.array([0.2, 0.001, 0.499, 0.3]))
S_eq, I_eq, R_eq, C_eq = eq

incidence_eq = 600 * S_eq * I_eq  # beta * S * I at equilibrium

# Calibrated rate coefficients
delta_prime = delta / incidence_eq
gamma_prime = gamma / incidence_eq

#%%

# ─────────────────────────────────────────────
# TROPICAL SIMULATIONS
# ─────────────────────────────────────────────
# Run all four variants: original/modified x seasonal/non-seasonal
# We simulate for 300 years and inspect only the last x years (t > 300-x)
# (focusing on the attractor)

tropical_nonseasonal_params = {
    'mu': mu, 'alpha': alpha, 'delta': delta,
    'gamma': gamma, 'sigma': sigma,
    'beta0': 1200, 'eps': epsnon,
}

tropical_params = {
    'mu': mu, 'alpha': alpha, 'delta': delta, 'gamma': gamma,
    'sigma': sigma, 'beta0': 1200, 'eps': 0.07,
}

tropical_mod_nonseasonal_params = {
    'mu': mu, 'alpha': alpha, 'sigma': sigma,
    'beta0': 1200, 'eps': epsnon,
    'delta_prime': delta_prime, 'gamma_prime': gamma_prime,
}

tropical_mod_params = {
    'mu': mu, 'alpha': alpha, 'sigma': sigma,
    'beta0': 1200, 'eps': 0.07,
    'delta_prime': delta_prime, 'gamma_prime': gamma_prime,
}

y0 = np.array([0.2, 0.001, 0.499, 0.3])

t_trop, y_trop = solve(sirc, y0, (0, 300), 1/365, tropical_params)
t_trop_nonseasonal, y_trop_nonseasonal  = solve(sirc, y0, (0, 300), 1/365, tropical_nonseasonal_params)
t_trop_m, y_trop_m = solve(sirc_modified, y0, (0, 300), 1/365, tropical_mod_params)
t_trop_m_nonseasonal, y_trop_m_nonseasonal = solve(sirc_modified, y0, (0, 300), 1/365, tropical_mod_nonseasonal_params)

mask = t_trop > 280  # last x years of the 300-year run

fig, ax = plt.subplots(figsize=(10, 9))
# Time is shifted to [0, 12] months for readability
ax.plot((t_trop_nonseasonal[mask] - 280) * 12, y_trop_nonseasonal[mask, 1], 'b--', linewidth=1.5, label='Original SIRC')
ax.plot((t_trop[mask] - 280) * 12, y_trop[mask, 1], 'b-',  linewidth=1.5, label='Original SIRC (seasonally adjusted)')
ax.plot((t_trop_m_nonseasonal[mask] - 280) * 12, y_trop_m_nonseasonal[mask, 1], 'r--', linewidth=1.5, label='Modified SIRC')
ax.plot((t_trop_m[mask] - 280) * 12, y_trop_m[mask, 1], 'r-',  linewidth=1.5, label='Modified SIRC (seasonally adjusted)')
ax.set_xlabel('Time (months)', fontsize=12)
ax.set_ylabel('Prevalence I(t)', fontsize=12)
ax.set_title('Tropical (ε = 0.07, β₀ = 1200)', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
#ax.set_ylim(0, 0.0035)
plt.tight_layout()
plt.show()
plt.close()
print("done")

#%%

# ─────────────────────────────────────────────
# TEMPERATE SIMULATIONS
# ─────────────────────────────────────────────
# Same structure

temperate_nonseasonal_params = {
    'mu': mu, 'alpha': alpha, 'delta': delta,
    'gamma': gamma, 'sigma': sigma,
    'beta0': beta0_temp, 'eps': epsnon,
}

temperate_params = {
    'mu': mu, 'alpha': alpha, 'delta': delta,
    'gamma': gamma, 'sigma': sigma,
    'beta0': beta0_temp, 'eps': eps_temp,
}

temperate_mod_params = {
    'mu': mu, 'alpha': alpha, 'sigma': sigma,
    'beta0': beta0_temp, 'eps': eps_temp,
    'delta_prime': delta_prime, 'gamma_prime': gamma_prime,
}

t_temp_nonseasonal, y_temp_nonseasonal = solve(sirc, y0, (0, 300), 1/365, temperate_nonseasonal_params)
t_temp, y_temp = solve(sirc, y0, (0, 300), 1/365, temperate_params)
t_temp_m, y_temp_m = solve(sirc_modified, y0, (0, 300), 1/365, temperate_mod_params)

mask = t_temp > 290  # last 10 years

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_temp_nonseasonal[mask] - 290, y_temp_nonseasonal[mask, 1], 'b--', linewidth=1, label='Original SIRC')
ax.plot(t_temp[mask] - 290, y_temp[mask, 1], 'k-',  linewidth=1, label='Original SIRC (seasonally adjusted)')
ax.plot(t_temp_m[mask] - 290, y_temp_m[mask, 1], 'r--', linewidth=1, label='Modified SIRC (seasonally adjusted)')
ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('Prevalence I(t)', fontsize=12)
ax.set_title('Temperate (ε = 0.18, β₀ = 400)', fontsize=13)
ax.set_ylim(0, 0.07)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()
print("done")


# ─────────────────────────────────────────────
# FULL TIME SERIES COMPARISON (no seasonality)
# ─────────────────────────────────────────────
# Run both models for 50 years and plot the full transient + long-run behavior
# Shows all four compartments

y0 = np.array([0.2, 0.001, 0.499, 0.3])
h = 1/365

orig_params = {
    'mu': mu, 'alpha': alpha, 'delta': delta, 'gamma': gamma,
    'sigma': sigma, 'beta0': 1200,
}

mod_params = {
    'mu': mu, 'alpha': alpha, 'sigma': sigma,
    'beta0': 1200, 'delta_prime': delta_prime, 'gamma_prime': gamma_prime,
}

t_orig, y_orig = solve(sirc,          y0, (0, 50), h, orig_params)
t_mod,  y_mod  = solve(sirc_modified, y0, (0, 50), h, mod_params)

# --- I(t) and C(t) ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(t_orig, y_orig[:, 1], 'b-',  linewidth=1, label='Original SIRC')
axes[0].plot(t_mod,  y_mod[:, 1],  'r--', linewidth=1, label='Modified SIRC')
axes[0].set_ylabel('Prevalence I(t)', fontsize=12)
axes[0].set_title('Original vs Modified SIRC (no seasonality)', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_orig, y_orig[:, 3], 'b-',  linewidth=1, label='C (original)')
axes[1].plot(t_mod,  y_mod[:, 3],  'r--', linewidth=1, label='C (modified)')
axes[1].set_xlabel('Time (years)', fontsize=12)
axes[1].set_ylabel('Cross-immune fraction', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()

# --- S(t) and R(t) ---
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(t_orig, y_orig[:, 0], 'b-',  linewidth=1, label='Original SIRC')
axes[0].plot(t_mod,  y_mod[:, 0],  'r--', linewidth=1, label='Modified SIRC')
axes[0].set_ylabel('Susceptible fraction', fontsize=12)
axes[0].set_title('Original vs Modified SIRC (no seasonality)', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_orig, y_orig[:, 2], 'b-',  linewidth=1, label='R (original)')
axes[1].plot(t_mod,  y_mod[:, 2],  'r--', linewidth=1, label='R (modified)')
axes[1].set_xlabel('Time (years)', fontsize=12)
axes[1].set_ylabel('Recovered fraction', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()