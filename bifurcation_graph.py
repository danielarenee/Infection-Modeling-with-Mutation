"""
Bifurcation map for the seasonal SIRC model

For each parameter pair (beta and epsilon), it uses RK4 to go through a 300 year transient
phase and then samples the infected population once a year for 50 years
It finally groups them to plot the period

The output is a 2D color map over the (eps, beta0) parameter space where each
pixel is colored by the period of the attractor.

"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time

#%%

# SIRC MODEL

@njit
def sirc_fast(t, S, I, R, C, mu, alpha, delta, gamma, sigma, beta0, eps):
    """
    SIRC system
    Returns the four derivatives (dS, dI, dR, dC) as a tuple.
    """
    beta = beta0*(1.0 + eps*np.cos(2.0*np.pi*t))
    dSdt = mu*(1.0 - S) - beta*S*I + gamma*C
    dIdt = beta*S*I + sigma*beta*C*I - (mu + alpha)*I
    dRdt = (1.0 - sigma)*beta*C*I + alpha*I - (mu + delta)*R
    dCdt = delta*R - beta*C*I - (mu + gamma)*C
    return dSdt, dIdt, dRdt, dCdt


# RK4 + CLASSIFY
# Simulates the SIRC model at (beta0, eps),
# discards N_transient steps, then sample I for N_sample steps

@njit
def classify_point(beta0, eps, mu, alpha, delta, gamma, sigma,
                   h, N_transient, N_sample):
    """
    Runs the full simulation for a single (beta0, eps) pixel and returns
    the period of the long-run attractor.

        1. Transient phase (N_transient steps = 300 years): just integrates,
           discards everything
        2. Sampling phase (N_sample steps = 50 years): integrates and records I
           once per year
        3. The 50 annual I values are sorted and grouped by proximity, the number
        of distinct groups is the period. Anything above 8 is treated as chaos

    Args:
        beta0, eps         : transmission parameters for this pixel
        mu, alpha, delta,
        gamma, sigma       : fixed biological parameters
        h                  : RK4 step size (1/365)
        N_transient        : number of steps to discard
        N_sample           : number of steps to sample over

    Returns:
        groups : integer in [1, 8] representing the period (8 = chaos)
    """

    # initial condition near endemic equilibrium
    S = 0.2
    I = 0.001
    R = 0.499
    C = 0.3
    t = 0.0

    # transient (300 yrs of simulation)
    for n in range(N_transient):
        # RK4 step
        k1S, k1I, k1R, k1C = sirc_fast(t, S, I, R, C,
            mu, alpha, delta, gamma, sigma, beta0, eps)
        k2S, k2I, k2R, k2C = sirc_fast(t + h/2,
            S + h/2*k1S, I + h/2*k1I, R + h/2*k1R, C + h/2*k1C,
            mu, alpha, delta, gamma, sigma, beta0, eps)
        k3S, k3I, k3R, k3C = sirc_fast(t + h/2,
            S + h/2*k2S, I + h/2*k2I, R + h/2*k2R, C + h/2*k2C,
            mu, alpha, delta, gamma, sigma, beta0, eps)
        k4S, k4I, k4R, k4C = sirc_fast(t + h,
            S + h*k3S, I + h*k3I, R + h*k3R, C + h*k3C,
            mu, alpha, delta, gamma, sigma, beta0, eps)
        S += h/6 * (k1S + 2*k2S + 2*k3S + k4S)
        I += h/6 * (k1I + 2*k2I + 2*k3I + k4I)
        R += h/6 * (k1R + 2*k2R + 2*k3R + k4R)
        C += h/6 * (k1C + 2*k2C + 2*k3C + k4C)
        t += h

    # now we sample I
    n_years = N_sample // 365
    I_samples = np.empty(n_years)
    step_count = 0
    year_idx = 0

    for n in range(N_sample):
        k1S, k1I, k1R, k1C = sirc_fast(t, S, I, R, C,
            mu, alpha, delta, gamma, sigma, beta0, eps)
        k2S, k2I, k2R, k2C = sirc_fast(t + h/2,
            S + h/2*k1S, I + h/2*k1I, R + h/2*k1R, C + h/2*k1C,
            mu, alpha, delta, gamma, sigma, beta0, eps)
        k3S, k3I, k3R, k3C = sirc_fast(t + h/2,
            S + h/2*k2S, I + h/2*k2I, R + h/2*k2R, C + h/2*k2C,
            mu, alpha, delta, gamma, sigma, beta0, eps)
        k4S, k4I, k4R, k4C = sirc_fast(t + h,
            S + h*k3S, I + h*k3I, R + h*k3R, C + h*k3C,
            mu, alpha, delta, gamma, sigma, beta0, eps)
        S += h/6 * (k1S + 2*k2S + 2*k3S + k4S)
        I += h/6 * (k1I + 2*k2I + 2*k3I + k4I)
        R += h/6 * (k1R + 2*k2R + 2*k3R + k4R)
        C += h/6 * (k1C + 2*k2C + 2*k3C + k4C)
        t += h
        step_count += 1

        # at every full year, record I
        if step_count == 365:
            I_samples[year_idx] = I
            year_idx += 1
            step_count = 0

    # I_samples will now have the value of I at the end of each year

    # finally we classify (count distinct values)
    sorted_I = np.sort(I_samples)
    tol = 1e-5
    groups = 1
    for k in range(1, len(sorted_I)):
        if abs(sorted_I[k] - sorted_I[k-1]) > tol + tol * abs(sorted_I[k]):
            groups += 1

    if groups > 8:
        groups = 8

    return groups

# PARAMETERS
mu    = 0.02
alpha = 365.0 / 3.0
delta = 1.0 / 1.61
gamma = 0.35
sigma = 0.07874

h = 1.0 / 365.0
T_transient = 300 # we want 300 years (transient)
T_sample = 50 # we want 50 yrs per pixel
N_transient = T_transient * 365 # 109500 RK4 steps
N_sample = T_sample * 365 # total RK4 steps per pixel

# GRID
n_eps  = 300
n_beta = 200
eps_vals  = np.linspace(0.001, 0.35, n_eps)
beta_vals = np.linspace(100, 2000, n_beta)


# COMPILE

# (the first call triggers numba compilation)
print("Compiling numba functions...")
_ = classify_point(600.0, 0.1, mu, alpha, delta, gamma, sigma,
                   h, 1000, 1000)
print("done!\n")

# COMPUTE
# period_map[i, j] = period at (beta_vals[i], eps_vals[j])
period_map = np.zeros((n_beta, n_eps))
total = n_beta*n_eps

print(f"Grid: {n_beta} x {n_eps} = {total} simulations")
print(f"Each: {T_transient} yr transient + {T_sample} yr sampling")

start = time.time()

for i in range(n_beta):
    for j in range(n_eps):
        period_map[i, j] = classify_point(
            beta_vals[i], eps_vals[j],
            mu, alpha, delta, gamma, sigma,
            h, N_transient, N_sample
        )

    # progress
    elapsed = time.time() - start
    done = (i+1)*n_eps # number of pixels done
    rate = done/elapsed
    remaining = (total-done)/rate # estimated seconds remaining
    print(f"  row {i+1}/{n_beta} | beta0 = {beta_vals[i]:.0f} | "
          f"{elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining")

total_time = time.time() - start
print(f"\nFinished in {total_time:.1f}s ({total_time/60:.1f} min)")

# PLOT

from matplotlib.colors import ListedColormap, BoundaryNorm

colors = ['#FFFFFF', # period 1 white
          '#AED6F1', # period 2 light blue
          '#3498DB', # period 3 blue
          '#E74C3C', # period 4 red
          '#A0A0A0', # period 5 gray
          '#707070', # period 6 dark gray
          '#404040', # period 7 darker
          '#1A1A1A'] # period 8+ chaos (black)

cmap = ListedColormap(colors)
bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
norm = BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(11, 7))
im = ax.pcolormesh(eps_vals, beta_vals, period_map,
                   cmap=cmap, norm=norm, shading='nearest')

# line at Beta_0= mu+alpha (below this line R_=<1 and the disease dies out)
ax.axhline(y=(mu + alpha), color='k', linestyle=':', linewidth=1, label='R₀ = 1')

ax.set_xlabel('Degree of seasonality ε', fontsize=13)
ax.set_ylabel('Baseline rate of transmission β₀', fontsize=13)
ax.set_title('Bifurcation map of the SIRC model', fontsize=14)

cbar = fig.colorbar(im, ax=ax, ticks=[1, 2, 3, 4, 5, 6, 7, 8])
cbar.set_ticklabels(['Period 1', 'Period 2', 'Period 3', 'Period 4',
                      'P5', 'P6', 'P7', 'Chaos'])

ax.set_xlim(0, 0.35)
ax.set_ylim(100, 2000)
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig('bifurcation_map_SIRC.png', dpi=300)
plt.show()
plt.close()

#%%
# single test

group = classify_point(
    500, 0.1,
    mu, alpha, delta, gamma, sigma,
    h, N_transient, N_sample)

print(group)

#%%