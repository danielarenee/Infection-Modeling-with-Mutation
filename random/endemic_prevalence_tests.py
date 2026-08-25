import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# SIRC MODELS

def sirc(t, y, params):
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
# PARAMETERS

mu    = 0.02
alpha = 365/3
delta = 1/1.6
gamma = 0.35
sigma = 0.07874

#%%
# CALIBRATION

calib_params = {'mu': mu, 'alpha': alpha, 'delta': delta, 'gamma': gamma,
                'sigma': sigma, 'beta0': 600, 'eps': 0}

def equilibrium_equations(x):
    return sirc(0, x, calib_params)

eq = fsolve(equilibrium_equations, np.array([0.2, 0.001, 0.499, 0.3]))
S_eq, I_eq, R_eq, C_eq = eq
incidence_eq = 600 * S_eq * I_eq
delta_prime = delta / incidence_eq
gamma_prime = gamma / incidence_eq

#%%
# DIAGNOSTIC 1: WHERE EXACTLY DOES THE SOLVER FAIL?
# try different beta values including below the "dead zone" (beta < 183)

#test_betas = [150, 250, 400, 600, 800, 1500]
#test_betas = [50, 100, 150, 160, 170, 180, 190, 200]
#test_betas = [180, 181, 182, 183, 184, 185]
#test_betas = [190, 200, 250, 300, 350, 400, 600, 800, 1000, 1200, 1400, 16000]
test_betas = [3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000]

#test_betas = [2000, 2100, 2200, 2300, 2500,2600]


y0 = np.array([0.2, 0.001, 0.499, 0.3])

print("DIAGNOSTIC 1:")

for beta in test_betas:
    pm = {'mu': mu, 'alpha': alpha, 'sigma': sigma,
          'beta0': beta, 'eps': 0,
          'delta_prime': delta_prime, 'gamma_prime': gamma_prime}

    # run for just 50 years
    sol = solve_ivp(lambda t, y: sirc_modified(t, y, pm),
                    (0, 50), y0, method='BDF',
                    rtol=1e-11, atol=1e-11, max_step=0.01)

    # check if it failed, and if so, when
    if sol.status == 0:
        print(f"β={beta:5d}: SUCCESS, "
              f"final I = {sol.y[1, -1]:.2e}")
    else:
        # sol.t[-1] tells us the last time the solver reached before crashing
        print(f"β={beta:5d}: FAILED at t = {sol.t[-1]:.4f} years, last I = {sol.y[1, -1]:.2e}")

#%%
# =============================================================================
# DIAGNOSTIC 2: ZOOM INTO THE FIRST FEW YEARS
# Run for very short times to see exactly when the crash happens
# Pick a beta that failed in Diagnostic 1
# =============================================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC 2: Short time runs at beta = 800")
print("=" * 70)

beta_test = 800
pm = {'mu': mu, 'alpha': alpha, 'sigma': sigma,
      'beta0': beta_test, 'eps': 0,
      'delta_prime': delta_prime, 'gamma_prime': gamma_prime}

# try progressively longer runs to find when it breaks
for T_end in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
    sol = solve_ivp(lambda t, y: sirc_modified(t, y, pm),
                    (0, T_end), y0, method='Radau',
                    rtol=1e-8, atol=1e-10, max_step=0.01)

    if sol.status == 0:
        print(f"T = {T_end:6.2f} yr: SUCCESS, {sol.nfev:6d} f-evals, "
              f"{sol.njev:4d} jac-evals, {len(sol.t):5d} pts")
    else:
        # this is the one that broke — print where
        print(f"T = {T_end:6.2f} yr: FAILED  at t = {sol.t[-1]:.6f} yr, "
              f"{sol.nfev:6d} f-evals, {sol.njev:4d} jac-evals")
        break  # no point trying longer if shorter already failed

#%%
# =============================================================================
# DIAGNOSTIC 3: COMPARE STEP SIZES (original vs modified)
# Run both models at the same beta and look at how the solver
# chooses its step sizes. The original should be smooth; the modified
# should show step sizes collapsing right before the crash.
# =============================================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC 3: Step size comparison at beta = 800")
print("=" * 70)

beta_test = 800

# original SIRC
p = {'mu': mu, 'alpha': alpha, 'delta': delta, 'gamma': gamma,
     'sigma': sigma, 'beta0': beta_test, 'eps': 0}
sol_o = solve_ivp(lambda t, y: sirc(t, y, p),
                  (0, 50), y0, method='Radau',
                  rtol=1e-8, atol=1e-10, max_step=0.01)

# modified SIRC (might fail, that's okay)
pm = {'mu': mu, 'alpha': alpha, 'sigma': sigma,
      'beta0': beta_test, 'eps': 0,
      'delta_prime': delta_prime, 'gamma_prime': gamma_prime}
sol_m = solve_ivp(lambda t, y: sirc_modified(t, y, pm),
                  (0, 50), y0, method='Radau',
                  rtol=1e-8, atol=1e-10, max_step=0.01)

print(f"Original: status={sol_o.status}, {len(sol_o.t)} pts, "
      f"{sol_o.nfev} f-evals, {sol_o.njev} jac-evals")
print(f"Modified: status={sol_m.status}, {len(sol_m.t)} pts, "
      f"{sol_m.nfev} f-evals, {sol_m.njev} jac-evals")

# plot the step sizes the solver chose
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# top left: original step sizes over time
dt_o = np.diff(sol_o.t)
axes[0, 0].semilogy(sol_o.t[:-1], dt_o, 'b-', linewidth=0.5)
axes[0, 0].set_ylabel('Step size (years)', fontsize=10)
axes[0, 0].set_title('Original SIRC: step sizes', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# top right: modified step sizes over time
dt_m = np.diff(sol_m.t)
axes[0, 1].semilogy(sol_m.t[:-1], dt_m, 'r-', linewidth=0.5)
axes[0, 1].set_ylabel('Step size (years)', fontsize=10)
axes[0, 1].set_title('Modified SIRC: step sizes', fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# bottom left: original I(t)
axes[1, 0].plot(sol_o.t, sol_o.y[1], 'b-', linewidth=0.8)
axes[1, 0].set_xlabel('Time (years)', fontsize=10)
axes[1, 0].set_ylabel('I(t)', fontsize=10)
axes[1, 0].set_title('Original SIRC: I(t)', fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# bottom right: modified I(t) (might be garbage after failure, but we plot anyway)
axes[1, 1].plot(sol_m.t, sol_m.y[1], 'r-', linewidth=0.8)
axes[1, 1].set_xlabel('Time (years)', fontsize=10)
axes[1, 1].set_ylabel('I(t)', fontsize=10)
axes[1, 1].set_title('Modified SIRC: I(t)', fontsize=11)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(f'Step size analysis at β = {beta_test}', fontsize=13)
plt.tight_layout()
plt.show()
plt.close()

# print the smallest step sizes to see how small they got before crashing
print(f"\nOriginal: min step = {dt_o.min():.2e}, max step = {dt_o.max():.2e}")
print(f"Modified: min step = {dt_m.min():.2e}, max step = {dt_m.max():.2e}")

#%%
# =============================================================================
# DIAGNOSTIC 4: DOES RELAXING TOLERANCES HELP?
# If the solution exists but the solver can't hit the accuracy target,
# looser tolerances should let it through.
# =============================================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC 4: Tolerance sweep at beta = 800")
print("=" * 70)

beta_test = 800
pm = {'mu': mu, 'alpha': alpha, 'sigma': sigma,
      'beta0': beta_test, 'eps': 0,
      'delta_prime': delta_prime, 'gamma_prime': gamma_prime}

# try from tight to loose tolerances
tol_pairs = [
    (1e-10, 1e-12, "very tight"),
    (1e-8,  1e-10, "tight (our default)"),
    (1e-6,  1e-8,  "moderate"),
    (1e-4,  1e-6,  "loose"),
    (1e-3,  1e-5,  "very loose"),
]

for rtol, atol, label in tol_pairs:
    sol = solve_ivp(lambda t, y: sirc_modified(t, y, pm),
                    (0, 50), y0, method='Radau',
                    rtol=rtol, atol=atol, max_step=0.01)

    if sol.status == 0:
        # check if the solution is physical (no negatives)
        min_val = min(sol.y.min(), 0)
        print(f"rtol={rtol:.0e}, atol={atol:.0e} ({label:15s}): "
              f"SUCCESS, {len(sol.t):5d} pts, min(y) = {min_val:.2e}, "
              f"final I = {sol.y[1, -1]:.2e}")
    else:
        print(f"rtol={rtol:.0e}, atol={atol:.0e} ({label:15s}): "
              f"FAILED  at t = {sol.t[-1]:.4f} yr")

#%%
# =============================================================================
# DIAGNOSTIC 5: DOES first_step HELP?
# Force the solver to start with a tiny step instead of guessing.
# =============================================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC 5: first_step sweep at beta = 800")
print("=" * 70)

for fs in [None, 1e-4, 1e-6, 1e-8, 1e-10]:
    kwargs = {'rtol': 1e-8, 'atol': 1e-10, 'max_step': 0.01}
    if fs is not None:
        kwargs['first_step'] = fs

    sol = solve_ivp(lambda t, y: sirc_modified(t, y, pm),
                    (0, 50), y0, method='Radau', **kwargs)

    fs_str = f"{fs:.0e}" if fs is not None else "auto"
    if sol.status == 0:
        print(f"first_step = {fs_str:>8s}: SUCCESS, reached t = {sol.t[-1]:.1f}")
    else:
        print(f"first_step = {fs_str:>8s}: FAILED  at t = {sol.t[-1]:.6f} yr")

#%%
# =============================================================================
# DIAGNOSTIC 6: DOES BDF DO BETTER THAN RADAU?
# Same test, just swap the method.
# =============================================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC 6: BDF vs Radau at several beta values")
print("=" * 70)

for beta in [250, 400, 600, 800, 1500]:
    pm = {'mu': mu, 'alpha': alpha, 'sigma': sigma,
          'beta0': beta, 'eps': 0,
          'delta_prime': delta_prime, 'gamma_prime': gamma_prime}

    sol_r = solve_ivp(lambda t, y: sirc_modified(t, y, pm),
                      (0, 50), y0, method='Radau',
                      rtol=1e-8, atol=1e-10, max_step=0.01)

    sol_b = solve_ivp(lambda t, y: sirc_modified(t, y, pm),
                      (0, 50), y0, method='BDF',
                      rtol=1e-8, atol=1e-10, max_step=0.01)

    # one-line summary for each
    def status_str(sol, name):
        if sol.status == 0:
            return f"{name}: OK (t={sol.t[-1]:.1f})"
        else:
            return f"{name}: FAIL (t={sol.t[-1]:.4f})"

    print(f"β={beta:5d}:  {status_str(sol_r, 'Radau'):30s}  "
          f"{status_str(sol_b, 'BDF')}")