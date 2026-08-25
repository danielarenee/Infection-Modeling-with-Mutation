import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

#%%
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
delta = 1/1.61
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
# ORIGINAL SIRC: fsolve (algebraic equilibrium, instant)

def equilibrium_curve_orig(beta_values):
    I_plus = np.zeros(len(beta_values))
    x_prev = None
    for i, beta in enumerate(beta_values):
        R0 = beta / (mu + alpha)
        if R0 <= 1.0:
            I_plus[i] = 0.0
            continue
        p = {'mu': mu, 'alpha': alpha, 'delta': delta, 'gamma': gamma,
             'sigma': sigma, 'beta0': beta, 'eps': 0}
        def equations(x):
            return sirc(0, x, p)
        if x_prev is not None and x_prev[1] > 0:
            guess = x_prev
        else:
            S_g = 1.0 / R0
            I_g = mu * (1 - 1/R0) / (mu + alpha)
            R_g = alpha * I_g / (mu + delta)
            C_g = max(1e-8, 1 - S_g - I_g - R_g)
            guess = [S_g, I_g, R_g, C_g]
        sol = fsolve(equations, guess, full_output=True)
        x_sol = sol[0]
        if all(x > -1e-10 for x in x_sol) and x_sol[1] > 1e-15:
            I_plus[i] = max(0, x_sol[1])
            x_prev = x_sol
        else:
            I_plus[i] = 0.0
    return I_plus

#%%
# COMPUTE BOTH CURVES

beta_range = np.linspace(130, 5000, 300)
y0 = np.array([0.2, 0.001, 0.499, 0.3])
T_sim = 100
T_avg = 10
t_eval = np.linspace(T_sim - T_avg, T_sim, int(T_avg * 365) + 1)

# original: fsolve
print("Computing original SIRC (fsolve)...")
I_orig = equilibrium_curve_orig(beta_range)

# modified: BDF simulation
I_mod = np.zeros(len(beta_range))
failures_mod = []

print("Computing modified SIRC (BDF)...")
for i, beta in enumerate(beta_range):
    pm = {'mu': mu, 'alpha': alpha, 'sigma': sigma,
          'beta0': beta, 'eps': 0,
          'delta_prime': delta_prime, 'gamma_prime': gamma_prime}
    sol_m = solve_ivp(lambda t, y: sirc_modified(t, y, pm),
                      (0, T_sim), y0, method='BDF',
                      rtol=1e-8, atol=1e-10, max_step=0.01,
                      t_eval=t_eval)

    if sol_m.status == 0:
        I_mod[i] = np.mean(sol_m.y[1])
    else:
        I_mod[i] = np.nan
        failures_mod.append(beta)

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(beta_range)}")

print(f"\nModified failures: {len(failures_mod)}")
if failures_mod:
    print(f"Failed at: {[f'{b:.0f}' for b in failures_mod[:15]]}")

#%%
# PLOT

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(beta_range, I_orig, 'b-', linewidth=2, label='Original SIRC (constant δ)')
ax.plot(beta_range, I_mod, 'r-', linewidth=2, label='Modified SIRC (δ ∝ βSI)')
ax.axvline(x=600, color='gray', linestyle=':', alpha=0.5, label='Calibration point')
ax.set_xlabel('Contact rate β', fontsize=12)
ax.set_ylabel('Endemic prevalence I⁺', fontsize=12)
ax.set_title('Endemic prevalence: constant vs infection-driven drift', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

idx_orig = np.argmax(I_orig)
idx_mod  = np.argmax(np.nan_to_num(I_mod))
print(f"\nOriginal: peak at β = {beta_range[idx_orig]:.0f}, I+ = {I_orig[idx_orig]:.6f}")
print(f"Modified: peak at β = {beta_range[idx_mod]:.0f}, I+ = {I_mod[idx_mod]:.6f}")