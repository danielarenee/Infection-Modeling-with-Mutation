
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ─────────────────────────────────────────────
# RK4 SOLVER
# ─────────────────────────────────────────────

def step(f, t_n, y_n, h, params):
    k1 = h * f(t_n, y_n, params)
    k2 = h * f(t_n + h/2, y_n + k1/2, params)
    k3 = h * f(t_n + h/2, y_n + k2/2, params)
    k4 = h * f(t_n + h, y_n + k3, params)
    return y_n + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

def solve(f, y0, t_span, h, params):
    t_start, t_end = t_span
    N_steps = int(np.round((t_end - t_start) / h))
    t_arr = np.linspace(t_start, t_end, N_steps + 1)
    y_arr = np.empty((N_steps + 1, len(y0)))
    y_arr[0] = y0
    for n in range(N_steps):
        y_arr[n+1] = step(f, t_arr[n], y_arr[n], h, params)
    return t_arr, y_arr

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

def sirc(t, y, params):
    """Original SIRC (Casagrandi 2006)"""
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
    """Modified SIRC: delta and gamma driven by infection incidence"""
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


def sirc_modified_floor(t, y, params):
    """Modified SIRC with numerical floor on I to prevent extinction"""
    S, I, R, C = y
    I = max(I, 1e-30)

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


# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────

mu    = 0.02
alpha = 365/3
delta = 1/1.61 # 0.62
gamma = 0.35
sigma = 0.07874

# ─────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────

calib_params = {'mu': mu, 'alpha': alpha, 'delta': delta, 'gamma': gamma,
                'sigma': sigma, 'beta0': 600, 'eps': 0}

def equilibrium_equations(x):
    return sirc(0, x, calib_params)

eq = fsolve(equilibrium_equations, np.array([0.2, 0.001, 0.499, 0.3]))
S_eq, I_eq, R_eq, C_eq = eq
incidence_eq = 600 * S_eq * I_eq

delta_prime = delta / incidence_eq
gamma_prime = gamma / incidence_eq

print(f"Calibration at beta0=600:")
print(f"  S+={S_eq:.6f}, I+={I_eq:.6f}, R+={R_eq:.6f}, C+={C_eq:.6f}")
print(f"  incidence = {incidence_eq:.6f}")
print(f"  delta' = {delta_prime:.4f}")
print(f"  gamma' = {gamma_prime:.4f}")

# ─────────────────────────────────────────────
# COMMON SETTINGS
# ─────────────────────────────────────────────

h = 1/365
y0_small = np.array([0.999, 0.000001, 0.0, 0.0])  # tiny seed, almost everyone susceptible


#%%

# TESTS: What is going on with the smaller betas?

pm_below = {'mu': mu, 'alpha': alpha, 'sigma': sigma,
            'beta0': 100, 'eps': 0,
            'delta_prime': delta_prime, 'gamma_prime': gamma_prime}

pm_dead = {'mu': mu, 'alpha': alpha, 'sigma': sigma,
           'beta0': 150, 'eps': 0,
           'delta_prime': delta_prime, 'gamma_prime': gamma_prime}

pm_above = {'mu': mu, 'alpha': alpha, 'sigma': sigma,
            'beta0': 200, 'eps': 0,
            'delta_prime': delta_prime, 'gamma_prime': gamma_prime}

# does the disease come back at beta=150?

t_floor, y_floor = solve(sirc_modified_floor, y0_small, (0, 500), h, pm_dead)
I_floor = np.maximum(y_floor[:, 1], 1e-30)

# what about beta = 200?

t_floor2, y_floor2 = solve(sirc_modified_floor, y0_small, (0, 500), h, pm_above)
I_floor2 = np.maximum(y_floor2[:, 1], 1e-30)


t_floor3, y_floor3 = solve(sirc_modified, y0_small, (0, 500), h, pm_below)
I_floor3 = np.maximum(y_floor3[:, 1], 1e-30)

fig, ax = plt.subplots(figsize=(14, 5))
ax.semilogy(t_floor, I_floor, 'r-', linewidth=0.5, label='β₀=150 (dead zone)')
ax.semilogy(t_floor2, I_floor2, 'g-', linewidth=0.5, label='β₀=200 (above R_0 = 1.5)')
ax.semilogy(t_floor3, I_floor3, 'b-', linewidth=0.5, label='β₀=100 (below R_0 = 1)')
ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('I(t) [log scale]', fontsize=12)
ax.set_title('comparison: does I recover?', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()


#%%

# lets plot endemic prevalence for several beta

beta_test = np.linspace(104, 1000, 50)
I_final = np.zeros(len(beta_test))

for i, b in enumerate(beta_test):
    pm = {'mu': mu, 'alpha': alpha, 'sigma': sigma,
          'beta0': b, 'eps': 0,
          'delta_prime': delta_prime, 'gamma_prime': gamma_prime}
    _, y = solve(sirc_modified, y0, (0, 500), 1/365, pm)
    I_final[i] = np.mean(y[-3650:, 1])
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(beta_test)}, beta={b:.0f}, I_final={I_final[i]:.8f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(beta_test, I_final, 'r-o', markersize=3)
ax.axvline(x=mu+alpha, color='k', linestyle=':', label='R₀ = 1')
ax.set_xlabel('β₀', fontsize=12)
ax.set_ylabel('Long-run prevalence I', fontsize=12)
ax.set_title('At what β₀ does the modified model sustain disease?', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

#%%

# lets plot endemic prevalence for several beta

# first, with a small delta and gamma prime

y0_small = np.array([0.2, 0.001, 0.499, 0.3])

delta_prime = 0.25
gamma_prime = 0.15

beta_range = np.linspace(100, 200, 50)
I_orig = np.zeros(len(beta_range))
I_mod = np.zeros(len(beta_range))

print("Computing prevalence curves (this will take a while)...")
for i, beta in enumerate(beta_range):
    # original
    p = {'mu': mu, 'alpha': alpha, 'delta': delta, 'gamma': gamma,
         'sigma': sigma, 'beta0': beta, 'eps': 0}
    _, y = solve(sirc, y0_small, (0, 200), h, p)
    I_orig[i] = np.mean(y[-3650:, 1])

    # modified with floor, run longer
    pm = {'mu': mu, 'alpha': alpha, 'sigma': sigma,
          'beta0': beta, 'eps': 0,
          'delta_prime': delta_prime, 'gamma_prime': gamma_prime}
    _, ym = solve(sirc_modified_floor, y0_small, (0, 2000), h, pm)
    # average the last 500 years to capture multiple outbreak cycles
    I_mod[i] = np.mean(ym[-int(500*365):, 1])

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(beta_range)}, beta={beta:.0f}")

"""fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(beta_range, I_orig, 'b-', linewidth=2, label='Original SIRC')
ax.plot(beta_range, I_mod, 'r--', linewidth=2, label='Modified SIRC')
ax.axvline(x=mu+alpha, color='k', linestyle=':', label='R₀ = 1')
ax.set_xlabel('Contact rate β₀', fontsize=12)
ax.set_ylabel('Endemic prevalence I⁺', fontsize=12)
ax.set_title('Endemic prevalence (small delta and gamma prime)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()"""

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(beta_range, np.maximum(I_orig, 1e-30), 'b-', linewidth=2, label='Original SIRC')
ax.semilogy(beta_range, np.maximum(I_mod, 1e-30), 'r--', linewidth=2, label='Modified SIRC')
ax.axvline(x=mu+alpha, color='k', linestyle=':', label='R₀ = 1')
ax.set_xlabel('Contact rate β₀', fontsize=12)
ax.set_ylabel('Endemic prevalence I⁺ [log scale]', fontsize=12)
ax.set_title('Zoom near R₀=1', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()


# ─────────────────────────────────────────────
# PRINT VALUES ABOVE R0 = 1
# ─────────────────────────────────────────────

print(f"small values!!!!!!!")

# The epidemic threshold where R0 = 1
r0_1_threshold = mu + alpha

print(f"\nR0 = 1 occurs at β₀ = {r0_1_threshold:.2f}")
print("   β₀   | Original I⁺  | Modified I⁺ ")
print("-" * 40)

for b, i_o, i_m in zip(beta_range, I_orig, I_mod):
    if b > r0_1_threshold:
        # Print with 8 decimal places to easily spot small positive numbers
        print(f"{b:7.2f} |  {i_o:.8f}  |  {i_m:.8f}")


# then, with a bigger delta and gamma prime (the ones we've been using)

delta_prime = 5.06
gamma_prime = 2.85

beta_range = np.linspace(100, 200, 50)
I_orig = np.zeros(len(beta_range))
I_mod = np.zeros(len(beta_range))

print("Computing prevalence curves (this will take a while)...")
for i, beta in enumerate(beta_range):
    # original
    p = {'mu': mu, 'alpha': alpha, 'delta': delta, 'gamma': gamma,
         'sigma': sigma, 'beta0': beta, 'eps': 0}
    _, y = solve(sirc, y0_small, (0, 200), h, p)
    I_orig[i] = np.mean(y[-3650:, 1])

    # modified with floor, run longer
    pm = {'mu': mu, 'alpha': alpha, 'sigma': sigma,
          'beta0': beta, 'eps': 0,
          'delta_prime': delta_prime, 'gamma_prime': gamma_prime}
    _, ym = solve(sirc_modified_floor, y0_small, (0, 2000), h, pm)
    # average the last 500 years to capture multiple outbreak cycles
    I_mod[i] = np.mean(ym[-int(500*365):, 1])

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(beta_range)}, beta={beta:.0f}")

"""fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(beta_range, I_orig, 'b-', linewidth=2, label='Original SIRC')
ax.plot(beta_range, I_mod, 'r--', linewidth=2, label='Modified SIRC')
ax.axvline(x=mu+alpha, color='k', linestyle=':', label='R₀ = 1')
ax.set_xlabel('Contact rate β₀', fontsize=12)
ax.set_ylabel('Endemic prevalence I⁺', fontsize=12)
ax.set_title('Endemic prevalence (big delta and gamma prime)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()"""

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(beta_range, np.maximum(I_orig, 1e-30), 'b-', linewidth=2, label='Original SIRC')
ax.semilogy(beta_range, np.maximum(I_mod, 1e-30), 'r--', linewidth=2, label='Modified SIRC')
ax.axvline(x=mu+alpha, color='k', linestyle=':', label='R₀ = 1')
ax.set_xlabel('Contact rate β₀', fontsize=12)
ax.set_ylabel('Endemic prevalence I⁺ [log scale]', fontsize=12)
ax.set_title('Zoom near R₀=1', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

# ─────────────────────────────────────────────
# PRINT VALUES ABOVE R0 = 1
# ─────────────────────────────────────────────

# The epidemic threshold where R0 = 1
r0_1_threshold = mu + alpha

print(f"big values!!!!!!!")

print(f"\nR0 = 1 occurs at β₀ = {r0_1_threshold:.2f}")
print("   β₀   | Original I⁺  | Modified I⁺ ")
print("-" * 40)

for b, i_o, i_m in zip(beta_range, I_orig, I_mod):
    if b > r0_1_threshold:
        # Print with 8 decimal places to easily spot small positive numbers
        print(f"{b:7.2f} |  {i_o:.8f}  |  {i_m:.8f}")

