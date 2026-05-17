import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# RK4 solver

# RK4 step function
def step(f, t_n, y_n, h, params):
    k1 = h * f(t_n, y_n, params)
    k2 = h * f(t_n + h/2, y_n + k1/2, params)
    k3 = h * f(t_n + h/2, y_n + k2/2, params)
    k4 = h * f(t_n + h, y_n + k3, params)
    return y_n + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

# function to solve the ODE
def solve(f, y0, t_span, h, params):
    t_start, t_end = t_span
    N_steps = int(np.round((t_end - t_start) / h))
    t_arr = np.linspace(t_start, t_end, N_steps + 1)
    y_arr = np.empty((N_steps + 1, len(y0)))
    y_arr[0] = y0
    for n in range(N_steps):
        y_arr[n+1] = step(f, t_arr[n], y_arr[n], h, params)
    return t_arr, y_arr

# SIRC MODEL

# it takes the vector Y=[s,i,r,c] and a dictionary of parameters "params"
def sirc(t, y, params):
    S,I,R,C = y
    mu = params['mu']
    alpha = params['alpha']
    delta = params['delta']
    gamma = params['gamma']
    sigma = params['sigma']
    beta0 = params['beta0']
    eps = params.get('eps', 0) # so we can implement seasonality if we wish to

    beta = beta0*(1 + eps*np.cos(2*np.pi*t)) # when eps=0 this is just beta0

    dSdt = mu*(1 - S) - beta*S*I + gamma*C
    dIdt = beta*S*I + sigma*beta*C*I - (mu+alpha)*I
    dRdt = (1-sigma) * beta*C*I + alpha*I - (mu+delta)*R
    dCdt = delta*R - beta*C*I - (mu+gamma)*C
    return np.array([dSdt, dIdt, dRdt, dCdt])

# Realistic parameters for influenza A
base_params = {
    'mu':    0.02,
    'alpha': 365/3,
    'delta': 1/1.6,
    'gamma': 0.35,
    'sigma': 0.07874,
    'beta0': 600, #placeholder
    'eps':   0,
}

# Figures
# 1. equilibrium curve (Prevalence I+ vs beta)

def equilibrium_curve(beta_values, params):
    # array to store the endemic prevalence for each beta
    I_plus = np.zeros(len(beta_values))
    # x_prev stores the previous solution
    x_prev = None

    for i, beta in enumerate(beta_values):
        params['beta0'] = beta

        # compute R0 for this beta
        R0 = beta / (params['mu'] + params['alpha'])

        # endemic equilibrium
        if R0 <= 1.0:
            I_plus[i] = 0.0
            continue

        # we get the derivative vector to set it to zero via sirc function
        def equations(x):
            return sirc(0, x, params)

        # initial guess for Newton's method
        if x_prev is not None and x_prev[1] > 0:
            # we use the previous solution as a guess
            guess = x_prev
        else:
            # first time above R0=1
            S_g = 1.0 / R0
            I_g = params['mu']*(1-1/R0)/(params['mu'] + params['alpha'])
            R_g = params['alpha']*I_g / (params['mu'] + params['delta'])
            C_g = max(1e-8, 1-S_g-I_g-R_g) # c = 1-s-i-r
            guess = [S_g, I_g, R_g, C_g]

        # fsolve will find x where equations(x) = [0,0,0,0]
        sol = fsolve(equations, guess, full_output=True)
        x_sol = sol[0]  # the root: [S*, I*, R*, C*]

        # validate that all compartments non-negative, and I is meaningfully positive
        if all(x > -1e-10 for x in x_sol) and x_sol[1] > 1e-15:
            I_plus[i] = max(0, x_sol[1])  # store endemic prevalence
            x_prev = x_sol  # save for continuation
        else:
            I_plus[i] = 0.0

    return I_plus

mu    = base_params['mu']
alpha = base_params['alpha']
delta = base_params['delta']
gamma = base_params['gamma']

# SIRC: 500 points
beta_range = np.linspace(10, 5000, 500)
I_sirc = equilibrium_curve(beta_range, base_params.copy())

# SIRS (sigma=1 collapses SIRC into SIRS)
R0_arr = beta_range / (mu + alpha)
I_sirs = np.zeros(len(R0_arr))
for i in range(len(R0_arr)):
    if R0_arr[i] > 1:
        I_sirs[i] = (1 - 1/R0_arr[i]) * (mu + delta) / (mu + delta + alpha)
    else:
        I_sirs[i] = 0.0

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(beta_range, I_sirc, 'b-', linewidth=2, label='SIRC')
ax1.set_xlabel('Contact rate β', fontsize=12)
ax1.set_ylabel('Prevalence I⁺ (SIRC)', fontsize=12)
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.plot(beta_range, I_sirs, 'r--', linewidth=2, label='SIRS')
ax2.set_ylabel('Prevalence I⁺ (SIRS)', fontsize=12)
ax2.tick_params(axis='y')

ax1.set_title('Endemic prevalence vs contact rate', fontsize=13)
fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.88))
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

idx_max = np.argmax(I_sirc)
beta_star = beta_range[idx_max]
R0_star = beta_star / (mu + alpha)
print(f"β* ≈ {beta_star:.0f},  R₀* ≈ {R0_star:.2f}")
