
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# RK4 solver
def step(f, t_n, y_n, h, params):
    k1 = h * f(t_n,       y_n,          params)
    k2 = h * f(t_n + h/2, y_n + k1/2,  params)
    k3 = h * f(t_n + h/2, y_n + k2/2,  params)
    k4 = h * f(t_n + h,   y_n + k3,    params)
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

# Models

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


#%% Parameters
mu    = 0.02
alpha = 365/3
delta = 1/1.61
gamma = 0.35
sigma = 0.07874

# Tropical regime (Singapore): high beta, weak seasonality
beta0_trop = 1200
eps_trop   = 0.07

# Temperate regime (England): lower beta, stronger seasonality
beta0_temp = 400
eps_temp   = 0.18

delta_prime = 0.00001
gamma_prime = 0.00004

y0 = np.array([0.2, 0.001, 0.499, 0.3])


#%% Simulation (tropical)

tropical_params_sirc = {
    'mu': mu, 'alpha': alpha, 'delta': delta, 'gamma': gamma,
    'sigma': sigma, 'beta0': beta0_trop, 'eps': eps_trop,
}

tropical_params_sircm = {
    'mu': mu, 'alpha': alpha, 'sigma': sigma,
    'beta0': beta0_trop, 'eps': eps_trop,
    'delta_prime': delta_prime, 'gamma_prime': gamma_prime,
}


t_trop, y_trop = solve(sirc, y0, (0, 300), 1/365, tropical_params_sirc)
t_trop_m, y_trop_m = solve(sirc_modified, y0, (0, 300), 1/365, tropical_params_sircm)

mask = t_trop > 299  # last yr
x_data = (t_trop[mask]) * 12
x_data_mod = (t_trop_m[mask]) * 12

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(x_data, y_trop[mask, 1],  linewidth=2, label='Original SIRC (seasonally adjusted)')
ax.plot(x_data_mod, y_trop_m[mask, 1], 'r-',  linewidth=1.5, label='Modified SIRC (seasonally adjusted)')
ax.set_xlabel('Time (months)', fontsize=12)
ax.set_ylabel('Prevalence I(t)', fontsize=12)
ax.set_title('Tropical (ε = 0.07, β₀ = 1200)', fontsize=13)

custom_labels = [
    "'93 Mar", "'93 Apr", "'93 May", "'93 Jun",
    "'93 Jul", "'93 Aug", "'93 Sep", "'93 Oct",
    "'93 Nov", "'93 Dec", "'94 Jan", "'94 Feb", "'94 Mar"
]
tick_positions = np.linspace(x_data.min(), x_data.max(), len(custom_labels))
ax.set_xticks(tick_positions)
ax.set_xticklabels(custom_labels, rotation=45)

ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.0035)
plt.tight_layout()
plt.show()
plt.close()


#%%

import numpy as np
import matplotlib.pyplot as plt

# ... [tus parámetros y funciones solve se mantienen igual] ...

# 1. AMPLIAR EL TIEMPO: Cambiamos 299 por 295 para ver los últimos 5 años
mask = t_trop > 295

# 2. ESCALA: Quitamos el "* 12" porque ahora graficaremos años, no meses
x_data = t_trop[mask]
x_data_mod = t_trop_m[mask]

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(x_data, y_trop[mask, 1],  linewidth=2, label='Original SIRC (seasonally adjusted)')
ax.plot(x_data_mod, y_trop_m[mask, 1], 'r-',  linewidth=1.5, label='Modified SIRC (seasonally adjusted)')

# Cambiamos la etiqueta del eje X
ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('Prevalence I(t)', fontsize=12)
ax.set_title('Tropical (ε = 0.07, β₀ = 1200)', fontsize=13)

# 3. NUEVAS ETIQUETAS: Creamos una lista para los 5 años (+ el inicial)
custom_labels = [
    "'89", "'90", "'91", "'92", "'93", "'94"
]

# np.linspace distribuirá estas 6 etiquetas exactamente a lo largo de los 5 años de datos
tick_positions = np.linspace(x_data.min(), x_data.max(), len(custom_labels))
ax.set_xticks(tick_positions)
ax.set_xticklabels(custom_labels, rotation=0) # rotación en 0 porque los años caben bien sin inclinarse

ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.0035)
plt.tight_layout()
plt.show()
plt.close()