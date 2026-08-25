"""
Tests convergence between handcoded RK4 and solve_ivp Dormand Prince
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

#%%

# set parameters
mu = 0.02
alpha = 365/3
delta = 1/1.61
gamma = 0.35
sigma = 0.07874
beta00 = 400

#y0 = np.array([0.2, 0.001, 0.499, 0.3])
y0 = np.array([0.2, 0.499,0.001, 0.3])

params_season = {'beta0': beta00, 'eps': 0}

def beta_t(t, beta0, eps): # seasonally forced beta
    return beta0 * (1.0 + eps * np.cos(2.0 * np.pi * t))

# sirc model
def sirc(t, y, p):
    S, I, R, C = y
    b = beta_t(t, p['beta0'], p['eps'])
    dS = mu * (1 - S) - b * S * I + gamma * C
    dI = b * S * I + sigma * b * C * I - (mu + alpha) * I
    dR = (1 - sigma) * b * C * I + alpha * I - (mu + delta) * R
    dC = delta * R - b * C * I - (mu + gamma) * C
    return np.array([dS, dI, dR, dC])

# hand coded rk4 ===
def step(f, t_n, y_n, h, params):
    k1 = h * f(t_n, y_n, params)
    k2 = h * f(t_n + h / 2, y_n + k1 / 2, params)
    k3 = h * f(t_n + h / 2, y_n + k2 / 2, params)
    k4 = h * f(t_n + h, y_n + k3, params)
    return y_n + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

def solve_rk4(f, y0, t_span, h, params):
    t_start, t_end = t_span
    N_steps = int(np.round((t_end - t_start) / h))
    t_arr = np.linspace(t_start, t_end, N_steps + 1)
    y_arr = np.empty((N_steps + 1, len(y0)))
    y_arr[0] = y0
    for n in range(N_steps):
        y_arr[n + 1] = step(f, t_arr[n], y_arr[n], h, params)
    return t_arr, y_arr

#%% convergence test

T_end = 10 # num of years

sol_exact = solve_ivp(sirc, (0, T_end), y0, method='DOP853', args=(params_season,),
                      rtol=1e-13, atol=1e-13)
y_exact_end = sol_exact.y[:, -1] # extracts the final state

h_vals = [1/50, 1/100, 1/200, 1/400, 1/800] # array of step sizes to test
errors = []

# print convergence info...
print(f"{'h':<10} | {'Max error':<20} | {'Ratio E(h)/E(h/2)':<20} | {'Calculated Order':<20}")

for i, h in enumerate(h_vals): #loop through step sizes
    t_rk4, y_rk4 = solve_rk4(sirc, y0, (0, T_end), h, params_season) # hand coded rk4
    y_rk4_end = y_rk4[-1, :] # last state
    # I use last states to get global error, and also considering that step sizes are not compatible

    err = np.max(np.abs(y_rk4_end - y_exact_end))
    errors.append(err)

    if i == 0:
        print(f"{h:<10.5f} | {err:<20.5e} | {'-':<20} | {'-':<20}")
    else:
        ratio = errors[i - 1] / err
        order = np.log2(ratio)
        print(f"{h:<10.5f} | {err:<20.5e} | {ratio:<20.5f} | {order:<20.5f}")

#%%
# performance timing
print("performance timing... ")

# I got different results so I will average

times = []
for i in range(20):
    start = time.perf_counter()  # rk4
    solve_rk4(sirc, y0, (0, 100), 1 / 365, params_season)
    rk4_time = time.perf_counter() - start

    start = time.perf_counter()  # dp
    solve_ivp(sirc, (0, 100), y0, method='DOP853', args=(params_season,),
              rtol=1e-6, atol=1e-10, max_step=1 / 365)
    scipy_time = time.perf_counter() - start

    print(f"Iteration: {i+1}/20")
    print(f"RK4 Time:   {rk4_time:.4f} seconds")
    print(f"SciPy Time: {scipy_time:.4f} seconds\n")

    ratio = scipy_time / rk4_time
    times.append(ratio)

avg_ratio = np.mean(times)
print(f"solve_ivp is on average {avg_ratio} times slower than RK4 (handcoded)")
plt.show()