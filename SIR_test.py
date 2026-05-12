import numpy as np

# Solver

# function for the ODE
# it takes y, the vector [s,i,r]
def sir(t, y, beta, gamma):
    s,i,r = y
    dsdt = -beta*i*s
    didt = beta*i*s - gamma*i
    drdt = gamma*i
    return np.array([dsdt, didt, drdt])

# step h function
# where y_n is the current value at time t_n
def step(f, t_n, y_n, h, beta, gamma):
    k1 = h*f(t_n, y_n, beta, gamma)
    k2 = h*f(t_n + h/2, y_n + k1/2, beta, gamma)
    k3 = h*f(t_n + h/2, y_n + k2/2, beta, gamma)
    k4 = h*f(t_n + h, y_n + k3, beta, gamma)

    y_next = y_n + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

    return y_next

# loop function
# this integrates the system from tspan(0) to tspan(1)
def solve(f, y0, t_span, h, beta, gamma):
    t_start, t_end = t_span

    '''t_arr = np.arange(t_start, t_end, h)
    N_steps = len(t_arr)-1  # number of steps'''

    N_steps = int(np.round((t_end - t_start) / h))
    t_arr = np.linspace(t_start, t_end, N_steps + 1)

    y_list = [y0] # initial value

    for n in range(N_steps):
        y_next = step(f, t_arr[n], y_list[n], h, beta, gamma)
        y_list.append(y_next)

    return t_arr, np.array(y_list)


#%%
# Test run

import matplotlib.pyplot as plt
beta = 0.3
gamma = 0.1 # 10 days of being infected

# initial conditions (1% infected population)
y0 = np.array([0.99, 0.01, 0.00])

# lets solve for 200 days with h = 0.1
t, y = solve(sir, y0, (0, 200), h=0.1, beta=beta, gamma=gamma)

plt.figure(figsize=(10, 6))
plt.plot(t, y[:, 0], label='S(t)')
plt.plot(t, y[:, 1], label='I(t)')
plt.plot(t, y[:, 2], label='R(t)')
plt.xlabel('Time (days)')
plt.ylabel('Fraction of population')
plt.title(f'SIR epidemic model')
plt.legend()
plt.grid(True)
plt.show()

#%%

# Validation
from scipy.optimize import brentq

# 1. testing lim S(t) as t goes to infinity

# in our test, sigma is 0.3/0.1 = 3
sigma = beta / gamma
s0, i0, r0 = y0

# we know that at t-> infinity, i=0
C = i0 + s0 - (1 / sigma)*np.log(s0)

# so we look for s sucj that: s - (1/sigma)*ln(s) = C

def g(s):
    return s - (1/sigma)*np.log(s) - C

# find the root in (0, 1/sigma) as stated by theorem 2.1
s_inf_theory = brentq(g, 1e-10, 1 / sigma)

# finally we compare with the final values of the solution
s_inf_numerical = y[-1, 0]
i_inf_numerical = y[-1,1]

print(f"theoretical s_inf:  {s_inf_theory:.10f}")
print(f"numerical s_inf:    {s_inf_numerical:.10f}")
print(f"difference:         {abs(s_inf_theory - s_inf_numerical):.2e}")
print(f"numerical i_inf:    {i_inf_numerical:.2e}") # should be aprox. 0

# yay

#%%

# 2. testing convergence

# note: we will pick a time where a non-trivial part of the population
# is infected. Based on the graph, let's pick T_eval = 30

T_eval = 30.0

# lets first compute a reference solution
h_ref = 1e-5 # very small h for reference
t_ref, y_ref = solve(sir, y0, (0, T_eval), h_ref, beta=beta, gamma=gamma)
y_exact = y_ref[-1]  # state vector at T_eval

# lets run with different h values
h_values = [2, 1, 0.5, 0.25, 0.125, 0.0625]

# now lets compute the error for each...
errors = []

for h in h_values:
    t_arr, y_arr = solve(sir, y0, (0, T_eval), h, beta=beta, gamma=gamma)
    y_numerical = y_arr[-1]  # state at T_eval
    # error = max absolute difference across the 3 components
    error = np.max(np.abs(y_numerical - y_exact))
    errors.append(error)

plt.figure()
plt.loglog(h_values, errors, 'o-')
plt.xlabel('h (step size)')
plt.ylabel('E (global error)')
plt.title('RK4 convergence test')
plt.grid(True)
plt.legend()
plt.show()

# we compute the slope to check
for j in range(1, len(h_values)):
    slope = np.log(errors[j] / errors[j-1]) / np.log(h_values[j] / h_values[j-1])
    print(f"h={h_values[j]:.4f}, slope = {slope:.2f}")



