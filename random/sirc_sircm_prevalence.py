"""
calcula la prevalencia para SIRC y SIRCm
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# SETUP

# RIGHT-HAND SIDES
def sirc(t, y, p):
    """Original SIRC: delta and gamma are constant"""
    S, I, R, C = y
    b = beta_t(t, p['beta0'], p['eps'])
    dS = mu*(1 - S) - b*S*I + gamma*C
    dI = b*S*I + sigma*b*C*I - (mu + alpha)*I
    dR = (1 - sigma)*b*C*I + alpha*I - (mu + delta)*R
    dC = delta*R - b*C*I - (mu + gamma)*C
    return np.array([dS, dI, dR, dC])

def sircm(t, y, p):
    """
    Modified SIRCm: the immunity-erosion rates are driven by incidence
    """
    S, I, R, C = y
    b = beta_t(t, p['beta0'], p['eps'])
    inc = b*S*I
    de = p['delta_prime']*inc      # effective delta
    ge = p['gamma_prime']*inc      # effective gamma
    dS = mu*(1 - S) - b*S*I + ge*C
    dI = b*S*I + sigma*b*C*I - (mu + alpha)*I
    dR = (1 - sigma)*b*C*I + alpha*I - (mu + de)*R
    dC = de*R - b*C*I - (mu + ge)*C
    return np.array([dS, dI, dR, dC])

# seasonal forcing (if eps=0 then there is no seasonal forcing)
def beta_t(t, beta0, eps):
    return beta0 * (1.0 + eps * np.cos(2.0 * np.pi * t))

# PARAMETERS
mu    = 0.02       # birth = death rate
alpha = 365/3      # recovery rate I->R
delta = 1/1.61     # R->C rate (loss of full immunity SIRC)
gamma = 0.35       # C->S rate (loss of partial immunity SIRC)
sigma = 0.07874    # reduced susceptibility of cross-immune class C
beta0 = 600

# new constants
delta_prime = 0.001
gamma_prime = 0.0004

y0_ts = np.array([0.2, 0.001, 0.499, 0.3])  # initial val for S,I,R,C

# seasonal forcing param
p_orig = {'beta0': beta0, 'eps': 0}
p_mod  = {'beta0': beta0, 'eps': 0,
          'delta_prime': delta_prime, 'gamma_prime': gamma_prime}


# INTEGRATION & PREVALENCE CALCULATION

# integration parameters
t_span = (0, 50)  # years
t_eval = np.linspace(0, 50, 5000)

# solve SIRC
sol_sirc = solve_ivp(sirc, t_span, y0_ts, args=(p_orig,),
                     method='DOP853', t_eval=t_eval,
                     rtol=1e-6, atol=1e-9)

# solve SIRCm
sol_sircm = solve_ivp(sircm, t_span, y0_ts, args=(p_mod,),
                      method='DOP853', t_eval=t_eval,
                      rtol=1e-6, atol=1e-9)

# si no funciona le puedes agregar el max_step=1/365

# extract prevalence I(t) for both
I_sirc = sol_sirc.y[1, :]   # I is second component
I_sircm = sol_sircm.y[1, :]

# plot comparison
plt.figure(figsize=(10, 6))
plt.plot(sol_sirc.t, I_sirc, label='SIRC', linewidth=2)
plt.plot(sol_sircm.t, I_sircm, label='SIRCm', linewidth=2)
plt.xlabel('Time (years)')
plt.ylabel('Prevalence I(t)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Prevalence Comparison: SIRC vs SIRCm')
plt.tight_layout()
plt.show()
