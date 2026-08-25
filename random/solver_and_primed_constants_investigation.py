
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

""" SETUP """

#%% PARAMETERS ================================================
mu    = 0.02       # birth = death rate
alpha = 365/3      # recovery rate I->R
delta = 1/1.61     # R->C rate (loss of full immunity), original constant
gamma = 0.35       # C->S rate (loss of partial immunity), original constant
sigma = 0.07874    # reduced susceptibility of cross-immune class C

def beta_t(t, beta0, eps):
    return beta0 * (1.0 + eps * np.cos(2.0 * np.pi * t))

#%% RIGHT-HAND SIDES ================================================
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

#%% HAND CODED RK4 SOLVER ================================================

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

#%% ANALYTIC JACOBIANS ================================================

"""def jac_sirc(t, y, p):
    S, I, R, C = y
    b = beta_t(t, p['beta0'], p['eps'])
    return np.array([
        [-mu - b*I,  -b*S,                    0.0,           gamma          ],
        [ b*I,        b*S + sigma*b*C - (mu+alpha),    0.0,  sigma*b*I      ],
        [ 0.0,       (1-sigma)*b*C + alpha, -(mu+delta),    (1-sigma)*b*I   ],
        [ 0.0,       -b*C,                 delta,          -b*I - (mu+gamma)],
    ])

def jac_sircm(t, y, p):
    S, I, R, C = y
    b  = beta_t(t, p['beta0'], p['eps'])
    dp = p['delta_prime']
    gp = p['gamma_prime']
    return np.array([
        [-mu - b*I + gp*b*I*C,  -b*S + gp*b*S*C,               0.0,              gp*b*S*I        ],
        [ b*I,                   b*S + sigma*b*C - (mu+alpha), 0.0,              sigma*b*I       ],
        [-dp*b*I*R,              (1-sigma)*b*C + alpha - dp*b*S*R, -mu - dp*b*S*I, (1-sigma)*b*I ],
        [ dp*b*I*R - gp*b*I*C,   dp*b*S*R - b*C - gp*b*S*C,    dp*b*S*I,     -b*I - mu - gp*b*S*I],
    ])"""


A=0

#%%
""" ON CALIBRATING DELTA PRIME AND GAMMA PRIME """

#%% CALIBRATE WITH SPECIFIC BETA (BETA = 600)

beta0 = 600
eps   = 0
par = { 'beta0': beta0, 'eps': eps,}

def sirc_helper(x):
    return sirc(0.0, x, par)  # returns [dS,dI,dR,dC]

guess = [0.2, 0.001, 0.499, 0.3] # initial guess

eq    = fsolve(sirc_helper, guess) #solve dS=dI=dR=dC=0 for sirc at beta0
S_eq, I_eq, R_eq, C_eq = eq

incidence_eq = beta0 * S_eq * I_eq  # define BSI
delta_prime  = delta / incidence_eq  # force the new delta =  delta'*beta*S*I
gamma_prime  = gamma / incidence_eq  # same

print(f"Calibration @ beta0={beta0}: incidence={incidence_eq:.5f}, "
      f"delta'={delta_prime:.4f}, gamma'={gamma_prime:.4f}")


#%% MANUALLY SETTING (LOWER BOUND VALUES WHEN BSI IS MAXIMUM)

"""# old constants
delta_prime = 5.0616
gamma_prime = 2.8522"""

# new constants
delta_prime = 0.001
gamma_prime = 0.0004

beta00 = 600

y0_ts = np.array([0.2, 0.001, 0.499, 0.3])  # same IC as the RK4 script

# varying the initial values ====

#y0_ts = np.array([0.8000, 0.0001, 0.0999, 0.1])
#y0_ts = np.array([0.7990, 0.0011, 0.0999, 0.1])
#y0_ts = np.array([0.7950, 0.0051, 0.0999, 0.1])
#y0_ts = np.array([0.7900, 0.0101, 0.0999, 0.1])

#y0_ts = np.array([0.7500, 0.0500, 0.1, 0.1])
#y0_ts = np.array([0.6500, 0.1500, 0.1, 0.1])
#y0_ts = np.array([0.6000, 0.2000, 0.1, 0.1])


p_orig = {'beta0': beta00, 'eps': 0}
p_mod  = {'beta0': beta00, 'eps': 0,
          'delta_prime': delta_prime, 'gamma_prime': gamma_prime}

a=0

#%% USING SIRC ENDEMIC EQ


#%%
""" PLOTS """

#%% ===| PLOT 1 |==== SIRC vs SIRCm, 50 years, plot I(t)

T1 = 50

t_eval1 = np.linspace(0, T1, int(T1 * 365) + 1)   # daily output grid

sol_orig = solve_ivp(sirc, (0, T1), y0_ts, method='DOP853', args=(p_orig,),
                     rtol=1e-6, atol=1e-10,
                     max_step=0.002, t_eval=t_eval1)
sol_mod  = solve_ivp(sircm, (0, T1), y0_ts, method='DOP853', args=(p_mod,),
                     rtol=1e-8, atol=1e-10,
                     max_step=0.002, t_eval=t_eval1)

print(f"SIRC : success={sol_orig.success}  {sol_orig.message}")
print(f"SIRCm: success={sol_mod.success}  {sol_mod.message}")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(sol_orig.t, sol_orig.y[1], 'b-',  linewidth=1, label='Original SIRC')
ax.plot(sol_mod.t,  sol_mod.y[1],  'r--', linewidth=1, label='Modified SIRCm')
ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('Prevalence I(t)', fontsize=12)
ax.set_title(f'Original vs Modified SIRC- DOP853 (β₀ = {beta00}, no seasonality, {T1} yrs)', fontsize=13)

ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
#plt.ylim(0,0.10)
plt.show()
plt.close()

#%%

# === WITH HAND CODED SOLVER === #

h = 1/365
t_orig, y_orig = solve(sirc, y0_ts, (0, T1), h, p_orig)
t_mod,  y_mod  = solve(sircm, y0_ts, (0, T1), h, p_mod)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(t_orig, y_orig[:, 1], 'b-',  linewidth=1, label='Original SIRC')
ax.plot(t_mod,  y_mod[:, 1],  'r--', linewidth=1, label='Modified SIRCm')
ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('Prevalence I(t)', fontsize=12)
ax.set_title(f'Original vs Modified SIRC- hand coded RK4 (β₀ = {beta00}, no seasonality, {T1} yrs)', fontsize=13)

ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.ylim(0,0.05)
plt.show()
plt.close()

#%% ===| PLOT 2|==== SIRCm ALL COMPARTMENTS
T2 = 20

# === WITH SOLVE_IVP === #
t_eval2 = np.linspace(0, T2, int(T2 * 365) + 1)   # daily grid out to 200

sol_all = solve_ivp(sircm, (0, T2), y0_ts, method='DOP853', args=(p_mod,),
                    rtol=1e-8, atol=1e-10,
                    max_step=1/365, t_eval=t_eval2)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(sol_all.t, sol_all.y[0], 'b-',  linewidth=1, label='S (susceptible)')
ax.plot(sol_all.t, sol_all.y[1], 'r-',  linewidth=1, label='I (infected)')
ax.plot(sol_all.t, sol_all.y[2], 'g-',  linewidth=1, label='R (recovered)')
ax.plot(sol_all.t, sol_all.y[3], '-',   linewidth=1, color='orange', label='C (cross-immune)')
ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('Fraction of population', fontsize=12)
ax.set_title(f'solve_ivp DOP853 (β₀ = {beta00}, no seasonality, {T2} yrs)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0,1)
plt.tight_layout()
plt.show()
plt.close()

# === WITH HAND CODED SOLVER === #

t_mod, y_mod = solve(sircm, y0_ts, (0, T2), 1/365, p_mod)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(t_mod, y_mod[:, 0], 'b-',  linewidth=1, label='S (susceptible)')
ax.plot(t_mod, y_mod[:, 1], 'r-',  linewidth=1, label='I (infected)')
ax.plot(t_mod, y_mod[:, 2], 'g-',  linewidth=1, label='R (recovered)')
ax.plot(t_mod, y_mod[:, 3], '-',   linewidth=1, color='orange', label='C (cross-immune)')

ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('Fraction of population', fontsize=12)
ax.set_title(f'hand coded RK4 (β₀ = {beta00}, no seasonality, {T2} yrs)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

#%% ===| PLOT 3|==== ENDEMIC PREVALENCE WITH DIFF BETA VALUES

beta_range = np.linspace(100, 1500, 200)
T_sim = 300
T_avg = 50
t_eval = np.linspace(T_sim - T_avg, T_sim, int(T_avg * 365) + 1)  # last 10 yrs

I_orig = np.zeros(len(beta_range))
I_mod  = np.zeros(len(beta_range))
failures_orig = []
failures_mod  = []

print("Computing both curves (DOP853)...")
for i, beta in enumerate(beta_range):

    # --- original SIRC ---
    po = {'beta0': beta, 'eps': 0}
    sol_o = solve_ivp(sirc, (0, T_sim), y0_ts, method='DOP853', args=(po,),
                      rtol=1e-8, atol=1e-10,
                      max_step=1/365, t_eval=t_eval)
    if sol_o.status == 0:
        I_orig[i] = np.mean(sol_o.y[1])
    else:
        I_orig[i] = np.nan
        failures_orig.append(beta)

    # --- modified SIRCm ---
    pm = {'beta0': beta, 'eps': 0,
          'delta_prime': delta_prime, 'gamma_prime': gamma_prime}
    sol_m = solve_ivp(sircm, (0, T_sim), y0_ts, method='DOP853', args=(pm,),
                      rtol=1e-8, atol=1e-10,
                      max_step=1/365, t_eval=t_eval)

    if sol_m.status == 0:
        I_mod[i] = np.mean(sol_m.y[1])
    else:
        I_mod[i] = np.nan
        failures_mod.append(beta)

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(beta_range)}")

print(f"\nOriginal failures: {len(failures_orig)}   Modified failures: {len(failures_mod)}")

# PLOT
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(beta_range, I_orig, 'b-', linewidth=2, label='SIRC')
ax.plot(beta_range, I_mod, 'r-', linewidth=2, label='SIRCm')
ax.set_xlabel('Contact rate β', fontsize=12)
ax.set_ylabel('Endemic prevalence I⁺', fontsize=12)
ax.set_title('Endemic prevalence (Dormand Prince)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

idx_orig = np.argmax(np.nan_to_num(I_orig))
idx_mod  = np.argmax(np.nan_to_num(I_mod))
print(f"\nOriginal: peak at β = {beta_range[idx_orig]:.0f}, I+ = {I_orig[idx_orig]:.6f}")
print(f"Modified: peak at β = {beta_range[idx_mod]:.0f}, I+ = {I_mod[idx_mod]:.6f}")


#%% ===| PLOT 4|==== SIRC ALL COMPARTMENTS FOR COMPARISON

T3 = 20

# === WITH SOLVE_IVP === #
t_eval2 = np.linspace(0, T3, int(T3 * 365) + 1)   # daily grid out to 200

sol_all = solve_ivp(sirc, (0, T3), y0_ts, method='DOP853', args=(p_mod,),
                    rtol=1e-8, atol=1e-10,
                    max_step=1/365, t_eval=t_eval2)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(sol_all.t, sol_all.y[0], 'b-',  linewidth=1, label='S (susceptible)')
ax.plot(sol_all.t, sol_all.y[1], 'r-',  linewidth=1, label='I (infected)')
ax.plot(sol_all.t, sol_all.y[2], 'g-',  linewidth=1, label='R (recovered)')
ax.plot(sol_all.t, sol_all.y[3], '-',   linewidth=1, color='orange', label='C (cross-immune)')
ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('Fraction of population', fontsize=12)
ax.set_title(f'solve_ivp DOP853 (β₀ = {beta00}, no seasonality, {T3} yrs)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0,1)
plt.tight_layout()
plt.show()
plt.close()

# === WITH HAND CODED SOLVER === #

t_mod, y_mod = solve(sirc, y0_ts, (0, T3), 1/365, p_mod)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(t_mod, y_mod[:, 0], 'b-',  linewidth=1, label='S (susceptible)')
ax.plot(t_mod, y_mod[:, 1], 'r-',  linewidth=1, label='I (infected)')
ax.plot(t_mod, y_mod[:, 2], 'g-',  linewidth=1, label='R (recovered)')
ax.plot(t_mod, y_mod[:, 3], '-',   linewidth=1, color='orange', label='C (cross-immune)')

ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('Fraction of population', fontsize=12)
ax.set_title(f'hand coded RK4 (β₀ = {beta00}, no seasonality, {T3} yrs)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()


#%% PLOT THIS AT NIGHT ===| PLOT 3|==== ENDEMIC PREVALENCE WITH DIFF BETA VALUES


# %% ===| ENDEMIC PREVALENCE WITH DIFF BETA VALUES (RK4 SOLVER) |===
"""
Sweep over contact rate β ∈ [100, 1500] and compute steady-state endemic 
prevalence I⁺ for both original SIRC and modified SIRCm models.

Setup:
  - Integrate each model for T_sim = 500 years to reach steady state
  - Average I(t) over the last T_avg = 100 years (transients decayed)
  - Use hand-coded RK4 with fixed step h = 1/365 (daily steps)
  - Add floor on I to prevent numerical extinction artifacts
"""

print(delta_prime)
print(gamma_prime)

beta_range = np.linspace(100, 1500, 300)
T_sim = 500  # total simulation time (years)
T_avg = 100  # averaging window at tail (years)
h = 1 / 365  # RK4 step size (1 day)

I_orig = np.zeros(len(beta_range))
I_mod = np.zeros(len(beta_range))

I_FLOOR = 1e-30  # prevent numerical extinction during deep troughs


def sircm_floored(t, y, p):
    """
    SIRCm with a floor on I to prevent machine-zero extinction.

    When I drops below I_FLOOR, we clamp it. This is a practical safeguard:
    in reality, a disease cannot persist at I < 1e-30 of the population,
    so numeric extinction is physically realistic. But it can cause the ODE
    to behave strangely near zero. The floor keeps I in a sane region.
    """
    S, I, R, C = y
    I = max(I, I_FLOOR)
    b = beta_t(t, p['beta0'], p['eps'])
    inc = b * S * I
    de = p['delta_prime'] * inc
    ge = p['gamma_prime'] * inc
    dS = mu * (1 - S) - b * S * I + ge * C
    dI = b * S * I + sigma * b * C * I - (mu + alpha) * I
    dR = (1 - sigma) * b * C * I + alpha * I - (mu + de) * R
    dC = de * R - b * C * I - (mu + ge) * C
    return np.array([dS, dI, dR, dC])


print("Computing endemic prevalence curves (RK4, T_sim=500 yr)...")
print(f"  β range: {beta_range[0]:.0f} to {beta_range[-1]:.0f}")
print(f"  Grid: {len(beta_range)} points")
print(f"  Averaging over last {T_avg} years of {T_sim} year simulation")
print()

for i, beta in enumerate(beta_range):

    # ============================================
    # Original SIRC (constant δ, γ)
    # ============================================
    po = {'beta0': beta, 'eps': 0}
    t_orig, y_orig = solve(sirc, y0_ts, (0, T_sim), h, po)

    # Extract the last T_avg years
    # At h=1/365, there are ~365 points per year, so T_avg years ≈ T_avg*365 points
    idx_start = max(0, len(t_orig) - int(T_avg * 365))
    I_tail_orig = y_orig[idx_start:, 1]

    # Check for pathological values (NaN or Inf → integration failed)
    if np.any(np.isnan(I_tail_orig)) or np.any(np.isinf(I_tail_orig)):
        I_orig[i] = np.nan
    else:
        I_orig[i] = np.mean(I_tail_orig)

    # ============================================
    # Modified SIRCm (δ, γ ∝ incidence)
    # ============================================
    pm = {'beta0': beta, 'eps': 0,
          'delta_prime': delta_prime, 'gamma_prime': gamma_prime}
    t_mod, y_mod = solve(sircm_floored, y0_ts, (0, T_sim), h, pm)

    # Extract the last T_avg years
    idx_start = max(0, len(t_mod) - int(T_avg * 365))
    I_tail_mod = y_mod[idx_start:, 1]

    # Check for pathological values
    if np.any(np.isnan(I_tail_mod)) or np.any(np.isinf(I_tail_mod)):
        I_mod[i] = np.nan
    else:
        I_mod[i] = np.mean(I_tail_mod)

    # Progress indicator
    if (i + 1) % 30 == 0:
        print(f"  {i + 1}/{len(beta_range)}")

print("Computation complete.\n")

# ============================================
# PLOT: Endemic prevalence vs contact rate
# ============================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(beta_range, I_orig, 'b-', linewidth=2, label='SIRC (constant δ, γ)')
ax.plot(beta_range, I_mod, 'r-', linewidth=2, label='SIRCm (δ, γ ∝ incidence)')
ax.set_xlabel('Contact rate β', fontsize=12)
ax.set_ylabel('Endemic prevalence I⁺', fontsize=12)
ax.set_title('Endemic Prevalence: RK4 Integration (T=500 yr)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

# ============================================
# SUMMARY
# ============================================
idx_orig = np.argmax(np.nan_to_num(I_orig))
idx_mod = np.argmax(np.nan_to_num(I_mod))

print("=" * 70)
print("ENDEMIC PREVALENCE PEAKS")
print("=" * 70)
print(f"Original SIRC:")
print(f"  Peak at β = {beta_range[idx_orig]:7.1f},  I⁺ = {I_orig[idx_orig]:.6f}")
print()
print(f"Modified SIRCm (incidence-driven):")
print(f"  Peak at β = {beta_range[idx_mod]:7.1f},  I⁺ = {I_mod[idx_mod]:.6f}")
print("=" * 70)


#%%

