
#%%

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp
import time 
from tqdm import tqdm
from sircmw_I_utils import (
    sircmw,
    step,
    solve_rk4,
    _rk4_mean_endemic,
    beta_t,
    infection_eigvec,
    integrate_with_reseeding,
    MU as mu,
    ALPHA as alpha,
    DELTA as delta,
    GAMMA as gamma,
    SIGMA as sigma,
    BETA0 as beta0,
    SI_0
)




def sirc(t, y, p):
    """Original SIRC: delta and gamma are constant"""
    S, I, R, C = y
    b = beta_t(t, p['beta0'], p['eta'])
    dS = mu*(1 - S) - b*S*I + gamma*C
    dI = b*S*I + sigma*b*C*I - (mu + alpha)*I
    dR = (1 - sigma)*b*C*I + alpha*I - (mu + delta)*R
    dC = delta*R - b*C*I - (mu + gamma)*C
    return np.array([dS, dI, dR, dC])


# PARAMETERS
eta = 0

y0_ts = np.array([0.2, 0.001, 0.499, 0.3]) 

def make_params(beta, tilde_eps1=0, tilde_eps2=0, y0=None):
    """Return (y0, p_sirc, p_sircmw) given tilde-scaled epsilons"""
    if y0 is None:
        y0 = y0_ts
    # Scaling factor: choose between I_0 (0.001) and S_0 * I_0 (0.0002)
    # scale_factor = y0[0] * y0[1] # SI_0 scaling
    scale_factor = y0[1] # I_0 scaling

    eps1 = tilde_eps1 / scale_factor
    eps2 = tilde_eps2 / scale_factor

    p_sirc   = {'beta0': beta, 'eta': eta}
    p_sircmw = {'beta0': beta, 'eta': eta,
                'eps1': eps1, 'eps2': eps2,
                'tilde_eps1': tilde_eps1, 'tilde_eps2': tilde_eps2}
    return y0, p_sirc, p_sircmw, eps1, eps2


# INTEGRATION & PREVALENCE CALCULATION

#%% 1. I(t) time series and convergence of extrema

commontildeps = 0.55
y0, p_orig, p_mod,epss1, epss2 = make_params(370, tilde_eps1=commontildeps, tilde_eps2=commontildeps)

if epss1 == epss2:
    print(f"common epsilon = {epss1}")

yrs = 200

# solve SIRC
t_sirc, Y_sirc, n_reseed_sirc = integrate_with_reseeding(sirc, (0, yrs), y0, p_orig,
                                                          method='DOP853', rtol=1e-6, atol=1e-9)

# solve SIRCmw
t_sircmw, Y_sircmw, n_reseed_sircmw = integrate_with_reseeding(sircmw, (0, yrs), y0, p_mod,
                                                                method='DOP853', rtol=1e-6, atol=1e-9)

# extract prevalence I(t) for both
I_sirc   = Y_sirc[1, :]
I_sircmw = Y_sircmw[1, :]

avg_mask_sirc   = t_sirc   >= (yrs - 10)
avg_mask_sircmw = t_sircmw >= (yrs - 10)
meanprevsircmw = I_sircmw[avg_mask_sircmw].mean()
print(f"Mean prevalence (last 10 yrs) — SIRC: {I_sirc[avg_mask_sirc].mean():.4f}, SIRCmw: {meanprevsircmw:.4f}")
print(f"Reseed events — SIRC: {n_reseed_sirc}, SIRCmw: {n_reseed_sircmw}")

# plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t_sirc,   I_sirc,   label='SIRC',   linewidth=1, color='b', alpha=0.7)
ax.plot(t_sircmw, I_sircmw, label='SIRCmw', linewidth=1, color='r', alpha=0.7)
ax.set_xlabel('Time (years)')
ax.set_ylabel('Prevalence I(t)')
#ax.set_yscale('log')
ax.legend()
ax.set_ylim(0, 0.02)

ax.grid(True, alpha=0.3)
ax.set_title(f'Prevalence Comparison: SIRC vs SIRCmw (common tilde eps = {round(p_mod["tilde_eps1"],2)}) mean prev = {round(meanprevsircmw, 4)}')
plt.tight_layout()
plt.show()

#%% Convergence

#  1. locate extrema via finite differences 
# we make two passes of np.diff to detect sign changes and get the sequence of inf and sup
def find_extrema(t, I):
    # forward difference I[i+1]-I[i] to aproximate the derivative
    dI = np.diff(I)
    sign = np.sign(dI) # keep the sign 
    dsign = np.diff(sign) # diff of the sign array to detect flips

    loc_max_idx = np.where(dsign < 0)[0] + 1  # peaks
    loc_min_idx = np.where(dsign > 0)[0] + 1  # valleys 

    # we return times and prevalence values at each detected inf/sup
    return (t[loc_max_idx], I[loc_max_idx],
            t[loc_min_idx], I[loc_min_idx])

# apply to both models...
t_sup_sirc,   I_sup_sirc,   t_inf_sirc,   I_inf_sirc   = find_extrema(t_sirc,   I_sirc)
t_sup_sircmw, I_sup_sircmw, t_inf_sircmw, I_inf_sircmw = find_extrema(t_sircmw, I_sircmw)


# 2. check for convergence
# for each model we compare early vs late extrema 
# we hope for peaks to decrease and valleys to increase over time
def extrema_summary(label, t_sup, I_sup, t_inf, I_inf, n=5):
    print(f"\n── {label} ──────────────────────────────────")
    for name, t_ex, I_ex, good_slope in [('SUP (local max)', t_sup, I_sup, -1),
                                          ('INF (local min)', t_inf, I_inf, +1)]:

        # avg the first and last n extrema 
        early = I_ex[:n].mean()
        late  = I_ex[-n:].mean()

        # fit a straight line to log(I) vs t over the whole sequence
        # slope <0 means peaks are decaying and >0 means valleys are increasing
        slope, _ = np.polyfit(t_ex, np.log(I_ex), 1)

        # check sign: good_slope is -1 for sup and +1 for inf
        # np.sign(slope)==0 means not converging
        direction = "converging!" if np.sign(slope) == good_slope else "diverging"

        print(f"  {name}: first {n} mean = {early:.4e}  |  last {n} mean = {late:.4e}  "
              f"|  log slope = {slope:+.2e}  [{direction}]")

    # 3. Envelope gap = sup - inf
    # if the system is approaching a fixed point the oscillation amplitud shrinks and this gap should go to 0
    # letscompare the mean of the first n suprema minus first n infima (early)
    gap_early = I_sup[:n].mean() - I_inf[:n].mean()
    gap_late  = I_sup[-n:].mean() - I_inf[-n:].mean()
    print(f"  Envelope gap (sup−inf):  early = {gap_early:.4e}  →  late = {gap_late:.4e}  "
          f"({'narrowing!' if gap_late < gap_early else 'widening'})")

extrema_summary('SIRC',   t_sup_sirc,   I_sup_sirc,   t_inf_sirc,   I_inf_sirc)
extrema_summary('SIRCmw', t_sup_sircmw, I_sup_sircmw, t_inf_sircmw, I_inf_sircmw)


# plot extrema sequences with log-linear trend lines 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, label, t_sup, I_sup, t_inf, I_inf, color in [
    (axes[0], 'SIRC',   t_sup_sirc,   I_sup_sirc,   t_inf_sirc,   I_inf_sirc,   'b'),
    (axes[1], 'SIRCmw', t_sup_sircmw, I_sup_sircmw, t_inf_sircmw, I_inf_sircmw, 'r'),
]:
    # scatter the individual extrema so we can see their evolution over time
    ax.plot(t_sup, I_sup, 'v', color=color, markersize=4, alpha=0.7, label='local max (sup)')
    ax.plot(t_inf, I_inf, '^', color=color, markersize=4, alpha=0.5, label='local min (inf)')
    ax.set_yscale('log')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Prevalence I(t)')
    ax.set_title(f'{label}: sup/inf sequence\n(converging → endemic eq.)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
plt.suptitle(f'common tilde eps = {commontildeps}', fontsize=12)
plt.tight_layout()
plt.show()



#%% 2. all compartments

b = 700
tilde_epss = 1.2
t_span_20 = (0, 20)

_, _, p_r0, _, _ = make_params(b, tilde_eps1=tilde_epss, tilde_eps2=tilde_epss)

sol_mod = solve_ivp(sircmw, t_span_20, y0, args=(p_r0,),
                    method='DOP853',
                    rtol=1e-6, atol=1e-9)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(sol_mod.t, sol_mod.y[0], 'b-',  linewidth=1, label='S (susceptible)')
ax.plot(sol_mod.t, sol_mod.y[1], 'r-',  linewidth=1, label='I (infected)')
ax.plot(sol_mod.t, sol_mod.y[2], 'g-',  linewidth=1, label='R (recovered)')
ax.plot(sol_mod.t, sol_mod.y[3], '-',   linewidth=1, color='orange', label='C (cross-immune)')

ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('Fraction of population', fontsize=12)
ax.set_title(f'SIRCmw all compartments (β₀={b}, tilde eps:{tilde_epss})', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()


#%% 3. I(t) time series with different R_0

denom = mu + alpha  #aprox 121.69
betas = [105,600]

tilde_epss = 0.5

t_span_20 = (0, 20)

colors = ['steelblue', 'firebrick']

fig, ax = plt.subplots(figsize=(10, 5))
for b, color in zip(betas, colors):
    R0_actual = b / denom
    _, _, p_r0, _, _ = make_params(b, tilde_eps1=tilde_epss, tilde_eps2=tilde_epss)
    sol = solve_ivp(sircmw, t_span_20, y0, args=(p_r0,),
                    method='DOP853',
                    rtol=1e-6, atol=1e-9)
    ax.plot(sol.t, sol.y[1], label=f'β₀={b}  ($R_0$={R0_actual:.2f})', color=color, linewidth=2)

ax.set_xlabel('Time (years)')
ax.set_ylabel('Prevalence I(t)')
ax.set_title(f'SIRCmw — I(t) for different $R_0$ (20 years, tilde eps = {tilde_epss})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% 4. I(t) Varying beta
b = 125
tilde_epss = 2

r0 = b / (mu + alpha)
print(f"R_0: {r0}")

_, _, p_r0, _, _ = make_params(b, tilde_eps1=tilde_epss, tilde_eps2=tilde_epss)

t_sol, Y_sol, n_reseed = integrate_with_reseeding(sircmw, (0, 20), y0, p_r0,
                                                   method='DOP853', rtol=1e-6, atol=1e-9)
print(f"Reseed events: {n_reseed}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_sol, Y_sol[1], label=f'β₀={b}', linewidth=2)

ax.set_xlabel('Time (years)')
ax.set_ylabel('Prevalence I(t)')
ax.set_title(f'SIRCmw I(t) (20 years, tilde eps = {tilde_epss})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



#%% 5. Endemic prevalence curve: SIRC vs SIRCmw

T_SIM   = 200.0
AVG_YRS = 50.0
H       = 1.0 / 365.0

tilde_eps_fixed = 0.15
N_beta          = 60
beta_vals       = np.linspace(100.0, 1000.0, N_beta)
R0_thresh       = mu + alpha

I_sirc_B   = np.empty(N_beta)
I_sircmw_B = np.empty(N_beta)

for i, b in enumerate(beta_vals):
    _, p_sirc, p_mw, _, _ = make_params(b, tilde_eps1=tilde_eps_fixed, tilde_eps2=tilde_eps_fixed)
    I_sirc_B[i]   = _rk4_mean_endemic(sirc,   y0, p_sirc, T_SIM, H, AVG_YRS)
    I_sircmw_B[i] = _rk4_mean_endemic(sircmw, y0, p_mw,   T_SIM, H, AVG_YRS)

fig, ax = plt.subplots(figsize=(9, 5))
ax.axvline(R0_thresh, color='k', linestyle=':', linewidth=1,
           label=fr'$R_0=1$ ($\beta\approx{R0_thresh:.1f}$)')
ax.plot(beta_vals, I_sirc_B,   'b-', linewidth=1.5, label='SIRC')
ax.plot(beta_vals, I_sircmw_B, 'r-', linewidth=1.5, label='SIRCmw')
ax.set_xlabel(r'$\beta_0$', fontsize=12)
ax.set_ylabel(r'Mean endemic prevalence', fontsize=12)
ax.set_title(fr'Endemic prevalence vs $\beta_0$  (tilde eps ={tilde_eps_fixed}', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% INVESTIGATING THE THRESHOLD...

# Replicating carlos' result with many simulations
nsim   = 100
tsim   = 200.0
avgyrs = 10.0

eps_vals = np.linspace(0, 2, nsim)
mean_eps = np.empty((4, nsim))  # rows: S, I, R, C
amp_eps  = np.empty((4, nsim))

for i, te in tqdm(enumerate(eps_vals), total=nsim, desc="eps sweep"):
    _, _, p_mw, _, _ = make_params(600, tilde_eps1=te, tilde_eps2=te)
    t_sol, Y_sol, _ = integrate_with_reseeding(sircmw, (0, tsim), y0, p_mw,
                                                method='DOP853', rtol=1e-6, atol=1e-9,
                                                max_step=1/365)
    mask = t_sol >= (tsim - avgyrs)
    t_win = t_sol[mask]
    dt = t_win[-1] - t_win[0]
    for k in range(4):
        w = Y_sol[k, mask]
        mean_eps[k, i] = np.trapezoid(w, t_win) / dt
        amp_eps[k, i]  = w.max() - w.min()

print("Done.")

np.savez("sircmw_eps_sweep.npz",
         eps_vals=eps_vals, mean_eps=mean_eps, amp_eps=amp_eps,
         nsim=nsim, tsim=tsim, avgyrs=avgyrs)
print("Results saved to sircmw_eps_sweep.npz")

fig1, ax1 = plt.subplots(figsize=(9, 5))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
labels = ['S', 'I', 'R', 'C']
for k, (col, lbl) in enumerate(zip(colors, labels)):
    ax1.plot(eps_vals, mean_eps[k], color=col, linewidth=1.5, label=lbl)
ax1.set_xlabel(r'Common $\tilde{\varepsilon}$', fontsize=12)
ax1.set_ylabel('Mean prevalence (last 10 yrs)', fontsize=12)
ax1.set_title(f'SIRCmw mean prevalence vs tilde eps (β₀=600, {tsim:.0f} yrs)', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(9, 5))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
labels = ['S', 'I', 'R', 'C']
for k, (col, lbl) in enumerate(zip(colors, labels)):
    ax2.plot(eps_vals, amp_eps[k], color=col, linewidth=1.5, label=lbl)
ax2.set_xlabel(r'Common $\tilde{\varepsilon}$', fontsize=12)
ax2.set_ylabel('Amplitude (max − min, last 10 yrs)', fontsize=12)
ax2.set_title(f'SIRCmw compartment amplitudes vs tilde eps (β₀=600, {tsim:.0f} yrs)', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% SAME SWEEP WITH HAND-CODED RK4 (h = 1/365, no reseeding)

h_rk4 = 1 / 365

mean_eps_rk4 = np.empty((4, nsim))
amp_eps_rk4  = np.empty((4, nsim))

for i, te in tqdm(enumerate(eps_vals), total=nsim, desc="eps sweep RK4"):
    _, _, p_mw, _, _ = make_params(600, tilde_eps1=te, tilde_eps2=te)
    t_sol, Y_sol = solve_rk4(sircmw, y0, (0, tsim), h_rk4, p_mw)
    mask  = t_sol >= (tsim - avgyrs)
    t_win = t_sol[mask]
    dt    = t_win[-1] - t_win[0]
    for k in range(4):
        w = Y_sol[mask, k]
        mean_eps_rk4[k, i] = np.trapezoid(w, t_win) / dt
        amp_eps_rk4[k, i]  = w.max() - w.min()

print("Done (RK4).")

np.savez("sircmw_eps_sweep_rk4.npz",
         eps_vals=eps_vals, mean_eps=mean_eps_rk4, amp_eps=amp_eps_rk4,
         nsim=nsim, tsim=tsim, avgyrs=avgyrs)
print("Results saved to sircmw_eps_sweep_rk4.npz")

fig1, ax1 = plt.subplots(figsize=(9, 5))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
labels = ['S', 'I', 'R', 'C']
for k, (col, lbl) in enumerate(zip(colors, labels)):
    ax1.plot(eps_vals, mean_eps_rk4[k], color=col, linewidth=1.5, label=lbl)
ax1.set_xlabel(r'Common $\tilde{\varepsilon}$', fontsize=12)
ax1.set_ylabel('Mean prevalence (last 10 yrs)', fontsize=12)
ax1.set_title(f'SIRCmw mean prevalence vs tilde eps — RK4 h=1/365 (β₀=600, {tsim:.0f} yrs)', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(9, 5))
for k, (col, lbl) in enumerate(zip(colors, labels)):
    ax2.plot(eps_vals, amp_eps_rk4[k], color=col, linewidth=1.5, label=lbl)
ax2.set_xlabel(r'Common $\tilde{\varepsilon}$', fontsize=12)
ax2.set_ylabel('Amplitude (max − min, last 10 yrs)', fontsize=12)
ax2.set_title(f'SIRCmw compartment amplitudes vs tilde eps — RK4 h=1/365 (β₀=600, {tsim:.0f} yrs)', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%%SAME SWEEP WITH RADAU, PASSING THE JACOBIAN AND MAX STEP 1/365, WITH RESEEDING

def sircmw_jac(t, y, p):
    S, I, R, C = y
    b  = beta_t(t, p['beta0'], p['eta'])
    e1 = p['eps1']
    e2 = p['eps2']
    return np.array([
        # ∂/∂[S, I, R, C] of dS
        [-mu - b*I + e2*I*gamma*C,
         -b*S + e2*S*gamma*C,
         0.0,
         gamma*(1.0 + e2*S*I)],
        # ∂/∂[S, I, R, C] of dI
        [b*I,
         b*S + sigma*b*C - (mu + alpha),
         0.0,
         sigma*b*I],
        # ∂/∂[S, I, R, C] of dR
        [-e1*I*delta*R,
         (1.0 - sigma)*b*C + alpha - e1*S*delta*R,
         -(mu + delta*(1.0 + e1*S*I)),
         (1.0 - sigma)*b*I],
        # ∂/∂[S, I, R, C] of dC
        [e1*I*delta*R - e2*I*gamma*C,
         e1*S*delta*R - b*C - e2*S*gamma*C,
         delta*(1.0 + e1*S*I),
         -(b*I + mu + gamma*(1.0 + e2*S*I))],
    ])

mean_eps_radau = np.empty((4, nsim))
amp_eps_radau  = np.empty((4, nsim))

for i, te in tqdm(enumerate(eps_vals), total=nsim, desc="eps sweep Radau"):
    _, _, p_mw, _, _ = make_params(600, tilde_eps1=te, tilde_eps2=te)
    t_sol, Y_sol, _ = integrate_with_reseeding(
        sircmw, (0, tsim), y0, p_mw,
        method='Radau',
        jac=sircmw_jac,
        rtol=1e-6, atol=1e-9,
        max_step=1/365
    )
    mask  = t_sol >= (tsim - avgyrs)
    t_win = t_sol[mask]
    dt    = t_win[-1] - t_win[0]
    for k in range(4):
        w = Y_sol[k, mask]
        mean_eps_radau[k, i] = np.trapezoid(w, t_win) / dt
        amp_eps_radau[k, i]  = w.max() - w.min()

print("Done (Radau).")

np.savez("sircmw_eps_sweep_radau.npz",
         eps_vals=eps_vals, mean_eps=mean_eps_radau, amp_eps=amp_eps_radau,
         nsim=nsim, tsim=tsim, avgyrs=avgyrs)
print("Results saved to sircmw_eps_sweep_radau.npz")

fig1, ax1 = plt.subplots(figsize=(9, 5))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
labels = ['S', 'I', 'R', 'C']
for k, (col, lbl) in enumerate(zip(colors, labels)):
    ax1.plot(eps_vals, mean_eps_radau[k], color=col, linewidth=1.5, label=lbl)
ax1.set_xlabel(r'Common $\tilde{\varepsilon}$', fontsize=12)
ax1.set_ylabel('Mean prevalence (last 10 yrs)', fontsize=12)
ax1.set_title(f'SIRCmw mean prevalence vs tilde eps — Radau + Jacobian (β₀=600, {tsim:.0f} yrs)', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(9, 5))
for k, (col, lbl) in enumerate(zip(colors, labels)):
    ax2.plot(eps_vals, amp_eps_radau[k], color=col, linewidth=1.5, label=lbl)
ax2.set_xlabel(r'Common $\tilde{\varepsilon}$', fontsize=12)
ax2.set_ylabel('Amplitude (max − min, last 10 yrs)', fontsize=12)
ax2.set_title(f'SIRCmw compartment amplitudes vs tilde eps — Radau + Jacobian (β₀=600, {tsim:.0f} yrs)', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% What if we use initial values near to the endemic equilibrium for
# plotting time series for the problematic epsilon values?


y0_0 = np.array([0.2, 0.001, 0.499, 0.3]) # near SIRC end eq 
y0_0_5 = np.array([0.178346, 0.002140, 0.508811, 0.310704])
y0_1 = np.array([0.187644, 0.004014, 0.435686, 0.372656])
#y0_1_5 = np.array([0.172319, 0.015083, 0.446219, 0.366380])

#real numerical 1.5:
#y0_1_5 = np.array([0.184865, 0.300365, 0.286858, 0.227912])


commontildeps = 1.8
y0, p_orig, p_mod,epss1, epss2 = make_params(600, commontildeps, commontildeps, y0_1)

if epss1 == epss2:
    print(f"common epsilon = {epss1}")

yrs = 100

# solve SIRC
t_sirc, Y_sirc, n_reseed_sirc = integrate_with_reseeding(sirc, (0, yrs), y0, p_orig,
                                                          method='DOP853', rtol=1e-6, atol=1e-9)

# solve SIRCmw
t_sircmw, Y_sircmw, n_reseed_sircmw = integrate_with_reseeding(sircmw, (0, yrs), y0, p_mod,
                                                                method='DOP853', rtol=1e-6, atol=1e-9)

# extract prevalence I(t) for both
I_sirc   = Y_sirc[1, :]
I_sircmw = Y_sircmw[1, :]

avg_mask_sirc   = t_sirc   >= (yrs - 10)
avg_mask_sircmw = t_sircmw >= (yrs - 10)
meanprevsircmw = I_sircmw[avg_mask_sircmw].mean()
print(f"Mean prevalence (last 10 yrs) — SIRC: {I_sirc[avg_mask_sirc].mean():.4f}, SIRCmw: {meanprevsircmw:.4f}")
print(f"Reseed events — SIRC: {n_reseed_sirc}, SIRCmw: {n_reseed_sircmw}")

# plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t_sirc,   I_sirc,   label='SIRC',   linewidth=1, color='b', alpha=0.7)
ax.plot(t_sircmw, I_sircmw, label='SIRCmw', linewidth=1, color='r', alpha=0.7)
ax.set_xlabel('Time (years)')
ax.set_ylabel('Prevalence I(t)')
#ax.set_yscale('log')
ax.legend()
#ax.set_ylim(1e-10, 1e-9)

ax.grid(True, alpha=0.3)
ax.set_title(f'Prevalence Comparison: SIRC vs SIRCmw (common tilde eps = {round(p_mod["tilde_eps1"],2)}) mean prev = {round(meanprevsircmw, 4)}')
plt.tight_layout()
plt.show()


#%% Residual check 

y0_ts = y0_1_5

d = np.load("sircmw_eps_sweep.npz")
eps_vals = d["eps_vals"] # tilde_eps from 0 to 2
mean_eps = d["mean_eps"] 

SI_0 = y0_ts[0] * y0_ts[1]  

def sircmw_rhs_autonomous(y, tilde_eps):
    S, I, R, C = y
    eps = tilde_eps / SI_0
    b = beta0  # constnat
    dS = mu*(1 - S) - b*S*I + (1 + eps*S*I)*gamma*C
    dI = b*S*I + sigma*b*C*I - (mu + alpha)*I
    dR = (1 - sigma)*b*C*I + alpha*I - mu*R - (1 + eps*S*I)*delta*R
    dC = (1 + eps*S*I)*delta*R - b*C*I - mu*C - (1 + eps*S*I)*gamma*C
    return np.array([dS, dI, dR, dC])

residuals = np.array([sircmw_rhs_autonomous(mean_eps[:, i], te)
                      for i, te in enumerate(eps_vals)]).T  # (4, N)

fig, ax = plt.subplots(figsize=(9, 5))
for k, (lbl, col) in enumerate(zip(['dS/dt', 'dI/dt', 'dR/dt', 'dC/dt'],
                                    ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])):
    ax.plot(eps_vals, residuals[k], color=col, linewidth=1.2, label=lbl)
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.set_xlabel(r'Common $\tilde{\varepsilon}$', fontsize=12)
ax.set_ylabel(r'Residual', fontsize=12)
ax.set_title(r'SIRCmw residual: long-term mean substituted into RHS', fontsize=12)
ax.legend(fontsize=10)
#ax.set_ylim(-1e-5,1e-5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%


