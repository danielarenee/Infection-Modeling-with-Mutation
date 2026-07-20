import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# PARAMETERS
mu    = 0.02       # birth = death rate
alpha = 365/3      # recovery rate I->R
delta = 1/1.61     # R->C rate (loss of full immunity SIRC)
gamma = 0.35       # C->S rate (loss of partial immunity SIRC)
sigma = 0.07874    # reduced susceptibility of cross-immune class C
beta0 = 600

n_simulations_per_eps = 101
lower_relative_eps = -1
upper_relative_eps = 3


y0_ts = np.array([0.2, 0.001, 0.499, 0.3])  # initial val for S,I,R,C

t_span = (0, 100)  # years
t_eval = np.linspace(0, 100, 5000)

# seasonal forcing param
p_orig ={'beta0': beta0, 'eps': 0,
        'eps1': 0, 'eps2': 0}

def sircmw(t, y, p):
    """
    Modified SIRCm: the immunity-erosion rates are driven by incidence
    """
    S, I, R, C = y
    b = beta_t(t, p['beta0'], p['eps'])
    eps1_SI_p1 = p['eps1']*S*I+1; eps2_SI_p1 = p['eps2']*S*I+1
    dS = mu*(1 - S) - b*S*I + eps2_SI_p1*gamma*C
    dI = b*S*I + sigma*b*C*I - (mu + alpha)*I
    dR = (1 - sigma)*b*C*I + alpha*I - (mu + eps1_SI_p1*delta)*R
    dC = eps1_SI_p1*delta*R - b*C*I - (mu + eps2_SI_p1*gamma)*C
    return np.array([dS, dI, dR, dC])

# seasonal forcing (if eps=0 then there is no seasonal forcing)
def beta_t(t, beta0, eps):
    return beta0 * (1.0 + eps * np.cos(2.0 * np.pi * t))

def infection_eigvec(y, t, p):
    """Right eigenvector of J at I=0 for the infection eigenvalue λ_I,
    normalized so v_I = 1. Automatically sums to zero, so adding ε·v
    preserves S+I+R+C. Returns (v, λ_I)."""
    S, _, R, C = y
    b  = beta_t(t, p['beta0'], p['eps'])
    e1, e2 = p['eps1'], p['eps2']
    lam  = b*S + sigma*b*C - (mu + alpha)                 # growth rate of I
    a_SI = -b*S + e2*S*gamma*C                            # column-I couplings
    a_RI = (1 - sigma)*b*C + alpha - e1*S*delta*R
    a_CI = e1*S*delta*R - b*C - e2*S*gamma*C
    vI = 1.0
    vR = a_RI / (mu + delta + lam)
    vC = (a_CI + delta*vR) / (mu + gamma + lam)
    vS = (a_SI + gamma*vC) / (mu + lam)
    return np.array([vS, vI, vR, vC]), lam

def integrate_with_reseeding(rhs, t_span, y0, p, *, threshold=1e-15,
                             I_seed=1e-13, t_eval=None, max_events=10000,
                             **solver_kw):
    r"""Pushes the trajectory away from the (numerical) disease-free equilibrium whenever it gets too close
    by perturbing it in the direction of the leading eigenvector of the Jacobian at I=0."""
    t0, tf = t_span

    def hit_floor(t, y, p):
        return y[1] - threshold
    hit_floor.terminal  = True
    hit_floor.direction = -1            # only catch I decreasing through the floor

    ts, ys = [], []
    y = np.asarray(y0, float)
    t_start, n_ev = t0, 0

    while t_start < tf:
        te = None
        if t_eval is not None:
            te = t_eval[(t_eval >= t_start) & (t_eval <= tf)]
            if te.size == 0 or te[0] > t_start:
                te = np.concatenate(([t_start], te))

        sol = solve_ivp(rhs, (t_start, tf), y, args=(p,),
                        events=hit_floor, t_eval=te, **solver_kw)
        ts.append(sol.t); ys.append(sol.y)

        if sol.status == 1 and sol.t_events[0].size:      # event fired
            t_ev = sol.t_events[0][-1]
            y_ev = sol.y_events[0][-1].copy()
            v, _ = infection_eigvec(y_ev, t_ev, p)
            y = y_ev + (I_seed - y_ev[1]) * v             # I -> I_seed, along v
            y = np.clip(y, 0.0, None); y = y / y.sum()    # safety net (should not be needed since v should sum to zero)
            t_start = t_ev
            n_ev += 1
            if n_ev > max_events:
                print("max_events reached; stopping"); break
        else:
            break

    t = np.concatenate(ts)
    Y = np.concatenate(ys, axis=1)
    t, idx = np.unique(t, return_index=True)              # drop duplicate stitch points
    return t, Y[:, idx], n_ev

def run_simulation(relative_eps1, relative_eps2):
    #Choosing eps1 and eps2 as something times SI(0), so eps1=1 means that
    #nonlinear and linear contributions to the immunity erosion are of the same order at t=0.
    #This way we see the effect of the perturbation at "small" values of the relative perturbation parameters.
    eps1 = relative_eps1/(y0_ts[0]*y0_ts[1])
    eps2 = relative_eps2/(y0_ts[0]*y0_ts[1])
    p_mod  = {'beta0': beta0, 'eps': 0,
        'eps1': eps1, 'eps2': eps2}
    t_sircmw, Y_sircmw, n2 = integrate_with_reseeding(
    sircmw, t_span, y0_ts, p_mod, t_eval=t_eval,
    method='DOP853', rtol=1e-6, atol=1e-9)
    return t_sircmw, Y_sircmw, n2

def get_prevalence(t_sircmw, Y_sircmw):
    # extract prevalence I(t) for both
    I = Y_sircmw[1, :]   # I is second component
    t = t_sircmw
    #Average over last 10 years to get a stable estimate of prevalence
    last_10_years = t >= (t[-1] - 10)
    avg_prevalence = np.mean(I[last_10_years])
    return avg_prevalence

from joblib import Parallel, delayed

def run_one(relative_eps1, relative_eps2):
    t, Y, n_ev = run_simulation(relative_eps1, relative_eps2)
    return get_prevalence(t, Y), n_ev

if __name__ == "__main__":
    list_eps1 = np.linspace(lower_relative_eps, upper_relative_eps, n_simulations_per_eps)
    list_eps2 = np.linspace(lower_relative_eps, upper_relative_eps, n_simulations_per_eps)

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(run_one)(e1, e2)
        for e1 in list_eps1 for e2 in list_eps2
    )

    prevalences = np.array([r[0] for r in results]).reshape(
        len(list_eps1), len(list_eps2))
    reseed_counts = np.array([r[1] for r in results]).reshape(
        len(list_eps1), len(list_eps2))

    #Save results to file for later plotting
    np.savez("prevalence_results.npz", list_eps1=list_eps1, list_eps2=list_eps2,
             prevalences=prevalences, reseed_counts=reseed_counts)
    fig, ax = plt.subplots(figsize=(8, 6))
    # im = ax.contourf(list_eps2, list_eps1, prevalences, cmap='turbo', levels=50)
    im = ax.pcolormesh(list_eps2, list_eps1, prevalences, cmap='turbo', shading='auto')
    ax.set_xlabel('Relative eps2')   # note: x is eps2, y is eps1
    ax.set_ylabel('Relative eps1')
    ax.set_title('Average Prevalence I(t) over last 10 years')
    fig.colorbar(im, label='Average Prevalence I(t)')
    plt.show()