import numpy as np
from scipy.optimize import fsolve

# from Casagrandi's paper
MU    = 0.02        # birth/death rate
ALPHA = 365.0 / 3.0  # recovery rate 
DELTA = 1.0 / 1.61  # R -> C transition rate
GAMMA = 0.35        # C -> S transition rate
SIGMA = 0.07874     # cross-immunity factor

Y0 = np.array([0.2, 0.001, 0.499, 0.3])

def sirc(t, y, params):
    """
    Original SIRC model from Casagrandi
    """
    S, I, R, C = y
    mu    = params.get('mu', MU)
    alpha = params.get('alpha', ALPHA)
    delta = params.get('delta', DELTA)
    gamma = params.get('gamma', GAMMA)
    sigma = params.get('sigma', SIGMA)
    beta0 = params['beta0']
    eps   = params.get('eps', 0.0)

    beta = beta0 * (1 + eps * np.cos(2 * np.pi * t))

    dSdt = mu * (1 - S) - beta * S * I + gamma * C
    dIdt = beta * S * I + sigma * beta * C * I - (mu + alpha) * I
    dRdt = (1 - sigma) * beta * C * I + alpha * I - (mu + delta) * R
    dCdt = delta * R - beta * C * I - (mu + gamma) * C
    return np.array([dSdt, dIdt, dRdt, dCdt])


def sirc_modified(t, y, params):
    """
    SIRCm with infection-driven cross-immunity erosion
    """
    S, I, R, C = y
    mu          = params.get('mu', MU)
    alpha       = params.get('alpha', ALPHA)
    sigma       = params.get('sigma', SIGMA)
    beta0       = params['beta0']
    eps         = params.get('eps', 0.0)
    delta_prime = params['delta_prime']
    gamma_prime = params['gamma_prime']

    beta = beta0 * (1 + eps * np.cos(2 * np.pi * t))

    delta_final = delta_prime * beta * S * I
    gamma_final = gamma_prime * beta * S * I

    dSdt = mu * (1 - S) - beta * S * I + gamma_final * C
    dIdt = beta * S * I + sigma * beta * C * I - (mu + alpha) * I
    dRdt = (1 - sigma) * beta * C * I + alpha * I - (mu + delta_final) * R
    dCdt = delta_final * R - beta * C * I - (mu + gamma_final) * C
    return np.array([dSdt, dIdt, dRdt, dCdt])


def step(f, t_n, y_n, h, params):
    """Single RK4 step"""
    k1 = h * f(t_n,       y_n,          params)
    k2 = h * f(t_n + h/2, y_n + k1/2,  params)
    k3 = h * f(t_n + h/2, y_n + k2/2,  params)
    k4 = h * f(t_n + h,   y_n + k3,    params)
    return y_n + (1.0/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


def solve(f, y0, t_span, h, params):
    """Integrates the ODE system using fixed-step RK4"""
    t_start, t_end = t_span
    N_steps = int(np.round((t_end - t_start) / h))
    t_arr = np.linspace(t_start, t_end, N_steps + 1)
    y_arr = np.empty((N_steps + 1, len(y0)))
    y_arr[0] = y0
    for n in range(N_steps):
        y_arr[n+1] = step(f, t_arr[n], y_arr[n], h, params)
    return t_arr, y_arr

# CALIBRATION OF delta_prime AND gamma_prime

def _calibrate():
    calib_params = {
        'mu': MU, 'alpha': ALPHA, 'delta': DELTA, 'gamma': GAMMA,
        'sigma': SIGMA, 'beta0': 600, 'eps': 0
    }
    
    def eq_eqs(x):
        return sirc(0.0, x, calib_params)

    eq = fsolve(eq_eqs, np.array([0.2, 0.001, 0.499, 0.3]))
    S_eq, I_eq, _, _ = eq
    incidence_eq = 600 * S_eq * I_eq
    
    delta_prime = DELTA / incidence_eq
    gamma_prime = GAMMA / incidence_eq
    return delta_prime, gamma_prime

DELTA_PRIME, GAMMA_PRIME = _calibrate()
