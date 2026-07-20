import numpy as np
from scipy.optimize import fsolve

try:
    from numba import njit
except ImportError:
    def njit(func):
        return func


# Biological parameters from Casagrandi's paper
MU    = 0.02        # birth/death rate 
ALPHA = 365.0 / 3.0  # recovery rate 
DELTA = 1.0 / 1.61  # R -> C transition rate
GAMMA = 0.35        # C -> S transition rate
SIGMA = 0.07874     # cross-immunity factor
BETA0 = 600.0

Y0 = np.array([0.20, 0.001, 0.499, 0.30])

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


def step(f, t_n, y_n, h, params):
    """Single RK4 step"""
    k1 = h * f(t_n,       y_n,         params)
    k2 = h * f(t_n + h/2, y_n + k1/2,  params)
    k3 = h * f(t_n + h/2, y_n + k2/2,  params)
    k4 = h * f(t_n + h,   y_n + k3,    params)
    return y_n + (1.0/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


def solve(f, y0, t_span, h, params):
    """Integrates the SIRC model using RK4"""
    t_start, t_end = t_span
    N_steps = int(np.round((t_end - t_start) / h))
    t_arr = np.linspace(t_start, t_end, N_steps + 1)
    y_arr = np.empty((N_steps + 1, len(y0)))
    y_arr[0] = y0
    for n in range(N_steps):
        y_arr[n+1] = step(f, t_arr[n], y_arr[n], h, params)
    return t_arr, y_arr


def compute_r0(beta, mu, alpha):
    return beta / (mu + alpha)


def compute_endemic_equilibrium(beta, mu, alpha, delta, gamma, sigma):
    """
    Finds the endemic equilibrium (S*, I*, R*, C*) via fsolve
    """
    R0 = compute_r0(beta, mu, alpha)
    if R0 <= 1:
        return None

    def sirc_rhs(X):
        S, I, R, C = X
        dS = mu*(1-S) - beta*S*I + gamma*C
        dI = beta*S*I + sigma*beta*C*I - (mu+alpha)*I
        dR = (1-sigma)*beta*C*I + alpha*I - (mu+delta)*R
        dC = delta*R - beta*C*I - (mu+gamma)*C
        return [dS, dI, dR, dC]

    guess = [0.2, 0.001, 0.5, 0.3]
    sol = fsolve(sirc_rhs, guess, full_output=True)
    X = sol[0]
    S, I, R, C = X

    if I <= 1e-10:
        return None
    if S < 0 or R < 0 or C < 0:
        return None
    if abs(S + I + R + C - 1.0) > 1e-6:
        return None

    return S, I, R, C


def compute_jacobian(S, I, R, C, beta, mu, alpha, delta, gamma, sigma):
    """Computes the analytical Jacobian of the SIRC system"""
    J = np.array([
        [-mu - beta * I, -beta * S, 0, gamma],
        [beta * I, beta * S + sigma * beta * C - (mu + alpha), 0, sigma * beta * I],
        [0, (1 - sigma) * beta * C + alpha, -(mu + delta), (1 - sigma) * beta * I],
        [0, -beta * C, delta, -beta * I - (mu + gamma)]
    ])
    return J



def stability_analysis(beta, mu, alpha, delta, gamma, sigma):
    """evaluates endemic equilibrium stability via Jacobian eigenvalues"""
    eq = compute_endemic_equilibrium(beta, mu, alpha, delta, gamma, sigma)
    if eq is None:
        return np.nan, None

    S, I, R, C = eq
    J = compute_jacobian(S, I, R, C, beta, mu, alpha, delta, gamma, sigma)
    eigenvalues = np.linalg.eigvals(J)
    max_real = np.max(np.real(eigenvalues))
    return max_real, eigenvalues


def calculate_period(I_arr, sample_years=50, days_per_year=365, tol=1e-5):
    """calculates the period of the attractor using annual samples"""
    sample_steps = sample_years * days_per_year
    I_samples = I_arr[-sample_steps :: days_per_year]
    sorted_I = np.sort(I_samples)
    groups = 1
    for k in range(1, len(sorted_I)):
        if abs(sorted_I[k] - sorted_I[k-1]) > tol + tol * abs(sorted_I[k]):
            groups += 1
    return min(groups, 8)


