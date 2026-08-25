import numpy as np

# Baseline model parameters (from Casagrandi)
MU = 0.02
ALPHA = 365.0 / 3.0
DELTA = 1.0 / 1.61
GAMMA = 0.35
SIGMA = 0.07874
BETA0 = 600.0
SI_0 = 0.001
EQ_TOL = 1e-9

def sircmw(t, y, p):
    """ 
    SIRCmw model with optional seasonal forcing (I-feedback formulation)
    """
    S, I, R, C = y
    beta0 = p.get('beta0', BETA0)
    eta = p.get('eta', 0.0)
    
    # seasonally forced beta
    b = beta0 * (1.0 + eta * np.cos(2.0 * np.pi * t))
    
    # extract eps1 and eps2 (default to a common eps)
    eps = p.get('eps', 0.0)
    eps1 = p.get('eps1', eps)
    eps2 = p.get('eps2', eps)
    
    sigma = p.get('sigma', SIGMA)
    mu = p.get('mu', MU)
    alpha = p.get('alpha', ALPHA)
    delta = p.get('delta', DELTA)
    gamma = p.get('gamma', GAMMA)
    
    dS = mu * (1.0 - S) - b * S * I + (1.0 + eps2 * I) * gamma * C
    dI = b * S * I + sigma * b * C * I - (mu + alpha) * I
    dR = (1.0 - sigma) * b * C * I + alpha * I - mu * R - (1.0 + eps1 * I) * delta * R
    dC = (1.0 + eps1 * I) * delta * R - b * C * I - mu * C - (1.0 + eps2 * I) * gamma * C
    
    return np.array([dS, dI, dR, dC])

def sircmw_jacobian(y, eps1, eps2=None, p=None):
    """ 
    Analytical Jacobian of SIRCmw under new I-feedback (supports distinct eps1 and eps2) 
    """
    if eps2 is None:
        eps2 = eps1
    S, I, R, C = y
    if p is None:
        p = {}
    b = p.get('beta0', BETA0)
    sigma = p.get('sigma', SIGMA)
    mu = p.get('mu', MU)
    alpha = p.get('alpha', ALPHA)
    delta = p.get('delta', DELTA)
    gamma = p.get('gamma', GAMMA)
    
    return np.array([
        [-mu - b*I,      -b*S + eps2*gamma*C,                         0.0,                            gamma*(1.0 + eps2*I)      ],
        [b*I,            b*S + sigma*b*C - (mu + alpha),                0.0,                            sigma*b*I                  ],
        [0.0,            (1.0-sigma)*b*C + alpha - eps1*delta*R,      -(mu + delta*(1.0+eps1*I)),   (1.0-sigma)*b*I            ],
        [0.0,            eps1*delta*R - b*C - eps2*gamma*C,         delta*(1.0+eps1*I),           -(b*I + mu + gamma*(1.0+eps2*I))],
    ])


def poly_coeffs(beta, mu, alpha, gamma, delta, eps, sigma):

    a0 = (alpha - beta + mu) * (gamma + mu) * (delta + mu)
    
    a3 = beta * delta * eps * (gamma * eps + beta * sigma)
    
    a2 = (
        alpha * (beta + gamma * eps) * (beta + delta * eps)
        + gamma * delta * eps**2 * mu
        + beta**2 * (mu - delta * (eps - 1.0) * sigma)
        + beta * eps * (
            gamma * (2.0 * delta - delta * eps + mu)
            + delta * mu * (1.0 + sigma)
        )
    )
    
    a1 = (
        delta * eps * mu**2
        + gamma * eps * mu * (2.0 * delta + mu)
        + alpha * beta * (gamma + delta + 2.0 * mu)
        + beta * gamma * (delta - 2.0 * delta * eps + mu - eps * mu)
        + alpha * eps * (delta * mu + gamma * (2.0 * delta + mu))
        - beta**2 * (mu + delta * sigma)
        + beta * mu * (2.0 * mu + delta * (1.0 - eps + sigma))
    )
    
    return [a0, a1, a2, a3]

def recover_equilibrium(I_star, eps, p=None):
    """
    Given I* (a real root of the polynomial) and eps, return (S*, I*, R*, C*)
    """
    if p is None:
        p = {}
    beta = p.get('beta0', BETA0)
    sigma = p.get('sigma', SIGMA)
    mu = p.get('mu', MU)
    alpha = p.get('alpha', ALPHA)
    delta = p.get('delta', DELTA)
    gamma = p.get('gamma', GAMMA)
    
    eps1 = p.get('eps1', eps)
    eps2 = p.get('eps2', eps)
    
    S0 = (mu + alpha) / beta
    
    # Calculate C*
    P_val = (mu + alpha) * I_star - mu * (1.0 - S0)
    Q_val = (beta * sigma + eps2 * gamma) * I_star + (gamma + mu * sigma)
    
    if abs(Q_val) < 1e-15:
        return None
    C = P_val / Q_val
    
    # Clip C to [0, 1] to handle tiny numerical boundary deviations
    C = np.clip(C, 0.0, 1.0)
    
    # S* = S0 - sigma * C*
    S = S0 - sigma * C
    S = np.clip(S, 0.0, 1.0)
    
    # R* from dR/dt = 0
    denom = mu + (1.0 + eps1 * I_star) * delta
    if abs(denom) < 1e-15:
        return None
    R = ((1.0 - sigma) * beta * C * I_star + alpha * I_star) / denom
    R = np.clip(R, 0.0, 1.0)
    
    # Verify check using original tolerances
    res = abs((1.0 + eps1 * I_star) * delta * R - beta * C * I_star - mu * C - (1.0 + eps2 * I_star) * gamma * C)
    if res < 1e-6:
        return (S, I_star, R, C)
    return None

def plot_sircmw_timeseries(y0=None, p=None, years=100, save_path=None, show=True):
    """
    Simulate and plot the SIRCmw model time series
    """
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt

    if y0 is None:
        y0 = np.array([0.2, 0.001, 0.499, 0.3])
    if p is None:
        p = {'beta0': BETA0, 'sigma': SIGMA}
        
    sol = solve_ivp(sircmw, (0, years), y0, args=(p,),
                     method='DOP853',
                     rtol=1e-6, atol=1e-9,
                     dense_output=True)
                     
    t = sol.t
    I = sol.y[1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, I, linewidth=1, color='r')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Prevalence I(t)')
    beta0 = p.get('beta0', BETA0)
    eps = p.get('eps', 0.0)
    eps1 = p.get('eps1', eps)
    eps2 = p.get('eps2', eps)
    if 'eps1' in p or 'eps2' in p:
        # Show both relative epsilons in the title
        ax.set_title(f'SIRCmw prevalence (β₀={beta0}, eps1={eps1*SI_0:.4f}, eps2={eps2*SI_0:.4f}, {years} yrs)')
    else:
        ax.set_title(f'SIRCmw prevalence (β₀={beta0}, eps={eps:.4f}, {years} yrs)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved timeseries plot to {save_path}")
    if show:
        plt.show()
    return sol

def step(f, t_n, y_n, h, params):
    """ manual RK4 step """
    k1 = h * f(t_n, y_n, params)
    k2 = h * f(t_n + h / 2, y_n + k1 / 2, params)
    k3 = h * f(t_n + h / 2, y_n + k2 / 2, params)
    k4 = h * f(t_n + h, y_n + k3, params)
    return y_n + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def solve_rk4(f, y0, t_span, h, params):
    """ manual RK4 solver """
    t_start, t_end = t_span
    N_steps = int(np.round((t_end - t_start) / h))
    t_arr = np.linspace(t_start, t_end, N_steps + 1)
    y_arr = np.empty((N_steps + 1, len(y0)))
    y_arr[0] = y0
    for n in range(N_steps):
        y_arr[n + 1] = step(f, t_arr[n], y_arr[n], h, params)
    return t_arr, y_arr

def _rk4_mean_endemic(f, y0, p, T_years, h, avg_years):
    """ Returns mean I over the last avg_years of an RK4 simulation """
    N_total = int(np.round(T_years / h))
    N_avg   = int(np.round(avg_years / h))
    t = 0.0
    y = y0.copy()
    I_sum = 0.0
    for i in range(N_total):
        k1 = h * f(t,         y,        p)
        k2 = h * f(t + h/2,   y + k1/2, p)
        k3 = h * f(t + h/2,   y + k2/2, p)
        k4 = h * f(t + h,     y + k3,   p)
        y  = y + (1.0/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
        t += h
        if i >= N_total - N_avg:
            I_sum += y[1]
    return I_sum / N_avg

def beta_t(t, beta0, eta):
    """ Seasonally forced beta0 """
    return beta0 * (1.0 + eta * np.cos(2.0 * np.pi * t))

def infection_eigvec(y, t, p):
    """ Leading eigenvector of the infection subspace (used to reseed when I is near 0) """
    S, _, R, C = y
    beta0 = p.get('beta0', BETA0)
    eta = p.get('eta', 0.0)
    b = beta_t(t, beta0, eta)
    
    eps1 = p.get('eps1', p.get('eps', 0.0))
    eps2 = p.get('eps2', p.get('eps', 0.0))
    
    sigma = p.get('sigma', SIGMA)
    mu = p.get('mu', MU)
    alpha = p.get('alpha', ALPHA)
    delta = p.get('delta', DELTA)
    gamma = p.get('gamma', GAMMA)
    
    lam = b * S + sigma * b * C - (mu + alpha)
    a_SI = -b * S + eps2 * gamma * C
    a_RI = (1.0 - sigma) * b * C + alpha - eps1 * delta * R
    a_CI = eps1 * delta * R - b * C - eps2 * gamma * C
    
    vR = a_RI / (mu + delta + lam)
    vC = (a_CI + delta * vR) / (mu + gamma + lam)
    vS = (a_SI + gamma * vC) / (mu + lam)
    return np.array([vS, 1.0, vR, vC]), lam

def integrate_with_reseeding(rhs, t_span, y0, p, *, threshold=1e-15,
                              I_seed=1e-14, max_events=10000, **solver_kw):
    """
    Integrate ODE system; reseed along infection eigenvector whenever I < threshold
    """
    from scipy.integrate import solve_ivp
    
    t0, tf = t_span

    def hit_floor(t, y, p):
        return y[1] - threshold
    hit_floor.terminal  = True
    hit_floor.direction = -1

    ts, ys = [], []
    y = np.asarray(y0, float)
    t_start, n_ev = t0, 0

    while t_start < tf:
        sol = solve_ivp(rhs, (t_start, tf), y, args=(p,),
                        events=hit_floor, **solver_kw)
        ts.append(sol.t)
        ys.append(sol.y)

        if sol.status == 1 and sol.t_events[0].size:
            t_ev = sol.t_events[0][-1]
            y_ev = sol.y_events[0][-1].copy()
            v, _ = infection_eigvec(y_ev, t_ev, p)
            y = y_ev + (I_seed - y_ev[1]) * v
            y = np.clip(y, 0.0, None)
            y = y / y.sum()
            t_start = t_ev
            n_ev += 1
            if n_ev > max_events:
                break
        else:
            break

    t_arr = np.concatenate(ts)
    Y_arr = np.concatenate(ys, axis=1)
    t_arr, idx = np.unique(t_arr, return_index=True)
    return t_arr, Y_arr[:, idx], n_ev

def get_algebraic_equilibria(tilde_eps, p=None):
    """
    Find all physically valid endemic equilibria (S*, I*, R*, C*) algebraically
    for a given tilde_eps by finding roots of the characteristic polynomial
    Returns a list of tuples: [(S*, I*, R*, C*), ...]
    """
    if p is None:
        p = {}
    beta = p.get('beta0', BETA0)
    mu = p.get('mu', MU)
    alpha = p.get('alpha', ALPHA)
    gamma = p.get('gamma', GAMMA)
    delta = p.get('delta', DELTA)
    sigma = p.get('sigma', SIGMA)
    si_0 = p.get('si_0', SI_0)
    
    eps = tilde_eps / si_0
    coeffs = poly_coeffs(beta, mu, alpha, gamma, delta, eps, sigma)
    roots = np.polynomial.polynomial.polyroots(coeffs)
    
    # Keep real roots in (0, 1]
    valid_I = sorted({round(r.real, 8)
                      for r in roots
                      if abs(r.imag) < 1e-6 and 0.0 < r.real <= 1.0})
                      
    equilibria = []
    for I_star in valid_I:
        eq = recover_equilibrium(I_star, eps, p)
        if eq is not None:
            equilibria.append(eq)
            
    return equilibria

def get_C(I, eps2, S0, beta0, gamma=GAMMA, sigma=SIGMA, mu=MU):
    """Calculates C given I for the new formulation (supports asymmetric eps2)"""
    P_val = beta0 * S0 * I - mu * (1.0 - S0)
    Q_val = (beta0 * sigma + eps2 * gamma) * I + (gamma + mu * sigma)
    if abs(Q_val) < 1e-15:
        return None
    C = P_val / Q_val
    # Physical check: C must be a valid population fraction in [0, 1]
    if 0.0 <= C <= 1.0 + 1e-12:
        return np.clip(C, 0.0, 1.0)
    return None

def get_endemic_roots(eps1, eps2, beta0, mu=MU, alpha=ALPHA, delta=DELTA, gamma=GAMMA, sigma=SIGMA):
    """Finds all physical endemic equilibria analytically using the cubic polynomial reduction (supports eps1 != eps2)"""
    S0 = (mu + alpha) / beta0
    if S0 >= 1.0:
        return []
        
    p0 = -mu * (1.0 - S0)
    p1 = mu + alpha
    
    q0 = gamma + mu * sigma
    q1 = beta0 * sigma + eps2 * gamma
    
    A = np.array([
        0.0,
        delta * alpha * q0,
        delta * alpha * (q1 + eps1 * q0),
        delta * alpha * eps1 * q1
    ])
    
    B1 = np.array([
        0.0,
        (1.0 - sigma) * beta0 * delta,
        (1.0 - sigma) * beta0 * delta * eps1
    ])
    
    t10 = mu + gamma
    t11 = beta0 + eps2 * gamma
    t20 = mu + delta
    t21 = eps1 * delta
    
    B2 = np.array([
        t10 * t20,
        t10 * t21 + t11 * t20,
        t11 * t21
    ])
    
    B = B1 - B2
    
    PB = np.array([
        p0 * B[0],
        p0 * B[1] + p1 * B[0],
        p0 * B[2] + p1 * B[1],
        p1 * B[2]
    ])
    
    coeffs = A + PB
    poly_roots = np.roots(coeffs[::-1])
    
    roots = []
    for r in poly_roots:
        if abs(r.imag) < 1e-7 and 1e-10 < r.real <= 1.0:
            root_I = r.real
            C_star = get_C(root_I, eps2, S0, beta0, gamma, sigma, mu)
            if C_star is not None:
                S_star = S0 - sigma * C_star
                eps1_I_p1 = eps1 * root_I + 1.0
                denom = mu + eps1_I_p1 * delta
                if abs(denom) > 1e-15:
                    R_star = ((1.0 - sigma) * beta0 * C_star * root_I + alpha * root_I) / denom
                    if S_star >= -1e-12 and S_star <= 1.0 + 1e-12 and \
                       R_star >= -1e-12 and R_star <= 1.0 + 1e-12 and \
                       C_star >= -1e-12 and C_star <= 1.0 + 1e-12:
                        S_star = np.clip(S_star, 0.0, 1.0)
                        root_I = np.clip(root_I, 0.0, 1.0)
                        R_star = np.clip(R_star, 0.0, 1.0)
                        C_star = np.clip(C_star, 0.0, 1.0)
                        if not any(abs(root_I - r_existing[1]) < 1e-6 for r_existing in roots):
                            roots.append((S_star, root_I, R_star, C_star))
    return roots




