import numpy as np

# Baseline model parameters (from Casagrandi)
MU = 0.02
ALPHA = 365.0 / 3.0
DELTA = 1.0 / 1.61
GAMMA = 0.35
SIGMA = 0.07874
BETA0 = 600.0
SI_0 = 0.000178
EQ_TOL = 1e-9

def sircmw(t, y, p):
    """ 
    SIRCmw model with optional seasonal forcing
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
    
    dS = mu * (1.0 - S) - b * S * I + (1.0 + eps2 * S * I) * gamma * C
    dI = b * S * I + sigma * b * C * I - (mu + alpha) * I
    dR = (1.0 - sigma) * b * C * I + alpha * I - mu * R - (1.0 + eps1 * S * I) * delta * R
    dC = (1.0 + eps1 * S * I) * delta * R - b * C * I - mu * C - (1.0 + eps2 * S * I) * gamma * C
    
    return np.array([dS, dI, dR, dC])

def sircmw_jacobian(y, eps, p=None):
    """ 
    Analytical Jacobian of SIRCmw (with eps1 = eps2 = eps) 
    """
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
        [-mu - b*I + eps*I*gamma*C,      -b*S + eps*S*gamma*C,                       0.0,                          gamma*(1.0 + eps*S*I)      ],
        [b*I,                             b*S + sigma*b*C - (mu + alpha),              0.0,                          sigma*b*I                  ],
        [-eps*I*delta*R,                  (1.0-sigma)*b*C + alpha - eps*S*delta*R,    -(mu + delta*(1.0+eps*S*I)),   (1.0-sigma)*b*I            ],
        [eps*I*delta*R - eps*I*gamma*C,   eps*S*delta*R - b*C - eps*S*gamma*C,         delta*(1.0+eps*S*I),          -(b*I + mu + gamma*(1.0+eps*S*I))],
    ])

def poly_coeffs(beta, mu, alpha, gamma, delta, eps, sigma):
    """
    Coefficients of the 4th-degree characteristic polynomial in I for SIRCmw.
    """
    c0 = (
        beta**2 * mu**2
        * (alpha - beta + mu)
        * (gamma + mu)
        * (delta + mu)
        * (sigma - 1)
        * (gamma - delta * sigma)
    )

    c4 = (
        beta**2 * gamma * delta * eps * mu
        * (
            - alpha**2 * gamma * eps
            - gamma * eps * mu**2
            + alpha * (-2 * gamma * eps * mu + beta**2 * sigma)
            + beta * sigma * (beta * mu + beta * delta * sigma + delta * eps * mu * sigma)
        )
    )

    c3 = beta * mu * (
        - alpha**3 * gamma**2 * delta * eps**2
        - 2 * gamma**2 * delta * eps**2 * mu**3
        + beta * gamma * delta * eps**2 * mu**2 * (1 + sigma) * (gamma + delta*sigma)
        + beta**4 * (sigma - 1) * (gamma - delta*sigma) * (mu + delta*sigma)
        + beta**3 * delta * eps * sigma * (
            gamma*mu*(sigma - 3) - 2*gamma*delta*sigma - delta*mu*(sigma - 1)*sigma
        )
        - alpha**2 * gamma * eps * (
            - beta*gamma*delta*eps
            + 4*gamma*delta*eps*mu
            + beta**2 * (gamma + delta - 2*delta*sigma)
        )
        + beta**2 * gamma * eps * mu * (
            - gamma*mu
            + gamma*delta*(sigma - 2)
            + delta*mu*(4*sigma - 1)
            + delta**2 * sigma * (1 - 2*(eps - 1)*sigma)
        )
        + alpha * (
            - 5*gamma**2*delta*eps**2*mu**2
            - beta**3 * gamma*delta*eps*sigma
            + beta**4 * (sigma - 1) * (gamma - delta*sigma)
            + beta*gamma*delta*eps**2*mu * (delta*sigma*(1 + sigma) + gamma*(2 + sigma))
            + beta**2 * gamma*eps * (
                - 2*gamma*mu
                + gamma*delta*(sigma - 2)
                + delta**2*sigma*(1 + sigma)
                + delta*mu*(6*sigma - 2)
            )
        )
    )

    c2 = mu * (
        - alpha**3 * gamma**2 * delta * eps**2 * mu
        - gamma**2 * delta * eps**2 * mu**4
        + beta**5 * (sigma - 1) * (-gamma + delta*sigma) * (mu + delta*sigma)
        + beta * gamma * delta * eps**2 * mu**3 * (gamma + gamma*sigma + delta*sigma)
        - beta**2 * gamma * eps * mu**2 * (
            2*gamma*mu
            + delta*mu*(2 - 5*sigma)
            + gamma*delta*(4 + (eps - 2)*sigma)
            + delta**2*sigma*(-2 + eps - sigma + eps*sigma)
        )
        + beta**3 * eps * mu * (
            gamma**2 * (2*delta + mu + mu*sigma)
            - delta**2 * mu * sigma * (sigma**2 - 1)
            + gamma*delta * (mu - 5*mu*sigma + delta*(eps - 4)*sigma**2)
        )
        + alpha**2 * gamma * eps * (
            - 3*gamma*delta*eps*mu**2
            + beta*delta*eps*mu * (gamma + gamma*sigma + delta*sigma)
            + beta**2 * (-2*gamma*mu + gamma*delta*(sigma - 2) + delta**2*sigma + delta*mu*(4*sigma - 2))
        )
        + beta**4 * (
            gamma**2 * (delta + mu) * (sigma - 1)
            + delta*mu * (sigma - 1) * sigma * (-3*mu + delta*(-1 + (eps - 2)*sigma))
            + gamma * (
                3*mu**2*(sigma - 1)
                + delta**2*sigma*(1 + (eps - 1)*sigma)
                - delta*mu*(1 + eps*(sigma - 2)*sigma - sigma**2)
            )
        )
        + alpha * (
            - 3*gamma**2*delta*eps**2*mu**3
            + beta**4 * (gamma + delta + 3*mu) * (sigma - 1) * (gamma - delta*sigma)
            + 2*beta*gamma*delta*eps**2*mu**2 * (gamma + gamma*sigma + delta*sigma)
            - beta**2 * gamma * eps * mu * (
                4*gamma*mu
                + delta*mu*(4 - 9*sigma)
                + gamma*delta*(6 + (eps - 3)*sigma)
                + delta**2*sigma*(-3 + eps - sigma + eps*sigma)
            )
            - beta**3 * eps * (
                delta**2*mu*(sigma - 1)*sigma
                + gamma**2 * (delta*(sigma - 2) - mu*(1 + sigma))
                + gamma*delta * (delta*sigma*(1 + sigma) + mu*(-1 + 3*sigma + sigma**2))
            )
        )
    )

    c1 = -beta * mu * (
        beta**3 * (sigma - 1) * (gamma - delta*sigma) * (
            gamma*(delta + mu) + mu*(delta + 2*mu + delta*sigma)
        )
        + alpha**2 * gamma * eps * mu * (
            gamma*(mu - delta*(sigma - 2)) + delta*(mu - delta*sigma - 2*mu*sigma)
        )
        + gamma * eps * mu**3 * (
            gamma*(mu - delta*(sigma - 2)) + delta*(mu - delta*sigma - 2*mu*sigma)
        )
        - beta * eps * mu**2 * (
            - delta**2*mu*(sigma - 1)*sigma
            + gamma**2*(2*delta + mu + mu*sigma)
            - gamma*delta*(2*delta*sigma**2 + mu*(-1 + 2*sigma + sigma**2))
        )
        + beta**2 * mu * (
            gamma**2 * (delta + mu) * (2 + (eps - 2)*sigma)
            + delta*mu * (sigma - 1) * sigma * (3*mu + delta*(2 - eps + sigma))
            + gamma * (
                - 3*mu**2*(sigma - 1)
                + delta**2*sigma*(-2 + eps + 2*sigma - 2*eps*sigma)
                + delta*mu*(2 - 3*sigma - (eps - 1)*sigma**2)
            )
        )
        - alpha * (
            beta**2 * (gamma*(delta + 2*mu) + mu*(2*delta + 3*mu)) * (sigma - 1) * (gamma - delta*sigma)
            + 2*gamma*eps*mu**2 * (-gamma*mu + gamma*delta*(sigma - 2) + delta**2*sigma + delta*mu*(2*sigma - 1))
            + beta*eps*mu * (
                - delta**2*mu*(sigma - 1)*sigma
                + gamma**2*(2*delta + mu + mu*sigma)
                - gamma*delta*(2*delta*sigma**2 + mu*(-1 + 2*sigma + sigma**2))
            )
        )
    )

    return [c0, c1, c2, c3, c4]

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
    
    A = (mu + alpha) / beta
    qa = -eps * sigma * gamma * I_star
    qb =  mu*sigma + beta*sigma*I_star + gamma + eps*gamma*A*I_star
    qc =  mu*(1.0 - A) - beta*A*I_star

    if abs(qa) < 1e-14:       
        C_candidates = [-qc / qb] if abs(qb) > 1e-14 else []
    else:
        disc = qb**2 - 4.0*qa*qc
        if disc < 0.0:
            return None
        sqd = np.sqrt(disc)
        C_candidates = [(-qb + sqd) / (2.0*qa), (-qb - sqd) / (2.0*qa)]

    best, best_res = None, np.inf
    for C in C_candidates:
        if not (-EQ_TOL <= C <= 1.0 + EQ_TOL):
            continue
        C = np.clip(C, 0.0, 1.0)
        
        S = A - sigma * C
        if not (-EQ_TOL <= S <= 1.0 + EQ_TOL):
            continue
        S = np.clip(S, 0.0, 1.0)
        
        R = 1.0 - I_star - S - C
        if not (-EQ_TOL <= R <= 1.0 + EQ_TOL):
            continue
        R = np.clip(R, 0.0, 1.0)
        
        res = abs((1.0 - sigma) * beta * C * I_star + alpha * I_star
                  - R * (mu + (1.0 + eps * S * I_star) * delta))
        if res < best_res:
            best_res, best = res, (S, I_star, R, C)

    return best

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
    a_SI = -b * S + eps2 * S * gamma * C
    a_RI = (1.0 - sigma) * b * C + alpha - eps1 * S * delta * R
    a_CI = eps1 * S * delta * R - b * C - eps2 * S * gamma * C
    
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



