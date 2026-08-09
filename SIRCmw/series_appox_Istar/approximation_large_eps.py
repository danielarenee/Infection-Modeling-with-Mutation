import sys
import sympy as sp
import numpy as np

# Baseline model parameters (from Casagrandi)
MU = 0.02
ALPHA = 365.0 / 3.0
DELTA = 1.0 / 1.61
GAMMA = 0.35
SIGMA = 0.07874
BETA0 = 600.0
I_0 = 0.00114321
EQ_TOL = 1e-9
BRN = BETA0/(MU+ALPHA) #Basic reproduction number

#Some functions modfied from from sircmw_utils.py (to account for new nonlinear term in the model)
#We'll want to override this  once we settle in a model
def poly_coeffs(beta, mu, alpha, gamma, delta, eps, sigma):
    """Ascending coefficients [k0, k1, k2, k3] of P(I,eps)/mu as a polynomial in I."""
    r = alpha - beta + mu                       # recurs in k0, k1, k2

    k0 = (delta + mu) * (gamma + mu) * r

    k1 = (beta * (alpha * (gamma + delta + 2 * mu)
                  + gamma * (delta + mu)
                  + delta * mu * (1 + sigma)
                  + 2 * mu ** 2
                  - beta * (mu + delta * sigma))
          + eps * r * (delta * mu + gamma * (2 * delta + mu)))

    k2 = (beta ** 2 * (alpha + mu + delta * sigma)
          + eps * beta * (alpha * (gamma + delta)
                          + gamma * (2 * delta + mu)
                          + delta * (mu + mu * sigma - beta * sigma))
          + eps ** 2 * gamma * delta * r)

    k3 = eps * beta * delta * (beta * sigma + eps * gamma)

    return np.array([k0, k1, k2, k3], dtype=float)

def get_algebraic_I(tilde_eps, p=None):
    """
    Find all physically valid endemic equilibria (I^*) algebraically
    for a given tilde_eps by finding roots of the characteristic polynomial
    """
    if p is None:
        p = {}
    beta = p.get('beta0', BETA0)
    mu = p.get('mu', MU)
    alpha = p.get('alpha', ALPHA)
    gamma = p.get('gamma', GAMMA)
    delta = p.get('delta', DELTA)
    sigma = p.get('sigma', SIGMA)
    i_0 = p.get('i_0', I_0)
    
    eps = tilde_eps / i_0
    coeffs = poly_coeffs(beta, mu, alpha, gamma, delta, eps, sigma)
    roots = np.polynomial.polynomial.polyroots(coeffs)
    
    # Keep real roots in (0, 1]
    valid_I = sorted({round(r.real, 8)
                      for r in roots
                      if abs(r.imag) < 1e-6 and 0.0 < r.real <= 1.0})
    return valid_I


"""
Series expansion in eps of the root I(eps) of P(I,eps) = 0 that satisfies I(0) = I*.

P = mu * ( eps^{-2}*C(I) + eps^{-1}*B(I) + A(I) ),   deg_I A = deg_I B = 3,  deg_I C = 2.

Since A(I*) = 0 and A'(I*) != 0, the implicit function theorem gives a unique
analytic branch I(eps) = I* + sum_{n>=1} h_n eps^n, and the h_n are obtained from
a triangular linear recursion.
"""

I, eps, Is, u = sp.symbols('I epsilon Istar u')
al, be, ga, de, mu, si = sp.symbols('alpha beta gamma delta mu sigma', positive=True)

# --------------------------------------------------------------------------- #
# 1. the polynomial, and its split in powers of eps
# --------------------------------------------------------------------------- #
P = ((eps**2)*mu*((al - be + mu)*(ga + mu)*(de + mu)
           + I**2*be**2*(al + mu + de*si)
           + I*be*(-be*mu + de*mu + 2*mu**2 + ga*(de + mu)
                   + al*(ga + de + 2*mu) - be*de*si + de*mu*si))
+ I*eps*mu*((al - be + mu)*(de*mu + ga*(2*de + mu))
                 + I**2*be**2*de*si
                 + I*be*(al*(ga + de) + ga*(2*de + mu)
                         + de*(mu - be*si + mu*si)))
+I**2*ga*de*mu*(al + (-1 + I)*be + mu))
     

     

Q = sp.expand(sp.cancel(P/mu))               # mu factors out of P entirely
A = sp.expand(Q.subs(eps, 0))
B = sp.expand(sp.diff(Q, eps).subs(eps, 0))
C = sp.expand(sp.diff(Q, eps, 2).subs(eps, 0)/2)
assert sp.expand(Q - (A + eps*B + eps**2*C)) == 0

tup = sp.Poly(A, I).all_coeffs()  
print(f"Coefficients of A(I) = {tup}")


# --------------------------------------------------------------------------- #
# 2. symbolic triangular recursion:  h_n in terms of h_1..h_{n-1}
# --------------------------------------------------------------------------- #
def recursion(N):
    """Return [r_1,...,r_N] with h_n = r_n / A'(I*), r_n involving only h_<n."""
    h = sp.symbols('h1:%d' % (N + 1))
    F = sp.expand(Q.subs(I, Is + sum(h[k - 1]*eps**k for k in range(1, N + 1))))
    assert sp.expand(F.coeff(eps, 0) - A.subs(I, Is)) == 0      # = A(I*) = 0
    out = []
    for n in range(1, N + 1):
        eq = sp.expand(F.coeff(eps, n))                         # = A'(I*) h_n + known
        # assert sp.expand(eq.coeff(h[n - 1], 1) - dA) == 0
        out.append(sp.collect(sp.expand(-eq.subs(h[n - 1], 0)), h))
    return h, out

# --------------------------------------------------------------------------- #
# 3. numeric coefficients (substitute parameters first -> fast)
# --------------------------------------------------------------------------- #
def coeffs_num(params, istar, N):
    Qn = sp.expand(Q.subs(params))
    dAn = sp.diff(Qn.subs(eps, 0), I).subs(I, istar)
    h = sp.symbols('h1:%d' % (N + 1))
    F = sp.expand(Qn.subs(I, Is + sum(h[k - 1]*eps**k for k in range(1, N + 1)))
                  ).subs(Is, istar)
    vals, out = {}, []
    for n in range(1, N + 1):
        known = sp.expand(F.coeff(eps, n)).subs(h[n - 1], 0).subs(vals)
        vals[h[n - 1]] = sp.nsimplify(-known/dAn)
        out.append(vals[h[n - 1]])
    return out

def get_series_I(eps, params, istar, N):
    """Return I(eps) = I* + sum_{n=1}^N h_n eps^{-n}."""
    cn = coeffs_num(params, istar, N)
    return istar + sum(cn[k - 1]*eps**(-k) for k in range(1, N + 1))

if __name__ == "__main__":
    N = 3 #Number of terms in the series expansion
    # Analytic expression for the coefficients of the series expansion
    print(f"Truncating series at N={N} terms, we have:")
    c, r = recursion(N)
    for n, rn in enumerate(r, 1):
        print('h_%d = [%s] / A\'(I*)\n' % (n, rn))

    # Numerical check of the series expansion for various tilde_eps values
    params = {al: ALPHA, be: BETA0, ga: GAMMA,
                de: DELTA, mu: MU, si: SIGMA}
    istar = 1-1./BRN#get_algebraic_I(0.0)[0]
    cn = coeffs_num(params, istar, N)
    for k, v in enumerate(cn, 1):
        print(f"h_{k} = {float(v):.12g}")

    for tilde_eps in [1, 5,10,100]:
        e0 = tilde_eps / I_0
        series_approx = get_series_I(e0, params, istar, N).subs(eps, e0)
        numerical = get_algebraic_I(float(tilde_eps))[0]
        print(f"tilde_eps={tilde_eps:<7g} truncated={series_approx:.12f}  Numerical={numerical:.12f}  |err|={abs(series_approx - numerical):.1e}")