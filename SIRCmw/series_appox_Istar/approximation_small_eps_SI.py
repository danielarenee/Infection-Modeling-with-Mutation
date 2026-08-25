import sys
import sympy as sp
import numpy as np
from pathlib import Path

# Add workspace directory to path to import sircmw_utils
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent.parent))

from SIRCmw.sircmw_utils import (
    MU, ALPHA, DELTA, GAMMA, SIGMA, BETA0, SI_0,
    poly_coeffs
)

I_0 = SI_0 # = 0.000178 (SI_0 scaling)

def get_algebraic_I(tilde_eps, p=None):
    """Find all physically valid endemic prevalence roots algebraically from utils"""
    if p is None:
        p = {}
    beta = p.get('beta0', BETA0)
    mu = p.get('mu', MU)
    alpha = p.get('alpha', ALPHA)
    gamma = p.get('gamma', GAMMA)
    delta = p.get('delta', DELTA)
    sigma = p.get('sigma', SIGMA)
    i_0 = p.get('i_0', I_0)
    
    eps_val = tilde_eps / i_0
    coeffs = poly_coeffs(beta, mu, alpha, gamma, delta, eps_val, sigma)
    roots = np.polynomial.polynomial.polyroots(coeffs)
    
    # Keep real roots in (0, 1]
    valid_I = sorted({round(r.real, 8)
                      for r in roots
                      if abs(r.imag) < 1e-6 and 0.0 < r.real <= 1.0})
    return valid_I

# --------------------------------------------------------------------------- #
# 1. the polynomial, and its split in powers of eps
# --------------------------------------------------------------------------- #
I, eps, Is = sp.symbols('I epsilon Istar')
al, be, ga, de, mu_sym, si = sp.symbols('alpha beta gamma delta mu sigma', positive=True)

# Generate symbolic coefficients directly using the function from sircmw_utils
c0, c1, c2, c3, c4 = poly_coeffs(be, mu_sym, al, ga, de, eps, si)
# Cancel out the common factor mu_sym
Q = sp.expand(c4 * I**4 + c3 * I**3 + c2 * I**2 + c1 * I + c0) / mu_sym
Q = sp.expand(Q)

A = sp.expand(Q.subs(eps, 0))
B = sp.expand(sp.diff(Q, eps).subs(eps, 0))
C = sp.expand(sp.diff(Q, eps, 2).subs(eps, 0)/2)
assert sp.expand(Q - (A + eps*B + eps**2*C)) == 0

dA = sp.expand(sp.diff(A, I).subs(I, Is))

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
        assert sp.expand(eq.coeff(h[n - 1], 1) - dA) == 0
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
    """Return I(eps) = I* + sum_{n=1}^N h_n eps^n."""
    cn = coeffs_num(params, istar, N)
    return istar + sum(cn[k - 1]*eps**k for k in range(1, N + 1))

if __name__ == "__main__":
    N = 2 # Number of terms in the series expansion
    print(f"Truncating series at N={N} terms, we have:")
    c, r = recursion(N)
    for n, rn in enumerate(r, 1):
        print('h_%d = [%s] / A\'(I*)\n' % (n, rn))

    # Numerical check of the series expansion for various tilde_eps values
    params = {al: ALPHA, be: BETA0, ga: GAMMA,
              de: DELTA, mu_sym: MU, si: SIGMA}
    istar = get_algebraic_I(0.0)[0]
    cn = coeffs_num(params, istar, N)
    for k, v in enumerate(cn, 1):
        print(f"h_{k} = {float(v):.12g}")

    for tilde_eps in [1.e-4, 0.001, 0.01, 0.1, 1.0, 1.1, 1.2]:
        e0 = tilde_eps / I_0
        series_approx = get_series_I(e0, params, istar, N).subs(eps, e0)
        numerical = get_algebraic_I(float(tilde_eps))[0]
        print(f"tilde_eps={tilde_eps:<7g} truncated={series_approx:.12f}  Numerical={numerical:.12f}  |err|={abs(series_approx - numerical):.1e}")
