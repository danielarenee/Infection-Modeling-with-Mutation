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
Q_orig = sp.expand(c4 * I**4 + c3 * I**3 + c2 * I**2 + c1 * I + c0) / mu_sym
Q_orig = sp.expand(Q_orig)

# Since Q_orig is quadratic in eps:
# We divide by eps^2 and set u = 1/eps.
# The large-epsilon polynomial in terms of u (which we call 'eps' here) is:
# Q = Q0 * eps**2 + Q1 * eps + Q2
Q0 = sp.expand(Q_orig.subs(eps, 0)) # u^2 term
Q1 = sp.expand(sp.diff(Q_orig, eps).subs(eps, 0)) # u^1 term
Q2 = sp.expand(sp.diff(Q_orig, eps, 2).subs(eps, 0)/2) # u^0 term

Q = Q0 * eps**2 + Q1 * eps + Q2

# --------------------------------------------------------------------------- #
# 2. symbolic triangular recursion:  h_n in terms of h_1..h_{n-1}
# --------------------------------------------------------------------------- #
def recursion(N):
    """Return [r_1,...,r_N] with h_n = r_n / A'(I*), r_n involving only h_<n."""
    h = sp.symbols('h1:%d' % (N + 1))
    F = sp.expand(Q.subs(I, Is + sum(h[k - 1]*eps**k for k in range(1, N + 1))))
    A = Q2
    assert sp.expand(F.coeff(eps, 0) - A.subs(I, Is)) == 0      # = A(I*) = 0
    dA = sp.expand(sp.diff(A, I).subs(I, Is))
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
    An = sp.expand(Q2.subs(params))
    dAn = sp.diff(An, I).subs(I, istar)
    
    h = sp.symbols('h1:%d' % (N + 1))
    F = sp.expand(Qn.subs(I, Is + sum(h[k - 1]*eps**k for k in range(1, N + 1)))
                  ).subs(Is, istar)
    vals, out = {}, []
    for n in range(1, N + 1):
        known = sp.expand(F.coeff(eps, n)).subs(h[n - 1], 0).subs(vals)
        vals[h[n - 1]] = sp.nsimplify(-known/dAn)
        out.append(vals[h[n - 1]])
    return out

def get_series_I(eps_val, params, istar, N):
    """Return I(eps) = I* + sum_{n=1}^N h_n eps^{-n}."""
    cn = coeffs_num(params, istar, N)
    return istar + sum(cn[k - 1]*eps_val**(-k) for k in range(1, N + 1))

if __name__ == "__main__":
    N = 3 # Number of terms in the series expansion
    
    params = {al: ALPHA, be: BETA0, ga: GAMMA,
              de: DELTA, mu_sym: MU, si: SIGMA}
              
    Q2_num = sp.expand(Q2.subs(params))
    roots = sp.solve(Q2_num, I)
    istar_vals = [float(r) for r in roots if r.is_real and 0.0 < r <= 1.0]
    if not istar_vals:
        print("Error: Could not find valid I_inf* root!")
        sys.exit(1)
    istar = istar_vals[0]
    print(f"Limiting value I_inf* = {istar:.12f}")
    
    cn = coeffs_num(params, istar, N)
    for k, v in enumerate(cn, 1):
        print(f"h_{k} = {float(v):.12g}")

    # Verify against exact roots
    print("\nVerification:")
    for tilde_eps in [1, 5, 10, 100]:
        e0 = tilde_eps / I_0
        series_approx = get_series_I(e0, params, istar, N)
        numerical = get_algebraic_I(float(tilde_eps))[0]
        print(f"tilde_eps={tilde_eps:<7g} truncated={series_approx:.12f}  Numerical={numerical:.12f}  |err|={abs(series_approx - numerical):.1e}")
