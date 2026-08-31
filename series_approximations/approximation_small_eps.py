#!/usr/bin/env python3
"""
Small-Epsilon Taylor Series Expansion for Endemic Prevalence

Computes the 0th, 1st, 2nd, and 3rd order Taylor series approximations of the endemic
equilibrium prevalence I*(epsilon) around epsilon = 0 using implicit differentiation of
the characteristic polynomial reduction, comparing against exact algebraic roots.
Supports both prevalence-driven and transmission-driven model variants.
"""

import sys
from pathlib import Path
import sympy as sp
import numpy as np

# =============================================================================
#                        *** MODEL VARIANT SELECTION ***
# Choose between:
#   'prevalence'   : Prevalence-driven variant (1 + eps * I)
#   'transmission' : Transmission-driven variant (1 + eps * S * I)
# =============================================================================
MODEL_VARIANT = 'prevalence'   # 'prevalence' or 'transmission'

# Add model directories to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if MODEL_VARIANT == 'prevalence':
    sys.path.insert(0, str(REPO_ROOT / "SIRCm_prevalence"))
    from sircmw_I_utils import (
        MU, ALPHA, DELTA, GAMMA, SIGMA, BETA0,
        get_algebraic_equilibria
    )
    SCALE_FACTOR = 0.00114321
elif MODEL_VARIANT in ('transmission', 'infection'):
    sys.path.insert(0, str(REPO_ROOT / "SIRCm_transmission"))
    from sircmw_utils import (
        MU, ALPHA, DELTA, GAMMA, SIGMA, BETA0,
        get_algebraic_equilibria
    )
    SCALE_FACTOR = 0.0002045191
else:
    raise ValueError(f"Unknown MODEL_VARIANT '{MODEL_VARIANT}'. Choose 'prevalence' or 'transmission'.")


def get_algebraic_I(tilde_eps):
    """Find all physical endemic equilibria I* algebraically for a given tilde_eps."""
    eqs = get_algebraic_equilibria(tilde_eps)
    valid_I = sorted([eq[1] for eq in eqs if eq[1] > 1e-6])
    return valid_I


# =============================================================================
# 1. Symbolic Characteristic Polynomial Construction
# =============================================================================
I, eps, Is = sp.symbols('I epsilon Istar')
al, be, ga, de, mu, si = sp.symbols('alpha beta gamma delta mu sigma', positive=True)

if MODEL_VARIANT == 'prevalence':
    # Prevalence-driven polynomial: P_prev(I, eps) / mu
    P = (I**2*ga*de*eps**2*mu*(al + (-1 + I)*be + mu)
         + mu*((al - be + mu)*(ga + mu)*(de + mu)
               + I**2*be**2*(al + mu + de*si)
               + I*be*(-be*mu + de*mu + 2*mu**2 + ga*(de + mu)
                       + al*(ga + de + 2*mu) - be*de*si + de*mu*si))
         + I*eps*mu*((al - be + mu)*(de*mu + ga*(2*de + mu))
                     + I**2*be**2*de*si
                     + I*be*(al*(ga + de) + ga*(2*de + mu)
                             + de*(mu - be*si + mu*si))))
    Q = sp.expand(sp.cancel(P / mu))

else:
    # Transmission-driven polynomial coefficients
    c0 = be**2 * mu**2 * (al - be + mu) * (ga + mu) * (de + mu) * (si - 1) * (ga - de * si)
    c4 = be**2 * ga * de * eps * mu * (
        - al**2 * ga * eps - ga * eps * mu**2 + al * (-2 * ga * eps * mu + be**2 * si)
        + be * si * (be * mu + be * de * si + de * eps * mu * si)
    )
    c3 = be * mu * (
        - al**3 * ga**2 * de * eps**2 - 2 * ga**2 * de * eps**2 * mu**3
        + be * ga * de * eps**2 * mu**2 * (1 + si) * (ga + de*si)
        + be**4 * (si - 1) * (ga - de*si) * (mu + de*si)
        + be**3 * de * eps * si * (ga*mu*(si - 3) - 2*ga*de*si - de*mu*(si - 1)*si)
        - al**2 * ga * eps * (- be*ga*de*eps + 4*ga*de*eps*mu + be**2 * (ga + de - 2*de*si))
        + be**2 * ga * eps * mu * (- ga*mu + ga*de*(si - 2) + de*mu*(4*si - 1) + de**2 * si * (1 - 2*(eps - 1)*si))
        + al * (
            - 5*ga**2*de*eps**2*mu**2 - be**3 * ga*de*eps*si + be**4 * (si - 1) * (ga - de*si)
            + be*ga*de*eps**2*mu * (de*si*(1 + si) + ga*(2 + si))
            + be**2 * ga*eps * (- 2*ga*mu + ga*de*(si - 2) + de**2*si*(1 + si) + de*mu*(6*si - 2))
        )
    )
    c2 = mu * (
        - al**3 * ga**2 * de * eps**2 * mu - ga**2 * de * eps**2 * mu**4
        + be**5 * (si - 1) * (-ga + de*si) * (mu + de*si)
        + be * ga * de * eps**2 * mu**3 * (ga + ga*si + de*si)
        - be**2 * ga * eps * mu**2 * (2*ga*mu + de*mu*(2 - 5*si) + ga*de*(4 + (eps - 2)*si) + de**2*si*(-2 + eps - si + eps*si))
        + be**3 * eps * mu * (ga**2 * (2*de + mu + mu*si) - de**2 * mu * si * (si**2 - 1) + ga*de * (mu - 5*mu*si + de*(eps - 4)*si**2))
        + al**2 * ga * eps * (- 3*ga*de*eps*mu**2 + be*de*eps*mu * (ga + ga*si + de*si) + be**2 * (-2*ga*mu + ga*de*(si - 2) + de**2*si + de*mu*(4*si - 2)))
        + be**4 * (ga**2 * (de + mu) * (si - 1) + de*mu * (si - 1) * si * (-3*mu + de*(-1 + (eps - 2)*si)) + ga * (3*mu**2*(si - 1) + de**2*si*(1 + (eps - 1)*si) - de*mu*(1 + eps*(si - 2)*si - si**2)))
        + al * (
            - 3*ga**2*de*eps**2*mu**3 + be**4 * (ga + de + 3*mu) * (si - 1) * (ga - de*si)
            + 2*be*ga*de*eps**2*mu**2
            - be**2 * ga * eps * mu * (4*ga*mu + de*mu*(4 - 9*si) + ga*de*(6 + (eps - 3)*si) + de**2*si*(-3 + eps - si + eps*si))
            - be**3 * eps * (de**2*mu*(si - 1)*si + ga**2 * (de*(si - 2) - mu*(1 + si)) + ga*de * (de*si*(1 + si) + mu*(-1 + 3*si + si**2)))
        )
    )
    c1 = -be * mu * (
        be**3 * (si - 1) * (ga - de*si) * (ga*(de + mu) + mu*(de + 2*mu + de*si))
        + al**2 * ga * eps * mu * (ga*(mu - de*(si - 2)) + de*(mu - de*si - 2*mu*si))
        + ga * eps * mu**3 * (ga*(mu - de*(si - 2)) + de*(mu - de*si - 2*mu*si))
        - be * eps * mu**2 * (- de**2*mu*(si - 1)*si + ga**2*(2*de + mu + mu*si) - ga*de*(2*de*si**2 + mu*(-1 + 2*si + si**2)))
        + be**2 * mu * (ga**2 * (de + mu) * (2 + (eps - 2)*si) + de*mu * (si - 1) * si * (3*mu + de*(2 - eps + si)) + ga * (- 3*mu**2*(si - 1) + de**2*si*(-2 + eps + 2*si - 2*eps*si) + de*mu*(2 - 3*si - (eps - 1)*si**2)))
        - al * (
            be**2 * (ga*(de + 2*mu) + mu*(2*de + 3*mu)) * (si - 1) * (ga - de*si)
            + 2*ga*eps*mu**2 * (-ga*mu + ga*de*(si - 2) + de**2*si + de*mu*(2*si - 1))
            + be*eps*mu * (-de**2*mu*(si - 1)*si + ga**2*(2*de + mu + mu*si) - ga*de*(2*de*si**2 + mu*(-1 + 2*si + si**2)))
        )
    )
    Q = sp.expand(c0 + c1*I + c2*I**2 + c3*I**3 + c4*I**4)

# -----------------------------------------------------------------------------
# 2. Decomposition in powers of eps
# -----------------------------------------------------------------------------
A = sp.expand(Q.subs(eps, 0))
dA = sp.expand(sp.diff(A, I).subs(I, Is))


def recursion(N):
    """Return [r_1,...,r_N] with h_n = r_n / A'(I*), r_n involving only h_<n."""
    h = sp.symbols('h1:%d' % (N + 1))
    F = sp.expand(Q.subs(I, Is + sum(h[k - 1]*eps**k for k in range(1, N + 1))))
    out = []
    for n in range(1, N + 1):
        eq = sp.expand(F.coeff(eps, n))
        out.append(sp.collect(sp.expand(-eq.subs(h[n - 1], 0)), h))
    return h, out


def coeffs_num(params, istar, N):
    """Compute numerical Taylor series coefficients h_1, ..., h_N."""
    Qn = sp.expand(Q.subs(params))
    dAn = sp.diff(Qn.subs(eps, 0), I).subs(I, istar)
    h = sp.symbols('h1:%d' % (N + 1))
    F = sp.expand(Qn.subs(I, Is + sum(h[k - 1]*eps**k for k in range(1, N + 1)))).subs(Is, istar)
    vals, out = {}, []
    for n in range(1, N + 1):
        known = sp.expand(F.coeff(eps, n)).subs(h[n - 1], 0).subs(vals)
        vals[h[n - 1]] = sp.nsimplify(-known / dAn)
        out.append(vals[h[n - 1]])
    return out


def get_series_I(eps_val, params, istar, N):
    """Return I(eps) = I* + sum_{n=1}^N h_n eps^n."""
    cn = coeffs_num(params, istar, N)
    return istar + sum(cn[k - 1] * eps_val**k for k in range(1, N + 1))


if __name__ == "__main__":
    N = 2
    print(f"=== Small-Epsilon Taylor Expansion ({MODEL_VARIANT.upper()} variant) ===")
    print(f"Truncating series at N={N} terms, we have:")
    c, r = recursion(N)
    for n, rn in enumerate(r, 1):
        print(f"h_{n} = [{rn}] / A'(I*)\n")

    params = {al: ALPHA, be: BETA0, ga: GAMMA, de: DELTA, mu: MU, si: SIGMA}
    istar = get_algebraic_I(0.0)[0]
    cn = coeffs_num(params, istar, N)
    print("Numerical coefficients:")
    for k, v in enumerate(cn, 1):
        print(f"  h_{k} = {float(v):.12g}")

    print("\nComparison against exact algebraic equilibria:")
    test_eps = [1e-4, 0.001, 0.01, 0.1, 1.0, 1.1]
    for te in test_eps:
        e0 = te / SCALE_FACTOR
        series_approx = float(get_series_I(e0, params, istar, N))
        num_roots = get_algebraic_I(float(te))
        if num_roots:
            numerical = num_roots[0]
            err = abs(series_approx - numerical)
            print(f"  tilde_eps={te:<7g}  Taylor={series_approx:.10f}  Exact={numerical:.10f}  |err|={err:.2e}")