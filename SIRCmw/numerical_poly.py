import numpy as np

# coefficients of the 4deg polynomial in I for SIRCmw
def poly_coeffs(beta, mu, alpha, gamma, delta, eps, sigma):

    # c0 (constant)
    c0 = (
        beta**2 * mu**2
        * (alpha - beta + mu)
        * (gamma + mu)
        * (delta + mu)
        * (sigma - 1)
        * (gamma - delta * sigma)
    )

    # c4 - II^4 
    c4 = (
        beta**2 * gamma * delta * eps * mu
        * (
            - alpha**2 * gamma * eps
            - gamma * eps * mu**2
            + alpha * (-2 * gamma * eps * mu + beta**2 * sigma)
            + beta * sigma * (beta * mu + beta * delta * sigma + delta * eps * mu * sigma)
        )
    )

    #  c3 - II^3
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

    #  c2 - II^2
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

    #  c1 - II 
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


def endemic_roots(beta, mu, alpha, gamma, delta, eps, sigma, tol=1e-8):
    coeffs = poly_coeffs(beta, mu, alpha, gamma, delta, eps, sigma)
    roots = np.polynomial.polynomial.polyroots(coeffs)

    print("All roots:")
    for r in roots:
        print(f"  {r}")

    endemic = [
        r.real for r in roots
        if abs(r.imag) < tol and 0.0 <= r.real <= 1.0
    ]
    return sorted(endemic)


params = dict(
    beta=600,
    mu=0.02,
    alpha=365/3,
    gamma=0.35,
    delta=1/1.61,
    eps=200,
    sigma=0.07874,

)

print("Endemic I* values (roots in 0,1):", endemic_roots(**params))
