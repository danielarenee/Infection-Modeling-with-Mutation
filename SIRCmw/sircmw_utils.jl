# Shared utilities for SIRCmw model, equations, and analytical equilibrium solvers.

using LinearAlgebra

# MODEL AND PARAMETERS

const PAR_BASE = (μ = 0.02, α = 365.0/3, δ = 1.0/1.61, γ = 0.35,
                  σ = 0.07874, β0 = 600.0, tilde_eps = 0.01)

const SI_0 = 0.000178
const tilde_eps_MIN, tilde_eps_MAX = 0.0, 2.0
const β0_MIN, β0_MAX = 0.0, 2000.0
const EQ_TOL = 1e-9


function sircmw!(du, u, p, t = 0)
    if length(u) == 4
        S, I, R, C = u
        w1, w2 = 0.0, 0.0
    else
        S, I, R, C, w1, w2 = u
    end

    eta = hasproperty(p, :eta) ? p.eta : (hasproperty(p, :ε) ? p.ε : 0.0)
    reg = hasproperty(p, :reg) ? p.reg : 0.0
    
    b = p.β0 * (1.0 + eta * w1)
    eps = p.tilde_eps / SI_0
    
    du[1] = p.μ*(1 - S) - b*S*I + (1.0 + eps*S*I)*p.γ*C
    du[2] = b*S*I + p.σ*b*C*I - (p.μ + p.α)*I + reg
    du[3] = (1.0 - p.σ)*b*C*I + p.α*I - p.μ*R - (1.0 + eps*S*I)*p.δ*R
    du[4] = (1.0 + eps*S*I)*p.δ*R - b*C*I - p.μ*C - (1.0 + eps*S*I)*p.γ*C

    if length(u) == 6
        du[5] = w1 - 2π*w2 - (w1^2 + w2^2)*w1
        du[6] = 2π*w1 + w2 - (w1^2 + w2^2)*w2
    end
    du
end

# ANALYTICAL EQUILIBRIUM 

# polynomial coefficients for endemic equilibrium I*
function poly_coeffs(beta, mu, alpha, gamma, delta, eps, sigma)
    c0 = beta^2 * mu^2 * (alpha - beta + mu) * (gamma + mu) * (delta + mu) * (sigma - 1) * (gamma - delta * sigma)
    c4 = beta^2 * gamma * delta * eps * mu * (
        - alpha^2 * gamma * eps - gamma * eps * mu^2 + alpha * (-2 * gamma * eps * mu + beta^2 * sigma)
        + beta * sigma * (beta * mu + beta * delta * sigma + delta * eps * mu * sigma)
    )
    c3 = beta * mu * (
        - alpha^3 * gamma^2 * delta * eps^2 - 2 * gamma^2 * delta * eps^2 * mu^3
        + beta * gamma * delta * eps^2 * mu^2 * (1 + sigma) * (gamma + delta*sigma)
        + beta^4 * (sigma - 1) * (gamma - delta*sigma) * (mu + delta*sigma)
        + beta^3 * delta * eps * sigma * (gamma*mu*(sigma - 3) - 2*gamma*delta*sigma - delta*mu*(sigma - 1)*sigma)
        - alpha^2 * gamma * eps * (- beta*gamma*delta*eps + 4*gamma*delta*eps*mu + beta^2 * (gamma + delta - 2*delta*sigma))
        + beta^2 * gamma * eps * mu * (- gamma*mu + gamma*delta*(sigma - 2) + delta*mu*(4*sigma - 1) + delta^2 * sigma * (1 - 2*(eps - 1)*sigma))
        + alpha * (
            - 5*gamma^2*delta*eps^2*mu^2 - beta^3 * gamma*delta*eps*sigma + beta^4 * (sigma - 1) * (gamma - delta*sigma)
            + beta*gamma*delta*eps^2*mu * (delta*sigma*(1 + sigma) + gamma*(2 + sigma))
            + beta^2 * gamma*eps * (- 2*gamma*mu + gamma*delta*(sigma - 2) + delta^2*sigma*(1 + sigma) + delta*mu*(6*sigma - 2))
        )
    )
    c2 = mu * (
        - alpha^3 * gamma^2 * delta * eps^2 * mu - gamma^2 * delta * eps^2 * mu^4
        + beta^5 * (sigma - 1) * (-gamma + delta*sigma) * (mu + delta*sigma)
        + beta * gamma * delta * eps^2 * mu^3 * (gamma + gamma*sigma + delta*sigma)
        - beta^2 * gamma * eps * mu^2 * (2*gamma*mu + delta*mu*(2 - 5*sigma) + gamma*delta*(4 + (eps - 2)*sigma) + delta^2*sigma*(-2 + eps - sigma + eps*sigma))
        + beta^3 * eps * mu * (gamma^2 * (2*delta + mu + mu*sigma) - delta^2 * mu * sigma * (sigma^2 - 1) + gamma*delta * (mu - 5*mu*sigma + delta*(eps - 4)*sigma^2))
        + alpha^2 * gamma * eps * (- 3*gamma*delta*eps*mu^2 + beta*delta*eps*mu * (gamma + gamma*sigma + delta*sigma) + beta^2 * (-2*gamma*mu + gamma*delta*(sigma - 2) + delta^2*sigma + delta*mu*(4*sigma - 2)))
        + beta^4 * (gamma^2 * (delta + mu) * (sigma - 1) + delta*mu * (sigma - 1) * sigma * (-3*mu + delta*(-1 + (eps - 2)*sigma)) + gamma * (3*mu^2*(sigma - 1) + delta^2*sigma*(1 + (eps - 1)*sigma) - delta*mu*(1 + eps*(sigma - 2)*sigma - sigma^2)))
        + alpha * (
            - 3*gamma^2*delta*eps^2*mu^3 + beta^4 * (gamma + delta + 3*mu) * (sigma - 1) * (gamma - delta*sigma)
            + 2*beta*gamma*delta*eps^2*mu^2
            - beta^2 * gamma * eps * mu * (4*gamma*mu + delta*mu*(4 - 9*sigma) + gamma*delta*(6 + (eps - 3)*sigma) + delta^2*sigma*(-3 + eps - sigma + eps*sigma))
            - beta^3 * eps * (delta^2*mu*(sigma - 1)*sigma + gamma^2 * (delta*(sigma - 2) - mu*(1 + sigma)) + gamma*delta * (delta*sigma*(1 + sigma) + mu*(-1 + 3*sigma + sigma^2)))
        )
    )
    c1 = -beta * mu * (
        beta^3 * (sigma - 1) * (gamma - delta*sigma) * (gamma*(delta + mu) + mu*(delta + 2*mu + delta*sigma))
        + alpha^2 * gamma * eps * mu * (gamma*(mu - delta*(sigma - 2)) + delta*(mu - delta*sigma - 2*mu*sigma))
        + gamma * eps * mu^3 * (gamma*(mu - delta*(sigma - 2)) + delta*(mu - delta*sigma - 2*mu*sigma))
        - beta * eps * mu^2 * (- delta^2*mu*(sigma - 1)*sigma + gamma^2*(2*delta + mu + mu*sigma) - gamma*delta*(2*delta*sigma^2 + mu*(-1 + 2*sigma + sigma^2)))
        + beta^2 * mu * (gamma^2 * (delta + mu) * (2 + (eps - 2)*sigma) + delta*mu * (sigma - 1) * sigma * (3*mu + delta*(2 - eps + sigma)) + gamma * (- 3*mu^2*(sigma - 1) + delta^2*sigma*(-2 + eps + 2*sigma - 2*eps*sigma) + delta*mu*(2 - 3*sigma - (eps - 1)*sigma^2)))
        - alpha * (
            beta^2 * (gamma*(delta + 2*mu) + mu*(2*delta + 3*mu)) * (sigma - 1) * (gamma - delta*sigma)
            + 2*gamma*eps*mu^2 * (-gamma*mu + gamma*delta*(sigma - 2) + delta^2*sigma + delta*mu*(2*sigma - 1))
            + beta*eps*mu * (-delta^2*mu*(sigma - 1)*sigma + gamma^2*(2*delta + mu + mu*sigma) - gamma*delta*(2*delta*sigma^2 + mu*(-1 + 2*sigma + sigma^2)))
        )
    )
    return [c0, c1, c2, c3, c4]
end

# constructs companion matrix and gets eigenvalues (aka. roots of the poly)
function poly_roots(coeffs)
    c0, c1, c2, c3, c4 = coeffs
    if abs(c4) > 1e-11
        a0, a1, a2, a3 = c0/c4, c1/c4, c2/c4, c3/c4
        Comp = [ 
            0.0  0.0  0.0  -a0;
            1.0  0.0  0.0  -a1;
            0.0  1.0  0.0  -a2;
            0.0  0.0  1.0  -a3
        ]
        return eigvals(Comp)
    else
        # for eps = 0, its a cubic polynomial (c4 = 0, c3 != 0)
        a0, a1, a2 = c0/c3, c1/c3, c2/c3
        Comp = [
            0.0  0.0  -a0;
            1.0  0.0  -a1;
            0.0  1.0  -a2
        ]
        return eigvals(Comp)
    end
end

# recover other compartments from I*
function recover_equilibrium(I_star, beta, mu, alpha, gamma, delta, eps, sigma)
    A = (mu + alpha) / beta
    qa = -eps * sigma * gamma * I_star
    qb =  mu*sigma + beta*sigma*I_star + gamma + eps*gamma*A*I_star
    qc =  mu*(1.0 - A) - beta*A*I_star

    if abs(qa) < 1e-14 
        C_candidates = abs(qb) > 1e-14 ? [-qc / qb] : Float64[] 
    else 
        disc = qb^2 - 4.0*qa*qc
        if disc < 0.0
            return nothing
        end
        sqd = sqrt(disc)
        C_candidates = [(-qb + sqd) / (2.0*qa), (-qb - sqd) / (2.0*qa)]
    end

    best, best_res = nothing, Inf
    for C in C_candidates
        if !(-EQ_TOL <= C <= 1.0 + EQ_TOL) # valid region
            continue
        end
        C = clamp(C, 0.0, 1.0)
        S = A - sigma * C
        R = 1.0 - I_star - S - C
        if !(-EQ_TOL <= S <= 1.0 + EQ_TOL && -EQ_TOL <= R <= 1.0 + EQ_TOL)
            continue
        end
        S = clamp(S, 0.0, 1.0)
        R = clamp(R, 0.0, 1.0)
        res = abs((1 - sigma)*beta*C*I_star + alpha*I_star - R*(mu + (1.0 + eps*S*I_star)*delta))
        if res < best_res
            best_res, best = res, (S, I_star, R, C)
        end
    end
    return best
end


# builds the Jacobian matrix for SIRCmw
function jacobian_sircmw(u, p)
    S, I, R, C = u
    b = p.β0
    eps = p.tilde_eps / SI_0
    
    # Row 1
    J11 = -p.μ - b*I + eps*I*p.γ*C
    J12 = -b*S + eps*S*p.γ*C
    J13 = 0.0
    J14 = (1.0 + eps*S*I)*p.γ
    
    # Row 2
    J21 = b*I
    J22 = b*S + p.σ*b*C - (p.μ + p.α)
    J23 = 0.0
    J24 = p.σ*b*I
    
    # Row 3
    J31 = -eps*I*p.δ*R
    J32 = (1.0 - p.σ)*b*C + p.α - eps*S*p.δ*R
    J33 = -p.μ - (1.0 + eps*S*I)*p.δ
    J34 = (1.0 - p.σ)*b*I
    
    # Row 4
    J41 = eps*I*p.δ*R - eps*I*p.γ*C
    J42 = eps*S*p.δ*R - b*C - eps*S*p.γ*C
    J43 = (1.0 + eps*S*I)*p.δ
    J44 = -b*I - p.μ - (1.0 + eps*S*I)*p.γ
    
    return [J11 J12 J13 J14;
            J21 J22 J23 J24;
            J31 J32 J33 J34;
            J41 J42 J43 J44]
end
