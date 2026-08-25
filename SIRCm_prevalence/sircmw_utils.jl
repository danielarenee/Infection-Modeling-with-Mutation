# Shared utilities for SIRCmw model, equations, and analytical equilibrium solvers (I-feedback formulation).

using LinearAlgebra

# MODEL AND PARAMETERS

const PAR_BASE = (μ = 0.02, α = 365.0/3, δ = 1.0/1.61, γ = 0.35,
                  σ = 0.07874, β0 = 600.0, tilde_eps = 0.01)

const SI_0 = 0.001
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
    
    eps1 = p.tilde_eps / SI_0
    eps2 = p.tilde_eps / SI_0
    if hasproperty(p, :eps1)
        eps1 = p.eps1
    end
    if hasproperty(p, :eps2)
        eps2 = p.eps2
    end
    
    du[1] = p.μ*(1 - S) - b*S*I + (1.0 + eps2*I)*p.γ*C
    du[2] = b*S*I + p.σ*b*C*I - (p.μ + p.α)*I + reg
    du[3] = (1.0 - p.σ)*b*C*I + p.α*I - p.μ*R - (1.0 + eps1*I)*p.δ*R
    du[4] = (1.0 + eps1*I)*p.δ*R - b*C*I - p.μ*C - (1.0 + eps2*I)*p.γ*C

    if length(u) == 6
        du[5] = w1 - 2π*w2 - (w1^2 + w2^2)*w1
        du[6] = 2π*w1 + w2 - (w1^2 + w2^2)*w2
    end
    du
end

# ANALYTICAL EQUILIBRIUM 

# polynomial coefficients for endemic equilibrium I* (symmetric case eps1 = eps2 = eps)
function poly_coeffs(beta, mu, alpha, gamma, delta, eps, sigma)
    a0 = (alpha - beta + mu) * (gamma + mu) * (delta + mu)
    a3 = beta * delta * eps * (gamma * eps + beta * sigma)
    a2 = (
        alpha * (beta + gamma * eps) * (beta + delta * eps)
        + gamma * delta * eps^2 * mu
        + beta^2 * (mu - delta * (eps - 1.0) * sigma)
        + beta * eps * (
            gamma * (2.0 * delta - delta * eps + mu)
            + delta * mu * (1.0 + sigma)
        )
    )
    a1 = (
        delta * eps * mu^2
        + gamma * eps * mu * (2.0 * delta + mu)
        + alpha * beta * (gamma + delta + 2.0 * mu)
        + beta * gamma * (delta - 2.0 * delta * eps + mu - eps * mu)
        + alpha * eps * (delta * mu + gamma * (2.0 * delta + mu))
        - beta^2 * (mu + delta * sigma)
        + beta * mu * (2.0 * mu + delta * (1.0 - eps + sigma))
    )
    return [a0, a1, a2, a3]
end

# constructs companion matrix and gets eigenvalues (aka. roots of the poly)
function poly_roots(coeffs)
    if length(coeffs) == 4
        c0, c1, c2, c3 = coeffs
        if abs(c3) > 1e-15
            a0, a1, a2 = c0/c3, c1/c3, c2/c3
            Comp = [
                0.0  0.0  -a0;
                1.0  0.0  -a1;
                0.0  1.0  -a2
            ]
            return eigvals(Comp)
        else
            return ComplexF64[]
        end
    else
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
            a0, a1, a2 = c0/c3, c1/c3, c2/c3
            Comp = [
                0.0  0.0  -a0;
                1.0  0.0  -a1;
                0.0  1.0  -a2
            ]
            return eigvals(Comp)
        end
    end
end

# recover other compartments from I*
function recover_equilibrium(I_star, beta, mu, alpha, gamma, delta, eps, sigma)
    S0 = (mu + alpha) / beta
    
    # Calculate C*
    P_val = (mu + alpha) * I_star - mu * (1.0 - S0)
    Q_val = (beta * sigma + eps * gamma) * I_star + (gamma + mu * sigma)
    
    if abs(Q_val) < 1e-15
        return nothing
    end
    C = P_val / Q_val
    if !(-EQ_TOL <= C <= 1.0 + EQ_TOL)
        return nothing
    end
    C = clamp(C, 0.0, 1.0)
    
    S = S0 - sigma * C
    if !(-EQ_TOL <= S <= 1.0 + EQ_TOL)
        return nothing
    end
    S = clamp(S, 0.0, 1.0)
    
    denom = mu + (1.0 + eps * I_star) * delta
    if abs(denom) < 1e-15
        return nothing
    end
    R = ((1.0 - sigma) * beta * C * I_star + alpha * I_star) / denom
    if !(-EQ_TOL <= R <= 1.0 + EQ_TOL)
        return nothing
    end
    R = clamp(R, 0.0, 1.0)
    
    # Verify sum
    if abs(S + I_star + R + C - 1.0) < 1e-4
        return (S, I_star, R, C)
    end
    return nothing
end


# builds the Jacobian matrix for SIRCmw
function jacobian_sircmw(u, p)
    S, I, R, C = u
    b = p.β0
    
    eps1 = hasproperty(p, :eps1) ? p.eps1 : (hasproperty(p, :tilde_eps) ? p.tilde_eps / SI_0 : 0.0)
    eps2 = hasproperty(p, :eps2) ? p.eps2 : (hasproperty(p, :tilde_eps) ? p.tilde_eps / SI_0 : 0.0)

    
    # Row 1
    J11 = -p.μ - b*I
    J12 = -b*S + eps2*p.γ*C
    J13 = 0.0
    J14 = (1.0 + eps2*I)*p.γ
    
    # Row 2
    J21 = b*I
    J22 = b*S + p.σ*b*C - (p.μ + p.α)
    J23 = 0.0
    J24 = p.σ*b*I
    
    # Row 3
    J31 = 0.0
    J32 = (1.0 - p.σ)*b*C + p.α - eps1*p.δ*R
    J33 = -p.μ - (1.0 + eps1*I)*p.δ
    J34 = (1.0 - p.σ)*b*I
    
    # Row 4
    J41 = 0.0
    J42 = eps1*p.δ*R - b*C - eps2*p.γ*C
    J43 = (1.0 + eps1*I)*p.δ
    J44 = -b*I - p.μ - (1.0 + eps2*I)*p.γ
    
    return [J11 J12 J13 J14;
            J21 J22 J23 J24;
            J31 J32 J33 J34;
            J41 J42 J43 J44]
end

# Roots sweep helpers
function bisection(f, a, b, tol=1e-9, maxiter=100)
    fa = f(a)
    fb = f(b)
    for _ in 1:maxiter
        c = (a + b) / 2
        fc = f(c)
        if abs(fc) < tol || (b - a)/2 < tol
            return c
        end
        if sign(fc) == sign(fa)
            a = c
            fa = fc
        else
            b = c
            fb = fc
        end
    end
    return (a + b) / 2
end

function get_C(I, eps2, S0, beta0, γ=PAR_BASE.γ, σ=PAR_BASE.σ, μ=PAR_BASE.μ)
    P_val = beta0 * S0 * I - μ * (1.0 - S0)
    Q_val = (beta0 * σ + eps2 * γ) * I + (γ + μ * σ)
    if abs(Q_val) < 1e-15
        return nothing
    end
    C = P_val / Q_val
    if -1e-12 <= C <= 1.0 + 1e-12
        return clamp(C, 0.0, 1.0)
    end
    return nothing
end

function get_endemic_roots(eps1, eps2, beta0; μ=PAR_BASE.μ, α=PAR_BASE.α, δ=PAR_BASE.δ, γ=PAR_BASE.γ, σ=PAR_BASE.σ)
    S0 = (μ + α) / beta0
    if S0 >= 1.0
        return Tuple{Float64,Float64,Float64,Float64}[]
    end
    
    p0 = -μ * (1.0 - S0)
    p1 = μ + α
    
    q0 = γ + μ * σ
    q1 = beta0 * σ + eps2 * γ
    
    A_poly = [
        0.0,
        δ * α * q0,
        δ * α * (q1 + eps1 * q0),
        δ * α * eps1 * q1
    ]
    
    B1 = [
        0.0,
        (1.0 - σ) * beta0 * δ,
        (1.0 - σ) * beta0 * δ * eps1
    ]
    
    t10 = μ + γ
    t11 = beta0 + eps2 * γ
    t20 = μ + δ
    t21 = eps1 * δ
    
    B2 = [
        t10 * t20,
        t10 * t21 + t11 * t20,
        t11 * t21
    ]
    
    B = B1 - B2
    
    PB = [
        p0 * B[1],
        p0 * B[2] + p1 * B[1],
        p0 * B[3] + p1 * B[2],
        p1 * B[3]
    ]
    
    coeffs = A_poly + PB
    roots_poly = poly_roots(coeffs)
    
    roots = Tuple{Float64,Float64,Float64,Float64}[]
    for r in roots_poly
        if abs(imag(r)) < 1e-7 && 1e-10 < real(r) <= 1.0
            root_I = real(r)
            C_star = get_C(root_I, eps2, S0, beta0, γ, σ, μ)
            if C_star !== nothing
                S_star = S0 - σ * C_star
                denom = μ + (1.0 + eps1 * root_I) * δ
                if abs(denom) > 1e-15
                    R_star = ((1.0 - σ) * beta0 * C_star * root_I + α * root_I) / denom
                    if S_star >= -1e-12 && R_star >= -1e-12 && C_star >= -1e-12
                        S_star = clamp(S_star, 0.0, 1.0)
                        R_star = clamp(R_star, 0.0, 1.0)
                        root_I = clamp(root_I, 0.0, 1.0)
                        if !any(abs(root_I - r_existing[2]) < 1e-6 for r_existing in roots)
                            push!(roots, (S_star, root_I, R_star, C_star))
                        end
                    end
                end
            end
        end
    end
    return roots
end
