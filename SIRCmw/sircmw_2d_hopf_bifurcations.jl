
# 2D diagram identifying stable and unstable regions + Hopf bifurcation lines 
# across values of tilde epsilon and beta for SIRCmw

using Logging
using Serialization
using Plots
using LinearAlgebra
import BifurcationKit as BK
import BifurcationKit: @optic, @set

ENV["GKSwstype"] = "100"

# MODEL AND PARAMETERS

const PAR_BASE = (μ = 0.02, α = 365.0/3, δ = 1.0/1.61, γ = 0.35,
                  σ = 0.07874, β0 = 600.0, tilde_eps = 0.01)

const SI_0 = 0.000178
const tilde_eps_MIN, tilde_eps_MAX = 0.0, 2.0
const β0_MIN, β0_MAX = 0.0, 2000.0

function sircmw!(du, u, p, t = 0)
    S, I, R, C = u
    b = p.β0
    eps = p.tilde_eps / SI_0
    
    du[1] = p.μ*(1 - S) - b*S*I + (1.0 + eps*S*I)*p.γ*C
    du[2] = b*S*I + p.σ*b*C*I - (p.μ + p.α)*I
    du[3] = (1.0 - p.σ)*b*C*I + p.α*I - p.μ*R - (1.0 + eps*S*I)*p.δ*R
    du[4] = (1.0 + eps*S*I)*p.δ*R - b*C*I - p.μ*C - (1.0 + eps*S*I)*p.γ*C
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
        if !(0.0 <= C <= 1.0) # valid region
            continue
        end
        S = A - sigma * C
        R = 1.0 - I_star - S - C
        if !(0.0 <= S <= 1.0 && 0.0 <= R <= 1.0)
            continue
        end
        res = abs((1 - sigma)*beta*C*I_star + alpha*I_star - R*(mu + (1.0 + eps*S*I_star)*delta))
        if res < best_res
            best_res, best = res, (S, I_star, R, C)
        end
    end
    return best
end

# builds the acobian matrix for SIRCmw
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

# checks stability for a (eps,beta) pair, we use this to color the plot 
function check_stability(tilde_eps, beta0)
    p = PAR_BASE
    
    # check if DFE is stable (R0 < 1)
    # R0 = beta0 / (mu + alpha)
    if beta0 < (p.μ + p.α)
        return 0.0 # DFE is stable
    end
    
    # find endemic equilibrium
    eps = tilde_eps / SI_0
    coeffs = poly_coeffs(beta0, p.μ, p.α, p.γ, p.δ, eps, p.σ)
    roots = poly_roots(coeffs)
    
    endemic_I = [real(r) for r in roots if abs(imag(r)) < 1e-7 && 1e-9 <= real(r) <= 1.0]
    eq = recover_equilibrium(endemic_I[1], beta0, p.μ, p.α, p.γ, p.δ, eps, p.σ)

    # compute jacobian and check eigenvalues
    p_point = (μ = p.μ, α = p.α, δ = p.δ, γ = p.γ, σ = p.σ, β0 = beta0, tilde_eps = tilde_eps)
    J = jacobian_sircmw(eq, p_point)
    evals = eigvals(J)
    
    max_real = maximum(real.(evals))
    if max_real < 0.0
        return 1.0 # stable endemic eq
    else
        return 2.0 # unstable endemic eq
    end
end

# MAIN ANALYSIS PIPELINE

function run_analysis()
    println("Starting 2D Grid Sweep...")
    Nx, Ny = 300, 300 # grid resolution
    tilde_eps_grid = range(tilde_eps_MIN, tilde_eps_MAX, length=Nx)
    beta0_grid = range(β0_MIN, β0_MAX, length=Ny)

    # matrix to hold stability values (0,1,2)
    stability_matrix = zeros(Ny, Nx)

    @time for (j, b0) in enumerate(beta0_grid)
        for (i, te) in enumerate(tilde_eps_grid)
            stability_matrix[j, i] = check_stability(te, b0)
        end
    end

    println("\nStarting 2-Parameter Continuation with BifurcationKit...")

    # first we find the endemic equilibrium at the base parameters to have a starting point
    u0_eq = let
        p = PAR_BASE
        eps = p.tilde_eps / SI_0
        coeffs = poly_coeffs(p.β0, p.μ, p.α, p.γ, p.δ, eps, p.σ)
        roots = poly_roots(coeffs)
        endemic_I = [real(r) for r in roots if abs(imag(r)) < 1e-7 && 0.0 <= real(r) <= 1.0]
        eq = recover_equilibrium(endemic_I[1], p.β0, p.μ, p.α, p.γ, p.δ, eps, p.σ)
        [eq[1], eq[2], eq[3], eq[4]]
    end

    # wraps the model as a BK problem
    prob_eq = BK.ODEBifProblem(sircmw!, u0_eq, PAR_BASE, (@optic _.tilde_eps);
        record_from_solution = (x, p; k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))

    # parameters for the continuation in tilde eps only
    opts_eq = BK.ContinuationPar(
        p_min = tilde_eps_MIN, p_max = tilde_eps_MAX,
        ds = 0.005, dsmin = 1e-6, dsmax = 0.05,
        max_steps = 500,
        newton_options = BK.NewtonPar(tol = 1e-9, max_iterations = 25, linesearch = true),
        detect_bifurcation = 3,
        n_inversion = 6,
        nev = 6)

    # continuation...
    br_eq = BK.continuation(prob_eq, BK.PALC(), opts_eq; verbosity = 0)
    # extracts hopf point indices
    hopf_idx = findall(sp -> sp.type == :hopf, br_eq.specialpoint)

    # now we trace the hopf curve in 2 parameters starting from the first hopf point 
    hopf_branches = []

    if !isempty(hopf_idx)
        # parameters for the 2d continuation 
        opts_hopf2p = BK.ContinuationPar(
            p_min = β0_MIN, p_max = β0_MAX,
            ds = 2.0, dsmin = 1e-5, dsmax = 8.0,
            max_steps = 1200,
            newton_options = BK.NewtonPar(tol = 1e-8, max_iterations = 25, linesearch = true),
            detect_bifurcation = 1,
            nev = 6)

        h_idx = hopf_idx[1] # select first hopf point as seed

        # 1. Forward direction (ds > 0)
        try
            br_up = BK.continuation(
                br_eq, h_idx, (@optic _.β0), opts_hopf2p;
                detect_codim2_bifurcation = 2,
                start_with_eigen = true,
                verbosity = 0,
                bdlinsolver = BK.MatrixBLS())
            println("  Forward branch traced!")
            push!(hopf_branches, br_up)
        catch e
            @warn "  Forward branch tracing failed" exception=e
        end

        # 2. Backward direction (ds < 0)
        try
            println("  Tracing backward (ds < 0)...")
            opts_hopf2p_down = @set opts_hopf2p.ds = -2.0
            br_down = BK.continuation(
                br_eq, h_idx, (@optic _.β0), opts_hopf2p_down;
                detect_codim2_bifurcation = 2,
                start_with_eigen = true,
                verbosity = 0,
                bdlinsolver = BK.MatrixBLS())
            println("  Backward branch traced!")
            push!(hopf_branches, br_down)
        catch e
        end
    end

    # setup color palette
    custom_cmap = cgrad([RGB(0.9, 0.9, 0.9), RGB(0.7, 0.8, 0.95), RGB(0.95, 0.7, 0.75)])

    plt = plot(
        xlabel = "ε̃",
        ylabel = "Contact rate β₀",
        title = "SIRCmw 2D Hopf bifurcation and stability regions",
        xlims = (tilde_eps_MIN, tilde_eps_MAX),
        ylims = (β0_MIN, β0_MAX),
        legend = :topleft,
        size = (800, 600)
    )

    # plot stability regions as heatmap
    heatmap!(plt, tilde_eps_grid, beta0_grid, stability_matrix;
        c = custom_cmap,
        colorbar = :none,
        opacity = 0.85
    )

    # dummy empty traces for labels
    plot!(plt, [NaN], [NaN], seriestype=:shape, fillcolor=RGB(0.9, 0.9, 0.9), label="DFE stable (R₀ < 1)", linecolor=:transparent)
    plot!(plt, [NaN], [NaN], seriestype=:shape, fillcolor=RGB(0.7, 0.8, 0.95), label="Stable endemic equilibrium", linecolor=:transparent)
    plot!(plt, [NaN], [NaN], seriestype=:shape, fillcolor=RGB(0.95, 0.7, 0.75), label="Unstable endemic equilibrium", linecolor=:transparent)

    # overlay hopf lines
    hopf_plotted = false
    for (i, br) in enumerate(hopf_branches)
        if !isnothing(br)
            x_pts = br.branch.tilde_eps[1:5:end]
            y_pts = br.branch.β0[1:5:end]
            if last(br.branch.tilde_eps) != last(x_pts)
                push!(x_pts, last(br.branch.tilde_eps))
                push!(y_pts, last(br.branch.β0))
            end
            
            plot!(plt, x_pts, y_pts;
                label = hopf_plotted ? false : "Hopf bifurcation curve (BifurcationKit)",
                lc = :black,
                lw = 2.5,
                ls = :dash
            )
            hopf_plotted = true
        end
    end

    # find and plot bautin (generalized hopf) points
    gh_plotted = false
    for (i, br) in enumerate(hopf_branches)
        isnothing(br) && continue
        for sp in br.specialpoint
            if sp.type == :gh
                x_val = sp.printsol.tilde_eps
                y_val = sp.printsol.β0
                println("Detected Generalized Hopf (Bautin) point: ε̃ ≈ $(round(x_val; digits=4)), β₀ ≈ $(round(y_val; digits=2))")
                scatter!(plt, [x_val], [y_val];
                    mc = :red,
                    ms = 8,
                    marker = :diamond,
                    label = gh_plotted ? false : "Bautin (Generalized Hopf) point"
                )
                gh_plotted = true
            end
        end
    end

    # save
    output_path = "SIRCmw/sircmw_2d_hopf_bifurcations.png"
    savefig(plt, output_path)
    println("Saved: $output_path")
end

run_analysis()

