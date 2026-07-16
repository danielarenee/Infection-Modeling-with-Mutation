
"""
This script creates the 2D diagrams identifying stable and unstable regions + Hopf bifurcation lines
for the SIRCmw model by analyzing the eigenvalues of the Jacobian matrix 

It also generates a csv with the data to plot a 3D bifurcation surface to visualize the Hopf bifurcation 
region across values of tilde epsilon, beta, and sigma with the help of sircmw_3d_hopf_surface.py  
"""

using Plots
using LinearAlgebra
using DelimitedFiles
import BifurcationKit as BK
import BifurcationKit: @optic, @set

ENV["GKSwstype"] = "100"
const SCRIPT_DIR = dirname(@__FILE__)
mkpath(SCRIPT_DIR)  

# MODEL AND PARAMETERS

# These are the base parameters from Casagrandi's paper
const PAR_BASE = (μ = 0.02, α = 365.0/3, δ = 1.0/1.61, γ = 0.35,
                  σ = 0.07874, β0 = 600.0, tilde_eps = 0.01)

const SI_0 = 0.000178
const tilde_eps_MIN, tilde_eps_MAX = 0.0, 2.0 # range for tilde eps
const β0_MIN, β0_MAX = 0.0, 2000.0 # range for beta
const σ_MIN, σ_MAX = 0.0, 0.3 # range for sigma 
const σ_SLICES = range(0.0, 0.3, length=40)  

const EQ_TOL = 1e-9  

# SIRCmw model
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

# Polynomial coefficients for endemic equilibrium I*
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

# Construct companion matrix and gets eigenvalues (roots of the polynomial)
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


# Recover other compartments from I*
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

# Try every real root of the quartic, keeping whichever yields a valid (S,I,R,C) 
# with the lowest residual on the unused equation
function find_endemic_equilibrium(tilde_eps, beta0, sigma)
    p = PAR_BASE
    eps = tilde_eps / SI_0
    coeffs = poly_coeffs(beta0, p.μ, p.α, p.γ, p.δ, eps, sigma)
    roots = poly_roots(coeffs)
    # keep valid roots
    endemic_I = [real(r) for r in roots if abs(imag(r)) < 1e-7 && 1e-9 <= real(r) <= 1.0]

    best, best_res = nothing, Inf
    for I_star in endemic_I
        eq = recover_equilibrium(I_star, beta0, p.μ, p.α, p.γ, p.δ, eps, sigma)
        eq === nothing && continue
        S, I, R, C = eq
        res = abs((1 - sigma)*beta0*C*I + p.α*I - R*(p.μ + (1.0 + eps*S*I)*p.δ))
        if res < best_res
            best_res, best = res, eq
        end
    end
    return best
end

# Jacobian matrix for SIRCmw
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

# Checks stability for a (eps,beta) pair and uses this to color the plot 

function check_stability_full(tilde_eps, beta0, sigma)
    p = PAR_BASE
    if beta0 < (p.μ + p.α)       
        return 0.0 # DFE
    end

    eq = find_endemic_equilibrium(tilde_eps, beta0, sigma)
    eq === nothing && return 0.0

    p_point = (μ = p.μ, α = p.α, δ = p.δ, γ = p.γ, σ = sigma, β0 = beta0, tilde_eps = tilde_eps)
    J = jacobian_sircmw(eq, p_point)
    max_real = maximum(real.(eigvals(J)))
    return max_real < 0.0 ? 1.0 : 2.0 # stable (1) unstable (2)
end

# Fix beta and sigma for the 2d sweeps 
check_stability(tilde_eps, beta0) = check_stability_full(tilde_eps, beta0, PAR_BASE.σ)
check_stability_eps_sigma(tilde_eps, sigma) = check_stability_full(tilde_eps, PAR_BASE.β0, sigma)

# Color palette for stability regions
const COLOR_DFE = RGB(0.9, 0.9, 0.9) # light gray
const COLOR_STABLE = RGB(0.7, 0.8, 0.95) # baby blue
const COLOR_UNSTABLE = RGB(0.95, 0.7, 0.75) # pink

# eatmap plot
function plot_stability_regions!(plt, xgrid, ygrid, matrix)
    mask(val) = map(v -> v == val ? 1.0 : NaN, matrix)

    heatmap!(plt, xgrid, ygrid, mask(0.0); c = cgrad([COLOR_DFE, COLOR_DFE]), colorbar = :none, opacity = 0.85)          # DFE stable
    heatmap!(plt, xgrid, ygrid, mask(1.0); c = cgrad([COLOR_STABLE, COLOR_STABLE]), colorbar = :none, opacity = 0.85)    # stable endemic
    heatmap!(plt, xgrid, ygrid, mask(2.0); c = cgrad([COLOR_UNSTABLE, COLOR_UNSTABLE]), colorbar = :none, opacity = 0.85) # unstable endemic
end

function add_stability_legend!(plt)
    plot!(plt, [NaN], [NaN], seriestype=:shape, fillcolor=COLOR_DFE, label="DFE stable (R₀ < 1)", linecolor=:transparent)
    plot!(plt, [NaN], [NaN], seriestype=:shape, fillcolor=COLOR_STABLE, label="Stable endemic equilibrium", linecolor=:transparent)
    plot!(plt, [NaN], [NaN], seriestype=:shape, fillcolor=COLOR_UNSTABLE, label="Unstable endemic equilibrium", linecolor=:transparent)
end

# builds the endemic equilibrium at parameters p and continues it in tilde_eps 
function continue_in_tilde_eps(p)
    eq = find_endemic_equilibrium(p.tilde_eps, p.β0, p.σ)
    eq === nothing && return nothing, Int[]
    u0_eq = [eq[1], eq[2], eq[3], eq[4]]

    prob_eq = BK.ODEBifProblem(sircmw!, u0_eq, p, (@optic _.tilde_eps);
        record_from_solution = (x, p; k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))

    opts_eq = BK.ContinuationPar(
        p_min = tilde_eps_MIN, p_max = tilde_eps_MAX,
        ds = 0.005, dsmin = 1e-6, dsmax = 0.05, max_steps = 500,
        newton_options = BK.NewtonPar(tol = 1e-9, max_iterations = 25, linesearch = true),
        detect_bifurcation = 3, n_inversion = 6, nev = 6)

    br_eq = BK.continuation(prob_eq, BK.PALC(), opts_eq; verbosity = 0)
    hopf_idx = findall(sp -> sp.type == :hopf, br_eq.specialpoint)
    return br_eq, hopf_idx
end

# MAIN ANALYSIS PIPELINE

# sweep the (tilde_eps, beta0) plane and plot stability regions with the Hopf bifurcation curve
function run_eps_beta_analysis()
    println("Starting 2D Grid Sweep...")
    Nx, Ny = 300, 300 # grid resolution
    tilde_eps_grid = range(tilde_eps_MIN, tilde_eps_MAX, length=Nx)
    beta0_grid = range(β0_MIN, β0_MAX, length=Ny)

    # evaluate local stability at each grid point
    stability_matrix = zeros(Ny, Nx)
    @time for (j, b0) in enumerate(beta0_grid)
        for (i, te) in enumerate(tilde_eps_grid)
            stability_matrix[j, i] = check_stability(te, b0)
        end
    end

    # run 2-parameter continuation in (tilde_eps, beta0) using BifurcationKit
    println("\nStarting 2-Parameter Continuation with BifurcationKit...")
    br_eq, hopf_idx = continue_in_tilde_eps(PAR_BASE)

    hopf_branches = []
    if !isempty(hopf_idx)
        # solver settings for the 2-parameter curve continuation
        opts_hopf2p = BK.ContinuationPar(
            p_min = β0_MIN, p_max = β0_MAX,
            ds = 1.0, dsmin = 1e-5, dsmax = 4.0, max_steps = 2000,
            newton_options = BK.NewtonPar(tol = 1e-8, max_iterations = 25, linesearch = true),
            detect_bifurcation = 1, nev = 6)

        h_idx = hopf_idx[1] # select first hopf point as seed

        # trace the Hopf curve in both directions from the seed
        for (label, ds_sign) in (("Forward", 1.0), ("Backward", -1.0))
            try
                println("  Tracing $label (ds = $(ds_sign*1.0))...")
                opts_dir = @set opts_hopf2p.ds = ds_sign * 1.0
                br = BK.continuation(br_eq, h_idx, (@optic _.β0), opts_dir;
                    detect_codim2_bifurcation = 2, start_with_eigen = true, verbosity = 0,
                    bdlinsolver = BK.MatrixBLS())
                push!(hopf_branches, br)
            catch e
                @warn "  $label branch tracing failed" exception=e
            end
        end
    end

    # initialize the plot
    plt = plot(
        xlabel = "ε̃",
        ylabel = "Contact rate β₀",
        title = "SIRCmw 2D Hopf bifurcation and stability regions",
        xlims = (tilde_eps_MIN, tilde_eps_MAX),
        ylims = (β0_MIN, β0_MAX),
        legend = :topleft,
        size = (800, 600)
    )

    # plot stability regions and legend
    plot_stability_regions!(plt, tilde_eps_grid, beta0_grid, stability_matrix)
    add_stability_legend!(plt)

    # overlay the Hopf bifurcation curves
    hopf_plotted = false
    for (i, br) in enumerate(hopf_branches)
        x_pts = br.branch.tilde_eps[1:5:end]
        y_pts = br.branch.β0[1:5:end]
        if last(br.branch.tilde_eps) != last(x_pts)
            push!(x_pts, last(br.branch.tilde_eps))
            push!(y_pts, last(br.branch.β0))
        end
        
        plot!(plt, x_pts, y_pts;
            label = hopf_plotted ? false : "Hopf bifurcation curve (BK)",
            lc = :black,
            lw = 2.5,
            ls = :dash
        )
        hopf_plotted = true
    end

    # find and plot Bautin (Generalized Hopf) points
    gh_plotted = false
    for (i, br) in enumerate(hopf_branches)
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

    # save the plot
    output_path = joinpath(SCRIPT_DIR, "sircmw_2d_hopf_bifurcations.png")
    savefig(plt, output_path)
    println("Saved: $output_path")
end

# Sweep the (tilde_eps, sigma) plane and plot stability regions with the Hopf bifurcation curves
function run_eps_sigma_analysis()
    println("Starting (ε̃, σ) grid sweep at fixed β0 = $(PAR_BASE.β0)...")
    Nx, Ny = 300, 300
    tilde_eps_grid = range(tilde_eps_MIN, tilde_eps_MAX, length=Nx)
    sigma_grid     = range(σ_MIN, σ_MAX, length=Ny)

    # evaluate local stability at each grid point
    stability_matrix = zeros(Ny, Nx)
    @time for (j, sg) in enumerate(sigma_grid)
        for (i, te) in enumerate(tilde_eps_grid)
            stability_matrix[j, i] = check_stability_eps_sigma(te, sg)
        end
        if j % 50 == 0
            println("  row $j/$Ny done (σ = $(round(sg, digits=4)))")
        end
    end

    # run 2-parameter continuation in (tilde_eps, sigma) using BifurcationKit
    println("\nStarting 2-parameter continuation in (ε̃, σ) at β0 = $(PAR_BASE.β0)...")
    br_eq, hopf_idx = continue_in_tilde_eps(PAR_BASE)
    println("Found $(length(hopf_idx)) Hopf point(s) along the ε̃-sweep at σ = $(PAR_BASE.σ)")

    # we trace the hopf curve in 2 parameters from every hopf point found above and not just the first 
    # because (unlike the beta0 case) the two crossings here turned out to lie on two separate curves
    hopf_branches = []
    if !isempty(hopf_idx)
        # solver settings for the 2-parameter curve continuation
        opts_hopf2p = BK.ContinuationPar(
            p_min = σ_MIN, p_max = σ_MAX,
            ds = 0.01, dsmin = 1e-6, dsmax = 0.02, max_steps = 1200,
            newton_options = BK.NewtonPar(tol = 1e-8, max_iterations = 25, linesearch = true),
            detect_bifurcation = 1, nev = 6)

        for (hi, h_idx) in enumerate(hopf_idx)
            println("  Seeding from Hopf point $hi/$(length(hopf_idx))...")
            # trace the Hopf curve in both directions from the seed
            for (label, ds_sign) in (("Forward", 1.0), ("Backward", -1.0))
                try
                    println("    Tracing $label (ds = $(ds_sign*0.01)) from seed $hi...")
                    opts_dir = @set opts_hopf2p.ds = ds_sign * 0.01
                    br = BK.continuation(br_eq, h_idx, (@optic _.σ), opts_dir;
                        detect_codim2_bifurcation = 2, start_with_eigen = true, verbosity = 0,
                        bdlinsolver = BK.MatrixBLS())
                    println("    $label branch from seed $hi traced! ($(length(br.branch)) points)")
                    push!(hopf_branches, br)
                catch e
                    @warn "    $label branch tracing failed for seed $hi" exception=e
                end
            end
        end
    end

    # initialize the plot
    plt = plot(xlabel = "ε̃", ylabel = "σ",
        title = "SIRCmw (ε̃, σ) Hopf bifurcation and stability regions at β₀ = $(PAR_BASE.β0)",
        xlims = (tilde_eps_MIN, tilde_eps_MAX), ylims = (σ_MIN, σ_MAX),
        legend = :topleft, size = (800, 600))

    # plot stability regions and legend
    plot_stability_regions!(plt, tilde_eps_grid, sigma_grid, stability_matrix)
    add_stability_legend!(plt)

    # overlay the Hopf bifurcation curves
    hopf_plotted = false
    for br in hopf_branches
        x_pts = br.branch.tilde_eps[1:5:end]
        y_pts = br.branch.σ[1:5:end]
        plot!(plt, x_pts, y_pts; label = hopf_plotted ? false : "Hopf bifurcation curve (BK)",
              lc = :black, lw = 2.5, ls = :dash)
        hopf_plotted = true
    end

    # save the plot
    output_path = joinpath(SCRIPT_DIR, "sircmw_2d_eps_sigma_hopf.png")
    savefig(plt, output_path)
    println("Saved: $output_path")
end

# Traces the Hopf bifurcation curve in the (eps_tilde, beta0) plane for a single, fixed sigma value
function trace_hopf_curve_at_sigma(sigma_val)
    # update the parameters with the target sigma value
    p_sigma = merge(PAR_BASE, (σ = sigma_val,))  

    # find a Hopf point along a tilde_eps sweep to use as a continuation seed
    br_eq, hopf_idx = continue_in_tilde_eps(p_sigma)
    isempty(hopf_idx) && return nothing
    h_idx = hopf_idx[1]

    # solver settings: small ds/dsmax values prevent branch jumping at low beta0
    opts_hopf2p = BK.ContinuationPar(
        p_min = β0_MIN, p_max = β0_MAX,
        ds = 0.02, dsmin = 1e-6, dsmax = 0.5, max_steps = 8000,
        newton_options = BK.NewtonPar(tol = 1e-8, max_iterations = 25, linesearch = true),
        detect_bifurcation = 1, nev = 6)

    points = Tuple{Float64,Float64,Float64,String}[]

    # trace the curve in both directions (forward/upward and backward/downward)
    for (label, ds_sign) in (("forward", 1.0), ("backward", -1.0))
        try
            opts_dir = @set opts_hopf2p.ds = ds_sign * 0.02
            br = BK.continuation(br_eq, h_idx, (@optic _.β0), opts_dir;
                detect_codim2_bifurcation = 2, start_with_eigen = true, verbosity = 0,
                bdlinsolver = BK.MatrixBLS())
            
            # extract and keep points that fall within our physical plot boundaries
            n_kept = 0
            for (te, b0) in zip(br.branch.tilde_eps, br.branch.β0)
                if tilde_eps_MIN <= te <= tilde_eps_MAX && β0_MIN <= b0 <= β0_MAX
                    push!(points, (te, b0, sigma_val, label))
                    n_kept += 1
                end
            end
            println("    $label: $(length(br.branch)) points traced, $n_kept kept within (ε̃,β0) bounds")
        catch e
            @warn "    $label branch failed at σ=$sigma_val" exception=e
        end
    end
    return points
end

# Loop of slices across sigma that repeats the 2-parameter trace and stacks the results
function export_hopf_surface_slices()
    all_points = Tuple{Float64,Float64,Float64,String}[]

    println("Tracing Hopf curves across $(length(σ_SLICES)) σ slices...")
    for (k, sv) in enumerate(σ_SLICES)
        println("[$k/$(length(σ_SLICES))] σ = $(round(sv, digits=4))")
        pts = trace_hopf_curve_at_sigma(sv)
        if pts === nothing || isempty(pts)
            println("no Hopf curve found at this σ — skipping")
            continue
        end
        append!(all_points, pts)
    end

    println("\nTotal points collected: $(length(all_points))")

    output_path = joinpath(SCRIPT_DIR, "hopf_surface_eps_beta_sigma.csv")
    open(output_path, "w") do io
        println(io, "tilde_eps,beta0,sigma,branch")
        for (te, b0, sv, br) in all_points
            println(io, "$te,$b0,$sv,$br")
        end
    end
    println("Exported to $output_path")
end

# MAIN EXECUTION =======================================================

# Run the 2D parameters sweeps and overlay the bifurcation curves:
run_eps_beta_analysis()
run_eps_sigma_analysis()

# Export the stacked Hopf curves across many slices of sigma:
# (Uncomment this function to regenerate the CSV for the 3D surface plot)
# export_hopf_surface_slices()