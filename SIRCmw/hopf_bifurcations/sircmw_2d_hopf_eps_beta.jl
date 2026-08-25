"""
This script extracts the 2D Hopf bifurcation and stability analysis in the (tilde epsilon, beta0) plane
from sircmw_3d_hopf_surface_data.jl, extending the tilde epsilon range up to 4.0.
"""

using Plots
using LinearAlgebra
using DelimitedFiles
import BifurcationKit as BK
import BifurcationKit: @optic, @set

ENV["GKSwstype"] = "100"
const SCRIPT_DIR = dirname(@__FILE__)
mkpath(SCRIPT_DIR)  

include("../sircmw_utils.jl")

# Range extension to 10.0 and 10000.0 as requested
const tilde_eps_MAX_10 = 10.0
const β0_MAX_10000 = 10000.0

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

check_stability(tilde_eps, beta0) = check_stability_full(tilde_eps, beta0, PAR_BASE.σ)

# Color palette for stability regions
const COLOR_DFE = RGB(0.9, 0.9, 0.9) # light gray
const COLOR_STABLE = RGB(0.7, 0.8, 0.95) # baby blue
const COLOR_UNSTABLE = RGB(0.95, 0.7, 0.75) # pink

# heatmap plot
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
        p_min = tilde_eps_MIN, p_max = tilde_eps_MAX_10,
        ds = 0.005, dsmin = 1e-6, dsmax = 0.05, max_steps = 2000,
        newton_options = BK.NewtonPar(tol = 1e-9, max_iterations = 25, linesearch = true),
        detect_bifurcation = 3, n_inversion = 6, nev = 6)

    br_eq = BK.continuation(prob_eq, BK.PALC(), opts_eq; verbosity = 0)
    hopf_idx = findall(sp -> sp.type == :hopf, br_eq.specialpoint)
    return br_eq, hopf_idx
end

# sweep the (tilde_eps, beta0) plane and plot stability regions with the Hopf bifurcation curve
function run_eps_beta_analysis()
    println("Starting 2D Grid Sweep (0.0 to 10.0)...")
    Nx, Ny = 300, 300 # grid resolution
    tilde_eps_grid = range(tilde_eps_MIN, tilde_eps_MAX_10, length=Nx)
    beta0_grid = range(β0_MIN, β0_MAX_10000, length=Ny)

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
            p_min = β0_MIN, p_max = β0_MAX_10000,
            ds = 2.0, dsmin = 1e-5, dsmax = 20.0, max_steps = 8000,
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
        title = "SIRCmw 2D Hopf bifurcation and stability regions (ε̃ ∈ [0, 10], β₀ ∈ [0, 10000])",
        xlims = (tilde_eps_MIN, tilde_eps_MAX_10),
        ylims = (β0_MIN, β0_MAX_10000),
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
    output_path = joinpath(SCRIPT_DIR, "sircmw_2d_hopf_bifurcations_eps_10_beta_10000.png")
    savefig(plt, output_path)
    println("Saved: $output_path")
end

# Run the epsilon-beta analysis
run_eps_beta_analysis()
