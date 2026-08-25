# 2D diagram identifying stable and unstable regions + Hopf bifurcation lines 
# across values of tilde epsilon and beta for SIRCmw

using Logging
using Serialization
using Plots
using LinearAlgebra
using DelimitedFiles
import BifurcationKit as BK
import BifurcationKit: @optic, @set

ENV["GKSwstype"] = "100"
mkpath("SIRCmw")   # single guard for all savefig/CSV outputs below

include("../sircmw_utils.jl")
const σ_MIN, σ_MAX = 0.0, 0.3   # broadened around Casagrandi's 0.05-0.2


# checks stability for a (eps,beta) pair, we use this to color the plot 
# any two of (tilde_eps, beta0, sigma) can be swept
function check_stability_full(tilde_eps, beta0, sigma)
    p = PAR_BASE
    if beta0 < (p.μ + p.α)       
        return 0.0
    end

    eps = tilde_eps / SI_0
    coeffs = poly_coeffs(beta0, p.μ, p.α, p.γ, p.δ, eps, sigma)
    roots = poly_roots(coeffs)

    endemic_I = [real(r) for r in roots if abs(imag(r)) < 1e-7 && 1e-9 <= real(r) <= 1.0]
    if isempty(endemic_I)
        return NaN   # no valid endemic root returns nan
    end

    eq = recover_equilibrium(endemic_I[1], beta0, p.μ, p.α, p.γ, p.δ, eps, sigma)
    eq === nothing && return NaN

    p_point = (μ = p.μ, α = p.α, δ = p.δ, γ = p.γ, σ = sigma, β0 = beta0, tilde_eps = tilde_eps)
    J = jacobian_sircmw(eq, p_point)
    max_real = maximum(real.(eigvals(J)))
    return max_real < 0.0 ? 1.0 : 2.0
end

check_stability(tilde_eps, beta0) = check_stability_full(tilde_eps, beta0, PAR_BASE.σ)
check_stability_eps_sigma(tilde_eps, sigma) = check_stability_full(tilde_eps, PAR_BASE.β0, sigma)

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

function run_eps_sigma_analysis()
    println("Starting (ε̃, σ) grid sweep at fixed β0 = $(PAR_BASE.β0)...")
    Nx, Ny = 300, 300
    tilde_eps_grid = range(tilde_eps_MIN, tilde_eps_MAX, length=Nx)
    sigma_grid     = range(σ_MIN, σ_MAX, length=Ny)

    stability_matrix = zeros(Ny, Nx)
    @time for (j, sg) in enumerate(sigma_grid)
        for (i, te) in enumerate(tilde_eps_grid)
            stability_matrix[j, i] = check_stability_eps_sigma(te, sg)
        end
        if j % 50 == 0
            println("  row $j/$Ny done (σ = $(round(sg, digits=4)))")
        end
    end

    println("\nStarting 2-parameter continuation in (ε̃, σ) at β0 = $(PAR_BASE.β0)...")

    u0_eq = let
        p = PAR_BASE
        eps = p.tilde_eps / SI_0
        coeffs = poly_coeffs(p.β0, p.μ, p.α, p.γ, p.δ, eps, p.σ)
        roots = poly_roots(coeffs)
        endemic_I = [real(r) for r in roots if abs(imag(r)) < 1e-7 && 0.0 <= real(r) <= 1.0]
        eq = recover_equilibrium(endemic_I[1], p.β0, p.μ, p.α, p.γ, p.δ, eps, p.σ)
        [eq[1], eq[2], eq[3], eq[4]]
    end

    prob_eq = BK.ODEBifProblem(sircmw!, u0_eq, PAR_BASE, (@optic _.tilde_eps);
        record_from_solution = (x, p; k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))

    opts_eq = BK.ContinuationPar(
        p_min = tilde_eps_MIN, p_max = tilde_eps_MAX,
        ds = 0.005, dsmin = 1e-6, dsmax = 0.05, max_steps = 500,
        newton_options = BK.NewtonPar(tol = 1e-9, max_iterations = 25, linesearch = true),
        detect_bifurcation = 3, n_inversion = 6, nev = 6)

    br_eq = BK.continuation(prob_eq, BK.PALC(), opts_eq; verbosity = 0)
    hopf_idx = findall(sp -> sp.type == :hopf, br_eq.specialpoint)
    println("Found $(length(hopf_idx)) Hopf point(s) along the ε̃-sweep at σ = $(PAR_BASE.σ)")

    hopf_branches = []
    if !isempty(hopf_idx)
        # NOTE: ds scaled to σ's range (~0.3 wide), unlike beta0's ds=2.0 over a ~2000-wide range
        opts_hopf2p = BK.ContinuationPar(
            p_min = σ_MIN, p_max = σ_MAX,
            ds = 0.01, dsmin = 1e-6, dsmax = 0.02, max_steps = 1200,
            newton_options = BK.NewtonPar(tol = 1e-8, max_iterations = 25, linesearch = true),
            detect_bifurcation = 1, nev = 6)

        h_idx = hopf_idx[1]

        for (label, ds_sign) in (("Forward", 1.0), ("Backward", -1.0))
            try
                println("  Tracing $label (ds = $(ds_sign*0.01))...")
                opts_dir = @set opts_hopf2p.ds = ds_sign * 0.01
                br = BK.continuation(br_eq, h_idx, (@optic _.σ), opts_dir;
                    detect_codim2_bifurcation = 2, start_with_eigen = true, verbosity = 0,
                    bdlinsolver = BK.MatrixBLS())
                println("  $label branch traced! ($(length(br.branch)) points)")
                push!(hopf_branches, br)
            catch e
                @warn "  $label branch tracing failed" exception=e
            end
        end
    end

    custom_cmap = cgrad([RGB(0.9, 0.9, 0.9), RGB(0.7, 0.8, 0.95), RGB(0.95, 0.7, 0.75)])
    plt = plot(xlabel = "ε̃", ylabel = "σ",
        title = "SIRCmw (ε̃, σ) Hopf bifurcation and stability regions at β₀ = $(PAR_BASE.β0)",
        xlims = (tilde_eps_MIN, tilde_eps_MAX), ylims = (σ_MIN, σ_MAX),
        legend = :topleft, size = (800, 600))

    heatmap!(plt, tilde_eps_grid, sigma_grid, stability_matrix; c = custom_cmap, colorbar = :none, opacity = 0.85)

    plot!(plt, [NaN], [NaN], seriestype=:shape, fillcolor=RGB(0.9,0.9,0.9), label="DFE stable (R₀ < 1)", linecolor=:transparent)
    plot!(plt, [NaN], [NaN], seriestype=:shape, fillcolor=RGB(0.7,0.8,0.95), label="Stable endemic equilibrium", linecolor=:transparent)
    plot!(plt, [NaN], [NaN], seriestype=:shape, fillcolor=RGB(0.95,0.7,0.75), label="Unstable endemic equilibrium", linecolor=:transparent)

    hopf_plotted = false
    for br in hopf_branches
        x_pts = br.branch.tilde_eps[1:5:end]
        y_pts = br.branch.σ[1:5:end]
        plot!(plt, x_pts, y_pts; label = hopf_plotted ? false : "Hopf bifurcation curve (BifurcationKit)",
              lc = :black, lw = 2.5, ls = :dash)
        hopf_plotted = true
    end

    output_path = "SIRCmw/sircmw_2d_eps_sigma_hopf.png"
    savefig(plt, output_path)
    println("Saved: $output_path")
end

const σ_SLICES = range(0.0, 0.3, length=8)  # exploratory: coarse and few, refine later

function trace_hopf_curve_at_sigma(sigma_val)
    p_sigma = merge(PAR_BASE, (σ = sigma_val,))   # NamedTuple with sigma overridden, everything else fixed

    u0_eq = let
        eps = p_sigma.tilde_eps / SI_0
        coeffs = poly_coeffs(p_sigma.β0, p_sigma.μ, p_sigma.α, p_sigma.γ, p_sigma.δ, eps, p_sigma.σ)
        roots = poly_roots(coeffs)
        endemic_I = [real(r) for r in roots if abs(imag(r)) < 1e-7 && 0.0 <= real(r) <= 1.0]
        isempty(endemic_I) && return nothing
        eq = recover_equilibrium(endemic_I[1], p_sigma.β0, p_sigma.μ, p_sigma.α, p_sigma.γ, p_sigma.δ, eps, p_sigma.σ)
        eq === nothing ? nothing : [eq[1], eq[2], eq[3], eq[4]]
    end
    u0_eq === nothing && return nothing

    prob_eq = BK.ODEBifProblem(sircmw!, u0_eq, p_sigma, (@optic _.tilde_eps);
        record_from_solution = (x, p; k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))

    opts_eq = BK.ContinuationPar(
        p_min = tilde_eps_MIN, p_max = tilde_eps_MAX,
        ds = 0.005, dsmin = 1e-6, dsmax = 0.05, max_steps = 500,
        newton_options = BK.NewtonPar(tol = 1e-9, max_iterations = 25, linesearch = true),
        detect_bifurcation = 3, n_inversion = 6, nev = 6)

    br_eq = BK.continuation(prob_eq, BK.PALC(), opts_eq; verbosity = 0)
    hopf_idx = findall(sp -> sp.type == :hopf, br_eq.specialpoint)
    isempty(hopf_idx) && return nothing
    h_idx = hopf_idx[1]

    opts_hopf2p = BK.ContinuationPar(
        p_min = β0_MIN, p_max = β0_MAX,
        ds = 2.0, dsmin = 1e-5, dsmax = 8.0, max_steps = 1200,
        newton_options = BK.NewtonPar(tol = 1e-8, max_iterations = 25, linesearch = true),
        detect_bifurcation = 1, nev = 6)

    points = Tuple{Float64,Float64,Float64}[]   # (tilde_eps, beta0, sigma)

    for (label, ds_sign) in (("forward", 1.0), ("backward", -1.0))
        try
            opts_dir = @set opts_hopf2p.ds = ds_sign * 2.0
            br = BK.continuation(br_eq, h_idx, (@optic _.β0), opts_dir;
                detect_codim2_bifurcation = 2, start_with_eigen = true, verbosity = 0,
                bdlinsolver = BK.MatrixBLS())
            for (te, b0) in zip(br.branch.tilde_eps, br.branch.β0)
                push!(points, (te, b0, sigma_val))
            end
            println("    $label: $(length(br.branch)) points")
        catch e
            @warn "    $label branch failed at σ=$sigma_val" exception=e
        end
    end
    return points
end

function export_hopf_surface_slices()
    all_points = Tuple{Float64,Float64,Float64}[]

    println("Tracing Hopf curves across $(length(σ_SLICES)) σ slices...")
    for (k, sv) in enumerate(σ_SLICES)
        println("[$k/$(length(σ_SLICES))] σ = $(round(sv, digits=4))")
        pts = trace_hopf_curve_at_sigma(sv)
        if pts === nothing || isempty(pts)
            println("    no Hopf curve found at this σ — skipping")
            continue
        end
        append!(all_points, pts)
    end

    println("\nTotal points collected: $(length(all_points))")

    output_path = "SIRCmw/hopf_surface_eps_beta_sigma.csv"
    open(output_path, "w") do io
        println(io, "tilde_eps,beta0,sigma")
        writedlm(io, hcat(getindex.(all_points,1), getindex.(all_points,2), getindex.(all_points,3)), ',')
    end
    println("Exported to $output_path")
end

# RUN EVERYTHING

run_analysis()
run_eps_sigma_analysis()
export_hopf_surface_slices()