# SIRC vs SIRCmw endemic prevalence comparison (2-Panel Plot)

using Logging
using Serialization
using Plots
using Plots.PlotMeasures
using LinearAlgebra
import BifurcationKit as BK
import BifurcationKit: @optic, @set

ENV["GKSwstype"] = "100"

# --- EXPERIMENTAL ZOOM PARAMETER ---
# Maximum contact rate beta0 for the right panel (decrease to zoom in, increase to zoom out)
const RIGHT_PANEL_BETA0_MAX = 1500.0
# -----------------------------------

# Include the shared utilities from the grandparent directory
const PLOTTING_DIR = @__DIR__
include(joinpath(PLOTTING_DIR, "..", "..", "sircmw_utils.jl"))

# Continuation parameters
const OPTS_SCAN = BK.ContinuationPar(
    p_min = β0_MIN, p_max = β0_MAX,
    dsmin = 1e-5, dsmax = 10.0,
    max_steps = 1000,
    newton_options = BK.NewtonPar(tol = 1e-9, max_iterations = 25, linesearch = true),
    detect_bifurcation = 3,
    n_inversion = 8,
    tol_stability = 1e-3,
    nev = 6
)

# Helper function to get the endemic equilibrium
function get_endemic_equilibrium(p)
    SI_0 = 0.178 * 0.001
    eps = p.tilde_eps / SI_0
    coeffs = poly_coeffs(p.β0, p.μ, p.α, p.γ, p.δ, eps, p.σ)
    roots = poly_roots(coeffs)
    endemic_I = [real(r) for r in roots if abs(imag(r)) < 1e-8 && 0.0 <= real(r) <= 1.0]
    if isempty(endemic_I)
        error("No real endemic equilibrium root found in [0,1]")
    end
    eq = recover_equilibrium(endemic_I[1], p.β0, p.μ, p.α, p.γ, p.δ, eps, p.σ)
    if isnothing(eq)
        error("Could not reconstruct equilibrium from root I* = $(endemic_I[1])")
    end
    return [eq[1], eq[2], eq[3], eq[4]]
end

# Helper to split a branch into stable and unstable segments
function split_by_stability(x_vals, y_vals, stable_mask)
    sx, sy, ux, uy = Float64[], Float64[], Float64[], Float64[]
    for j in 1:length(x_vals)
        if stable_mask[j]
            push!(sx, x_vals[j]); push!(sy, y_vals[j])
            push!(ux, NaN);       push!(uy, NaN)
        else
            push!(ux, x_vals[j]); push!(uy, y_vals[j])
            push!(sx, NaN);       push!(sy, NaN)
        end
    end
    return (sx, sy, ux, uy)
end

# Helper to run continuation for a specific tilde_eps value with custom p_max
function run_continuation_for_eps(te, p_max = β0_MAX)
    p_temp = @set PAR_BASE.tilde_eps = te
    u0_eq = get_endemic_equilibrium(p_temp)
    
    # Configure continuation options with custom p_max
    opts = @set OPTS_SCAN.p_max = p_max
    
    prob_beta = BK.ODEBifProblem(sircmw!, u0_eq, p_temp, (@optic _.β0);
        record_from_solution = (x, p; k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))
        
    br_up = BK.continuation(prob_beta, BK.PALC(), (@set opts.ds = 4.0); verbosity = 0)
    br_down = BK.continuation(prob_beta, BK.PALC(), (@set opts.ds = -4.0); verbosity = 0)
    
    # Merge branches
    param_vals = vcat(reverse(br_down.branch.param), br_up.branch.param[2:end])
    I_vals     = vcat(reverse(br_down.branch.I), br_up.branch.I[2:end])
    stable_vals = vcat(reverse(br_down.branch.stable), br_up.branch.stable[2:end])
    
    # Physical clipping below transcritical bifurcation
    transcritical_β0 = PAR_BASE.μ + PAR_BASE.α
    for j in 1:length(param_vals)
        if param_vals[j] < transcritical_β0
            I_vals[j] = 0.0
            stable_vals[j] = true
        end
    end
    I_vals = max.(0.0, I_vals)
    
    # Split by stability
    sx, sy, ux, uy = split_by_stability(param_vals, I_vals, stable_vals)
    
    # Extract Hopf bifurcation points
    sps = vcat(br_down.specialpoint, br_up.specialpoint)
    hopf_pts = Tuple{Float64, Float64}[]
    for sp in sps
        sp.type == :hopf || continue
        sp.step > 2 || continue
        idx = argmin(abs.(param_vals .- sp.param))
        push!(hopf_pts, (sp.param, I_vals[idx]))
    end
    
    return sx, sy, ux, uy, hopf_pts
end

function generate_plots()
    # Define regimes (weak mutation feedback: 0.0 to 0.3)
    tilde_eps_left = [0.0, 0.1, 0.2, 0.3]
    tilde_eps_right = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    
    # Color palette
    colors_left = palette(:tab10)[1:length(tilde_eps_left)]
    colors_right = palette(:tab10)[1:length(tilde_eps_right)]
    
    # Common font sizes and styling options
    plot_style = (
        xlabel = "Contact rate β₀",
        ylabel = "Infected fraction I*",
        grid = true,
        titlefontsize = 13,
        guidefontsize = 13,
        tickfontsize = 11,
        legendfontsize = 10,
        linewidth = 2.5,
        margin = 8mm
    )
    
    # Transcritical bifurcation reference line
    transcritical_β0 = PAR_BASE.μ + PAR_BASE.α
 
    println("Generating Left Panel (Weak Mutation Feedback Regime)...")
    plt_left = plot(; xlims = (0.0, 1500.0), ylims = (0.0, 0.003), legend = :topright, plot_style...)
    
    # Reference indicators for the left panel
    vline!(plt_left, [transcritical_β0]; lc = :gray, ls = :dash, lw = 1.2, label = false)
    
    for (i, te) in enumerate(tilde_eps_left)
        println("  · Running for ε̃ = $te")
        sx, sy, ux, uy, hopf_pts = run_continuation_for_eps(te, 1500.0)
        col = colors_left[i]
        
        lbl = te == 0.0 ? "ε̃ = 0.0 (SIRC)" : "ε̃ = $te"
        plot!(plt_left, sx, sy; lc = col, label = lbl)
        plot!(plt_left, ux, uy; lc = col, ls = :dash, label = false)
        
        for (hx, hy) in hopf_pts
            scatter!(plt_left, [hx], [hy]; mc = :green, ms = 7, marker = :diamond, label = false)
        end
    end
    
    println("Generating Right Panel (Moderate & High Mutation Feedback Regime)...")
    plt_right = plot(; xlims = (0.0, RIGHT_PANEL_BETA0_MAX), ylims = (0.0, 0.6), legend = :topright, plot_style...)
    
    # Reference indicators for the right panel
    vline!(plt_right, [transcritical_β0]; lc = :gray, ls = :dash, lw = 1.2, label = false)
    
    for (i, te) in enumerate(tilde_eps_right)
        println("  · Running for ε̃ = $te")
        sx, sy, ux, uy, hopf_pts = run_continuation_for_eps(te, RIGHT_PANEL_BETA0_MAX)
        col = colors_right[i]
        
        plot!(plt_right, sx, sy; lc = col, label = "ε̃ = $te")
        plot!(plt_right, ux, uy; lc = col, ls = :dash, label = false)
        
        for (hx, hy) in hopf_pts
            scatter!(plt_right, [hx], [hy]; mc = :green, ms = 7, marker = :diamond, label = false)
        end
    end
    
    # Add dummy entries at the end of the right panel legend to keep them ordered last
    plot!(plt_right, [NaN], [NaN]; lc = :black, lw = 2.0, label = "Stable state")
    plot!(plt_right, [NaN], [NaN]; lc = :black, lw = 2.0, ls = :dash, label = "Unstable state")
    scatter!(plt_right, [NaN], [NaN]; mc = :green, ms = 7, marker = :diamond, label = "Hopf point")
    
    # Combine the plots into a 2-panel figure
    println("Saving final combined plot...")
    plt_combined = plot(plt_left, plt_right; layout = (1, 2), size = (1500, 650))
    
    output_png = joinpath(PLOTTING_DIR, "prevalence_comparison_two_panels.png")
    savefig(plt_combined, output_png)
    println("Saved PNG plot to: $output_png")
    
    output_pdf = joinpath(PLOTTING_DIR, "prevalence_comparison_two_panels.pdf")
    savefig(plt_combined, output_pdf)
    println("Saved PDF plot to: $output_pdf")
end

generate_plots()
