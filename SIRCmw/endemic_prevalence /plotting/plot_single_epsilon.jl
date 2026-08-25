# SIRCmw single epsilon endemic prevalence plotting

using Logging
using Serialization
using Plots
using Plots.PlotMeasures
using LinearAlgebra
import BifurcationKit as BK
import BifurcationKit: @optic, @set

ENV["GKSwstype"] = "100"

# --- EXPERIMENTAL PARAMETER ---
# Change this value to experiment with different epsilon tilde regimes (e.g. 5.0, 10.0, 15.0)
const SELECTED_TILDE_EPS = 100.0
# Maximum beta0 to sweep (since epsilon_tilde is high, we zoom out to a wider range)
const MAX_BETA0 = 15000.0
# ------------------------------

# Include the shared utilities from the grandparent directory
const PLOTTING_DIR = @__DIR__
include(joinpath(PLOTTING_DIR, "..", "..", "sircmw_utils.jl"))

# Continuation parameters
const OPTS_SCAN = BK.ContinuationPar(
    p_min = β0_MIN, p_max = MAX_BETA0,
    dsmin = 1e-5, dsmax = 10.0,
    max_steps = 1500,
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

function plot_single_eps()
    println("Running continuation for single ε̃ = $SELECTED_TILDE_EPS up to β₀ = $MAX_BETA0...")
    p_temp = @set PAR_BASE.tilde_eps = SELECTED_TILDE_EPS
    u0_eq = get_endemic_equilibrium(p_temp)
    
    prob_beta = BK.ODEBifProblem(sircmw!, u0_eq, p_temp, (@optic _.β0);
        record_from_solution = (x, p; k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))
        
    br_up = BK.continuation(prob_beta, BK.PALC(), (@set OPTS_SCAN.ds = 4.0); verbosity = 0)
    br_down = BK.continuation(prob_beta, BK.PALC(), (@set OPTS_SCAN.ds = -4.0); verbosity = 0)
    
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
    
    # Plotting configuration
    plt = plot(
        xlabel = "Contact rate β₀",
        ylabel = "Infected fraction I*",
        title = "SIRCmw Endemic Prevalence for ε̃ = $SELECTED_TILDE_EPS",
        grid = true,
        xlims = (0.0, MAX_BETA0),
        titlefontsize = 14,
        guidefontsize = 12,
        tickfontsize = 10,
        legendfontsize = 10,
        linewidth = 2.5,
        legend = :topright,
        size = (800, 500),
        margin = 6mm
    )
    
    # Reference indicators
    vline!(plt, [transcritical_β0]; lc = :gray, ls = :dash, lw = 1.2, label = false)
    
    # Plot stable and unstable curves
    plot!(plt, sx, sy; lc = :purple, label = "Stable state")
    plot!(plt, ux, uy; lc = :purple, ls = :dash, label = "Unstable state")
    
    # Overlay Hopf bifurcation points
    hopf_plotted = false
    for (hx, hy) in hopf_pts
        println("Detected Hopf Bifurcation Point at β₀ ≈ $(round(hx; digits=2))")
        scatter!(plt, [hx], [hy]; mc = :green, ms = 8, marker = :diamond,
                 label = hopf_plotted ? false : "Hopf point")
        hopf_plotted = true
    end
    
    # Save the output plot
    output_png = joinpath(PLOTTING_DIR, "prevalence_single_epsilon.png")
    savefig(plt, output_png)
    println("Saved plot to: $output_png")
end

plot_single_eps()
