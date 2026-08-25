#  SIRC vs SIRCmw endemic prevalence comparison 

using Logging
using Serialization
using Plots
using LinearAlgebra
import BifurcationKit as BK
import BifurcationKit: @optic, @set
import OrdinaryDiffEq as DE

ENV["GKSwstype"] = "100"

include("sircmw_utils.jl")

# MODELS

function sirc_unforced!(du, u, p, t = 0)
    S, I, R, C = u
    β = p.β0
    du[1] = p.μ*(1.0 - S) - β*S*I + p.γ*C                     
    du[2] = β*S*I + p.σ*β*C*I - (p.μ + p.α)*I              
    du[3] = (1.0 - p.σ)*β*C*I + p.α*I - (p.μ + p.δ)*R     
    du[4] = p.δ*R - β*C*I - (p.μ + p.γ)*C   
    du
end

const U0_DEFAULT = [0.3, 1e-3, 0.4, 0.299]

# continuation parameters
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

# helper to split a branch into stable and unstable segments
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

# PLOT: ENDEMIC EQ OF UNFORCED SIRC

# find endemic equilibrium of unforced SIRC at β0 = 600.0
sol_burnin = DE.solve(
    DE.ODEProblem(sirc_unforced!, U0_DEFAULT, (0.0, 1000.0), PAR_BASE),
    DE.AutoTsit5(DE.Rosenbrock23());
    abstol = 1e-12, reltol = 1e-12, maxiters = 10^7
)
u0_sirc = sol_burnin.u[end]

prob_sirc = BK.ODEBifProblem(sirc_unforced!, u0_sirc, PAR_BASE, (@optic _.β0);
    record_from_solution = (x, p; k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))

# tracing the endemic branch
br_sirc_up = BK.continuation(prob_sirc, BK.PALC(), (@set OPTS_SCAN.ds = 2.0); verbosity = 0)
br_sirc_down = BK.continuation(prob_sirc, BK.PALC(), (@set OPTS_SCAN.ds = -2.0); verbosity = 0)

param_sirc = vcat(reverse(br_sirc_down.branch.param), br_sirc_up.branch.param[2:end])
I_sirc     = vcat(reverse(br_sirc_down.branch.I), br_sirc_up.branch.I[2:end])
stable_sirc = vcat(reverse(br_sirc_down.branch.stable), br_sirc_up.branch.stable[2:end])

# the transcritical bifurcation is a branch point (bp)
transcritical_β0 = PAR_BASE.μ + PAR_BASE.α
β0_grid = collect(range(β0_MIN, β0_MAX, length=1000))
stable_I = [β0 < transcritical_β0 ? 0.0 : I_sirc[argmin(abs.(param_sirc .- β0))] for β0 in β0_grid]
unstable_I = [β0 >= transcritical_β0 ? 0.0 : NaN for β0 in β0_grid]

plt_sirc = plot(xlabel = "β₀ (contact rate)",
                ylabel = "Infected fraction I*",
                title  = "Unforced SIRC endemic prevalence",
                xlims  = (β0_MIN, β0_MAX),
                ylims  = (-0.0005, 0.0035),
                size   = (800, 500),
                legend = :topleft)

# Plot physical stable branch in solid blue
plot!(plt_sirc, β0_grid, stable_I; lc = :blue, lw = 2.5, label = "Stable state")
# Plot unstable DFE as dashed red
plot!(plt_sirc, β0_grid, unstable_I; lc = :red, lw = 2.0, ls = :dash, label = "Unstable state")
# Mark transcritical bifurcation
scatter!(plt_sirc, [transcritical_β0], [0.0]; mc = :green, ms = 7, marker = :diamond, label = "Transcritical BP (β0 ≈ $(round(transcritical_β0; digits=1)))")

savefig(plt_sirc, "SIRC/unforced_sirc_prevalence.png")
println("Saved: SIRC/unforced_sirc_prevalence.png")


# 4. PLOT 2: SIRCmw COMPARISON PLOT

tilde_eps_vals = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
colors = [:blue, :red, :green, :orange, :purple, :magenta, :cyan, :brown, :olive]

plt_sircmw = plot(xlabel = "β₀ (contact rate)",
                  ylabel = "Infected fraction I*",
                  title  = "SIRCmw Prevalence Comparison",
                  xlims  = (β0_MIN, β0_MAX),
                  size   = (1000, 700),
                  legend = :outerright)

for (i, te) in enumerate(tilde_eps_vals)
    println("  · Running for ε̃ = $te")
    p_temp = @set PAR_BASE.tilde_eps = te
    
    # get endemic equilibrium
    u0_eq = get_endemic_equilibrium(p_temp)
    
    prob_beta = BK.ODEBifProblem(sircmw!, u0_eq, p_temp, (@optic _.β0);
        record_from_solution = (x, p; k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))
        
    br_up = BK.continuation(prob_beta, BK.PALC(), (@set OPTS_SCAN.ds = 4.0); verbosity = 0)
    br_down = BK.continuation(prob_beta, BK.PALC(), (@set OPTS_SCAN.ds = -4.0); verbosity = 0)
    
    # merge
    param_vals = vcat(reverse(br_down.branch.param), br_up.branch.param[2:end])
    I_vals     = vcat(reverse(br_down.branch.I), br_up.branch.I[2:end])
    stable_vals = vcat(reverse(br_down.branch.stable), br_up.branch.stable[2:end])
    
    # physical clipping
    for j in 1:length(param_vals)
        if param_vals[j] < transcritical_β0
            I_vals[j] = 0.0
            stable_vals[j] = true
        end
    end
    I_vals = max.(0.0, I_vals)
    
    # split by stability
    sx, sy, ux, uy = split_by_stability(param_vals, I_vals, stable_vals)
    
    col = colors[mod1(i, length(colors))]
    
    # plot stable branch in solid, unstable in dashed
    plot!(plt_sircmw, sx, sy; lc = col, lw = 2.0, label = "ε̃ = $te")
    plot!(plt_sircmw, ux, uy; lc = col, lw = 2.0, ls = :dash, label = false)
    
    # plot hopf bifurcation points
    sps = vcat(br_down.specialpoint, br_up.specialpoint)
    hopf_plotted = false
    for sp in sps
        sp.type == :hopf || continue
        sp.step > 2 || continue
        idx = argmin(abs.(param_vals .- sp.param))
        scatter!(plt_sircmw, [sp.param], [I_vals[idx]];
                 mc = :green, ms = 6, marker = :diamond, label = hopf_plotted ? false : "Hopf (ε̃ = $te)")
        hopf_plotted = true
    end
end

savefig(plt_sircmw, "SIRCmw/sircmw_prevalence_comparison.png")
println("Saved: SIRCmw/sircmw_prevalence_comparison.png")

# PLOT: SINGLE EPSILON PLOT
selected_te = 1

p_selected = @set PAR_BASE.tilde_eps = selected_te

u0_sel = get_endemic_equilibrium(p_selected)

prob_sel = BK.ODEBifProblem(sircmw!, u0_sel, p_selected, (@optic _.β0);
    record_from_solution = (x, p; k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))

br_sel_up = BK.continuation(prob_sel, BK.PALC(), (@set OPTS_SCAN.ds = 4.0); verbosity = 0)
br_sel_down = BK.continuation(prob_sel, BK.PALC(), (@set OPTS_SCAN.ds = -4.0); verbosity = 0)

param_sel = vcat(reverse(br_sel_down.branch.param), br_sel_up.branch.param[2:end])
I_sel     = vcat(reverse(br_sel_down.branch.I), br_sel_up.branch.I[2:end])
stable_sel = vcat(reverse(br_sel_down.branch.stable), br_sel_up.branch.stable[2:end])

for j in 1:length(param_sel)
    if param_sel[j] < transcritical_β0
        I_sel[j] = 0.0
        stable_sel[j] = true
    end
end
I_sel = max.(0.0, I_sel)

sx, sy, ux, uy = split_by_stability(param_sel, I_sel, stable_sel)

plt_sel = plot(xlabel = "β₀ (contact rate)",
               ylabel = "Infected fraction I*",
               title  = "SIRCmw Prevalence for ε̃ = $selected_te",
               xlims  = (β0_MIN, β0_MAX),
               size   = (800, 500),
               legend = :topright)
               
plot!(plt_sel, sx, sy; lc = :purple, lw = 2.5, label = "Stable equilibrium")
plot!(plt_sel, ux, uy; lc = :purple, lw = 2.0, ls = :dash, label = "Unstable equilibrium")

# plot hopf bifurcation points
sps = vcat(br_sel_down.specialpoint, br_sel_up.specialpoint)
for sp in sps
    sp.type == :hopf || continue
    sp.step > 2 || continue
    idx = argmin(abs.(param_sel .- sp.param))
    scatter!(plt_sel, [sp.param], [I_sel[idx]];
             mc = :green, ms = 7, marker = :diamond, label = "Hopf point (β0 ≈ $(round(sp.param; digits=1)))")
end

# R0 = 1 line reference
vline!(plt_sel, [PAR_BASE.μ + PAR_BASE.α]; ls = :dot, lc = :gray, label = "R₀ = 1 floor")

savefig(plt_sel, "SIRCmw/sircmw_prevalence_selected.png")
println("Saved selected plot to: SIRCmw/sircmw_prevalence_selected.png")

# PLOT: EQUILIBRIUM BRANCH FOR SIRCmw vs tilde_eps
println("\n5. Tracing and Plotting SIRCmw Equilibrium Branch in tilde_eps...")

const OPTS_EQ = BK.ContinuationPar(
    p_min = tilde_eps_MIN, p_max = tilde_eps_MAX, # range
    ds = 0.005, dsmin = 1e-6, dsmax = 0.05, # step size
    max_steps = 500,
    newton_options = BK.NewtonPar(tol = 1e-9, max_iterations = 25, linesearch = true),
    detect_bifurcation = 3, # monitor eigenvalues at each step
    n_inversion = 6, # for bisection
    nev = 6)

# Trace equilibrium branch in tilde_eps starting from endemic equilibrium of PAR_BASE
u0_eq = get_endemic_equilibrium(PAR_BASE)
prob_eq = BK.ODEBifProblem(sircmw!, u0_eq, PAR_BASE, (@optic _.tilde_eps);
    record_from_solution = (x, p; k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))
    
br_eq = BK.continuation(prob_eq, BK.PALC(), OPTS_EQ; verbosity = 0)

plt_eq = plot(xlabel = "tilde_eps",
              ylabel = "Infected fraction I*",
              title  = "SIRCmw Equilibrium Branch",
              xlims  = (tilde_eps_MIN, tilde_eps_MAX),
              legend = :topleft)
              
param_vals = br_eq.branch.param
I_vals = br_eq.branch.I
stability = br_eq.branch.stable

stable_x, stable_y = Float64[], Float64[]
unstable_x, unstable_y = Float64[], Float64[]

for i in 1:length(param_vals)
    if stability[i]
        push!(stable_x, param_vals[i]);   push!(stable_y, I_vals[i])
        push!(unstable_x, NaN);           push!(unstable_y, NaN)
    else
        push!(unstable_x, param_vals[i]); push!(unstable_y, I_vals[i])
        push!(stable_x, NaN);             push!(stable_y, NaN)
    end
end

if !isempty(stable_x)
    plot!(plt_eq, stable_x, stable_y; label="Stable equilibrium", lc=:blue, lw=2.0)
end
if !isempty(unstable_x)
    plot!(plt_eq, unstable_x, unstable_y; label="Unstable equilibrium", lc=:red, lw=2.0)
end

# Add Hopf marker
for sp in br_eq.specialpoint
    if sp.type == :hopf
        scatter!(plt_eq, [sp.param], [sp.x[2]]; 
                 mc=:green, ms=6, marker=:diamond, label="Hopf (tilde_eps ≈ $(round(sp.param; digits=4)))")
    end
end
savefig(plt_eq, "SIRCmw/sircmw_equilibrium_branch2.png")
println("Saved: SIRCmw/sircmw_equilibrium_branch2.png")

