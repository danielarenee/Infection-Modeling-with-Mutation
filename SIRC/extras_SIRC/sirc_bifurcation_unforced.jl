# ═════════════════════════════════════════════════════════════════════════════
#  Autonomous SIRC Bifurcation Analysis (No Seasonal Forcing)
#  This script performs a bifurcation analysis of the autonomous SIRC model
#  by tracing the equilibrium states S, I, R, C as a function of β₀.
# ═════════════════════════════════════════════════════════════════════════════

using Logging
using Serialization
using Plots
using LinearAlgebra
import BifurcationKit as BK
import BifurcationKit: @optic, @set
import OrdinaryDiffEq as DE

# Headless plotting to prevent opening windows on execution
ENV["GKSwstype"] = "100"

# ─── 1. THE MODEL ───────────────────────────────────────────────────────────
# Autonomous 4D SIRC system (without seasonal forcing, i.e., ε = 0)
function sirc_unforced!(du, u, p, t = 0)
    S, I, R, C = u
    β = p.β0
    
    du[1] = p.μ*(1.0 - S) - β*S*I + p.γ*C                     
    du[2] = β*S*I + p.σ*β*C*I - (p.μ + p.α)*I              
    du[3] = (1.0 - p.σ)*β*C*I + p.α*I - (p.μ + p.δ)*R     
    du[4] = p.δ*R - β*C*I - (p.μ + p.γ)*C   
    du
end

# Baseline parameters (standard SIRC values)
const PAR_BASE = (μ = 0.02, α = 365.0/3, δ = 1.0/1.61, γ = 0.35,
                  σ = 0.07874, β0 = 600.0)

# Default initial conditions for simulation
const U0_DEFAULT = [0.3, 1e-3, 0.4, 0.299]

# Range for transmission rate β0
const β0_MIN, β0_MAX = 10.0, 2000.0

# Continuation options
const OPTS_SCAN = BK.ContinuationPar(
    p_min = β0_MIN, p_max = β0_MAX,
    dsmin = 1e-5, dsmax = 10.0,
    max_steps = 1000,
    newton_options = BK.NewtonPar(tol = 1e-9, max_iterations = 20),
    detect_bifurcation = 3,  # monitors eigenvalues
    n_inversion = 8,
    tol_stability = 1e-3
)

# ─── 2. COMPUTE BRANCHES ────────────────────────────────────────────────────
println("1. Running burn-in simulation to find endemic equilibrium at β0 = 600.0")
sol_burnin = DE.solve(
    DE.ODEProblem(sirc_unforced!, U0_DEFAULT, (0.0, 1000.0), PAR_BASE),
    DE.AutoTsit5(DE.Rosenbrock23());
    abstol = 1e-12, reltol = 1e-12, maxiters = 10^7
)
u0_eq = sol_burnin.u[end]
println("Endemic equilibrium found: ", u0_eq)

# Setup bifurcation problem
prob = BK.ODEBifProblem(sirc_unforced!, u0_eq, PAR_BASE, (@optic _.β0);
    record_from_solution = (x, p; k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))

println("\n2. Tracing branches in β0")
# Run upward continuation (increasing β0)
println("Running upward scan (β0: 600.0 -> $β0_MAX)")
opts_up = @set OPTS_SCAN.ds = 1.0
br_up = BK.continuation(prob, BK.PALC(), opts_up; verbosity = 0)

# Run downward continuation (decreasing β0)
println("Running downward scan (β0: 600.0 -> $β0_MIN)")
opts_down = @set OPTS_SCAN.ds = -1.0
br_down = BK.continuation(prob, BK.PALC(), opts_down; verbosity = 0)

# ─── 3. EXTRACT AND MERGE DATA ──────────────────────────────────────────────
# We combine the downward branch (reversed) and the upward branch to get a single continuous curve
branch_param = vcat(reverse(br_down.branch.param), br_up.branch.param[2:end])
branch_S     = vcat(reverse(br_down.branch.S), br_up.branch.S[2:end])
branch_I     = vcat(reverse(br_down.branch.I), br_up.branch.I[2:end])
branch_R     = vcat(reverse(br_down.branch.R), br_up.branch.R[2:end])
branch_C     = vcat(reverse(br_down.branch.C), br_up.branch.C[2:end])
branch_stable = vcat(reverse(br_down.branch.stable), br_up.branch.stable[2:end])

# Identify bifurcation points
# Find the special points from both branches
special_points = Any[]
for sp in br_down.specialpoint
    # Skip endpoints
    sp.type != :endpoint || continue
    push!(special_points, sp)
end
for sp in br_up.specialpoint
    sp.type != :endpoint || continue
    push!(special_points, sp)
end

println("\nBifurcation points detected:")
if isempty(special_points)
    println("  None detected (other than endpoints).")
else
    for (i, sp) in enumerate(special_points)
        println("  Point $i: type = $(sp.type), β0 = $(round(sp.param; digits=4))")
    end
end

# ─── 4. PLOTTING ────────────────────────────────────────────────────────────
println("\n3. Generating bifurcation diagram plots")

# Helper to split a branch into stable and unstable segments for plotting
function split_by_stability(x_vals, y_vals, stable_mask)
    stable_x, stable_y = Float64[], Float64[]
    unstable_x, unstable_y = Float64[], Float64[]
    
    for i in 1:length(x_vals)
        if stable_mask[i]
            push!(stable_x, x_vals[i]);     push!(stable_y, y_vals[i])
            push!(unstable_x, NaN);         push!(unstable_y, NaN)
        else
            push!(unstable_x, x_vals[i]);   push!(unstable_y, y_vals[i])
            push!(stable_x, NaN);           push!(stable_y, NaN)
        end
    end
    return (stable_x, stable_y, unstable_x, unstable_y)
end

# Initialize plots for the four compartments
plt_S = plot(ylabel = "S (Susceptible fraction)")
plt_I = plot(ylabel = "I (Infected fraction)")
plt_R = plot(ylabel = "R (Recovered fraction)")
plt_C = plot(ylabel = "C (Partially immune fraction)")

plots_list = [plt_S, plt_I, plt_R, plt_C]
vars_list  = [branch_S, branch_I, branch_R, branch_C]
labels     = ["S", "I", "R", "C"]

for (plt, vals, name) in zip(plots_list, vars_list, labels)
    sx, sy, ux, uy = split_by_stability(branch_param, vals, branch_stable)
    
    # Plot stable segment in blue, unstable segment in red
    plot!(plt, sx, sy; lc = :blue, lw = 2.0, label = "Stable endemic")
    plot!(plt, ux, uy; lc = :red, lw = 2.0, ls = :dash, label = "Unstable/Unphysical endemic")
    
    # Plot R0 = 1 line (β0 = μ + α ≈ 121.69)
    vline!(plt, [PAR_BASE.μ + PAR_BASE.α]; ls = :dot, lc = :gray, label = "R0 = 1 floor")
    
    # Highlight special points (bifurcations)
    for sp in special_points
        # For values below R0=1, the endemic branch becomes negative (unphysical)
        # We find the closest index in the branch to plot the marker
        idx = argmin(abs.(branch_param .- sp.param))
        val_at_sp = vals[idx]
        scatter!(plt, [sp.param], [val_at_sp]; 
                 mc = :green, ms = 6, marker = :diamond, 
                 label = "$(sp.type) (β0 = $(round(sp.param; digits=1)))")
    end
    
    plot!(plt, xlabel = "β₀ (transmission rate)", xlims = (β0_MIN, β0_MAX), legend = :best)
end

# Combine all 4 subplots into a grid
plt_grid = plot(plt_S, plt_I, plt_R, plt_C, layout = (2, 2), size = (1000, 800),
                plot_title = "Autonomous SIRC Equilibrium Bifurcation Diagram")

output_path = joinpath(@__DIR__, "unforced_equilibrium_branch.png")
savefig(plt_grid, output_path)
println("Saved bifurcation diagram to: $output_path")
println("Done!")
