
#  1-parameter bifurcation diagram for SIRC 
# we set a fixed b0 and sweep epsilon while branching off at pd points


using Plots
import BifurcationKit as BK
import BifurcationKit: @optic, @set
import OrdinaryDiffEq as DE

# headless display 
ENV["GKSwstype"] = "100"

# model
function sirc!(du, u, p, t = 0)
    S, I, R, C, w1, w2 = u
    β = p.β0 * (1.0 + p.ε * w1)
    
    du[1] = p.μ*(1 - S) - β*S*I + p.γ*C                     
    du[2] = β*S*I + p.σ*β*C*I - (p.μ + p.α)*I              
    du[3] = (1 - p.σ)*β*C*I + p.α*I - (p.μ + p.δ)*R     
    du[4] = p.δ*R - β*C*I - (p.μ + p.γ)*C   
    
    du[5] = w1 - 2π*w2 - (w1^2 + w2^2)*w1
    du[6] = 2π*w1 + w2 - (w1^2 + w2^2)*w2
    du
end

# set a fixed beta0 and initial conditions 
const PAR_BASE = (μ = 0.02, α = 365.0/3, δ = 1.0/1.61, γ = 0.35,
                  σ = 0.07874, β0 = 1200.0, ε = 0.001)
const U0_DEFAULT = [0.3, 0.001, 0.4, 0.299, 1.0, 0.0]

# sweep epsilon
const ε_MIN, ε_MAX = 0.0, 0.35

# collocation parameters
const N_MESH = 15
const M_DEG  = 3

# now for the solver...
# continuation parameters
const OPTS_SCAN = BK.ContinuationPar(
    p_min = ε_MIN, p_max = ε_MAX,
    ds = 0.001, dsmin = 1e-6, dsmax = 0.01,
    max_steps = 1000,
    newton_options = BK.NewtonPar(tol = 1e-9, max_iterations = 25),
    detect_bifurcation = 3, # we enable stability check & bifurcation detection
    n_inversion = 8, # bisection levels to pinpoint branching locations
    tol_stability = 1e-3)

# Helper function to record the peak infectious prevalence, amplitude, and period at each step.
const ARGS_PO = (record_from_solution = (x, p; k...) -> begin
        xtt = BK.get_periodic_orbit(p.prob, x, p.p)
        return (I_max     = maximum(xtt[2,:]),
                amplitude = maximum(xtt[2,:]) - minimum(xtt[2,:]),
                period    = BK.getperiod(p.prob, x, p.p))
    end,)

# ─── SECTION 4: INITIAL SHAPE SETUP (BURN-IN) ─────────────────────────────────
# 500-year ODE run to settle on the initial attractor at ε = 0.001
sol_burnin = DE.solve(
    DE.ODEProblem(sirc!, U0_DEFAULT, (0.0, 500.0), PAR_BASE),
    DE.AutoTsit5(DE.Rosenbrock23());
    abstol = 1e-12, reltol = 1e-12, maxiters = 10^7)

# Capture one clean 3-year lap
sol_lap = DE.solve(
    DE.ODEProblem(sirc!, sol_burnin(499.0), (0.0, 3.0), PAR_BASE),
    DE.AutoTsit5(DE.Rosenbrock23());
    abstol = 1e-12, reltol = 1e-10)

# Build the periodic orbit collocation problem
prob_bif = BK.ODEBifProblem(sirc!, sol_lap(0.0), PAR_BASE, (@optic _.ε);
    record_from_solution = (x,p;k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))

pc, ci = BK.generate_ci_problem(BK.Collocation(N_MESH, M_DEG),
                                prob_bif, sol_lap, 1.0)

# ─── SECTION 5: RECURSIVE BIFURCATION TREE COMPUTATION ────────────────────────
println("Starting 1-parameter bifurcation tree computation using manual branch switching...")

# 1. Continue the main Period-1 branch
br1 = BK.continuation(pc, ci, BK.PALC(), OPTS_SCAN;
                      verbosity = 0, normC = BK.norminf, ARGS_PO...)

all_branches = [br1]

# 2. Branch off at Period-Doubling points of the Period-1 branch
pd1_indices = [i for (i, sp) in enumerate(br1.specialpoint) if sp.type == :pd]

for pd_idx in pd1_indices
    sp = br1.specialpoint[pd_idx]
    # Skip disease-free state bifurcation points
    br1.branch.amplitude[sp.step] > 1e-3 || continue
    
    println("  · Branching from Period-1 PD point at ε = $(round(sp.param; digits=5)) to Period-2...")
    br2 = try
        # Branch switching to Period-2 orbit (use_normal_form = false bypasses the Poincaré normal form)
        BK.continuation(br1, pd_idx, OPTS_SCAN; 
                        use_normal_form = false,
                        δp = 0.05, ampfactor = 1.0,
                        verbosity = 0, normC = BK.norminf, ARGS_PO...)
    catch e
        @warn "Branch switching to Period-2 failed" exception=e
        continue
    end
    push!(all_branches, br2)
    
    # 3. Look for PD points on the Period-2 branch to branch off to Period-4
    pd2_indices = [i for (i, sp) in enumerate(br2.specialpoint) if sp.type == :pd]
    for pd_idx2 in pd2_indices
        sp2 = br2.specialpoint[pd_idx2]
        println("    · Branching from Period-2 PD point at ε = $(round(sp2.param; digits=5)) to Period-4...")
        br4 = try
            BK.continuation(br2, pd_idx2, OPTS_SCAN; 
                            use_normal_form = false,
                            δp = 0.05, ampfactor = 1.0,
                            verbosity = 0, normC = BK.norminf, ARGS_PO...)
        catch e
            @warn "Branch switching to Period-4 failed" exception=e
            continue
        end
        push!(all_branches, br4)
        
        # 4. Look for PD points on the Period-4 branch to branch off to Period-8
        pd4_indices = [i for (i, sp) in enumerate(br4.specialpoint) if sp.type == :pd]
        for pd_idx4 in pd4_indices
            sp4 = br4.specialpoint[pd_idx4]
            println("      · Branching from Period-4 PD point at ε = $(round(sp4.param; digits=5)) to Period-8...")
            br8 = try
                BK.continuation(br4, pd_idx4, OPTS_SCAN; 
                                use_normal_form = false,
                                δp = 0.05, ampfactor = 1.0,
                                verbosity = 0, normC = BK.norminf, ARGS_PO...)
            catch e
                @warn "Branch switching to Period-8 failed" exception=e
                continue
            end
            push!(all_branches, br8)
        end
    end
end

println("Bifurcation tree computed successfully!")

# ─── SECTION 6: PLOTTING THE TREE ─────────────────────────────────────────────
# Set up a plot frame with ε on the horizontal axis and maximum infected population on the vertical axis.
plt = plot(xlabel = "ε (seasonality amplitude)",
           ylabel = "Max Infected Population (I_max)",
           title  = "SIRC 1-Parameter Bifurcation Tree (β₀ = 1200.0)",
           size   = (800, 500), legend = :topleft)

# Plot each branch with color corresponding to its period level
for br in all_branches
    ε = collect(br.branch.param)
    I_max = collect(br.branch.I_max)
    
    # Identify the average period of the branch to assign color and labels
    avg_period = sum(br.branch.period) / length(br.branch.period)
    
    col, label_name = if avg_period < 1.5
        :black, "Period-1"
    elseif avg_period < 3.0
        :blue, "Period-2"
    elseif avg_period < 6.0
        :red, "Period-4"
    else
        :orange, "Period-8"
    end
    
    plot!(plt, ε, I_max; lc = col, lw = 1.5, label = label_name)
end

# Save the resulting bifurcation diagram tree
output_path = "one_parameter_bifurcation_tree.png"
savefig(plt, output_path)
println("Saved bifurcation tree plot to: $output_path")
