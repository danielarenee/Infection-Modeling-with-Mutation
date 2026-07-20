# Bifurcation diagram recreation from Casagrandi figure 3

using Logging
using Serialization

# Suppress beningn warnings from BK's solvers 
struct WarningFilterLogger <: AbstractLogger
    parent::AbstractLogger
end
Logging.shouldlog(logger::WarningFilterLogger, level, _module, group, id) = true
Logging.min_enabled_level(logger::WarningFilterLogger) = Logging.min_enabled_level(logger.parent)
Logging.handle_message(logger::WarningFilterLogger, level, message, _module, group, id, file, line; kwargs...) = begin
    msg_str = string(message)
    if level == Logging.Warn && (occursin("should be zero", msg_str) || occursin("PD-Iooss", msg_str))
        return
    end
    Logging.handle_message(logger.parent, level, message, _module, group, id, file, line; kwargs...)
end
global_logger(WarningFilterLogger(global_logger()))

using Plots
import BifurcationKit as BK
import BifurcationKit: @optic, @set
import OrdinaryDiffEq as DE

ENV["GKSwstype"] = "100"

# SIRC MODEL --
function sirc!(du, u, p, t = 0)
    S, I, R, C, w1, w2 = u
    β = p.β0 * (1.0 + p.ε * w1)
    
    # SIRC equations
    du[1] = p.μ*(1 - S) - β*S*I + p.γ*C                     
    du[2] = β*S*I + p.σ*β*C*I - (p.μ + p.α)*I              
    du[3] = (1 - p.σ)*β*C*I + p.α*I - (p.μ + p.δ)*R     
    du[4] = p.δ*R - β*C*I - (p.μ + p.γ)*C   
    # oscillator equations 
    du[5] = w1 - 2π*w2 - (w1^2 + w2^2)*w1
    du[6] = 2π*w1 + w2 - (w1^2 + w2^2)*w2
    du
end

# default baseline parameters
const PAR_BASE = (μ = 0.02, α = 365.0/3, δ = 1.0/1.61, γ = 0.35,
                  σ = 0.07874, β0 = 600.0, ε = 0.01)

# default initial conditions for simulation
const U0_DEFAULT = [0.3, 1e-3, 0.4, 0.299, 1.0, 0.0]


# GRID SEEDS & AXIS BOUNDS --
# bounding for eps and beta0
const ε_MIN, ε_MAX = 0.0, 0.35
const β_MIN, β_MAX = 0.0, 2000.0

# beta0 values for the scans
const SEED_β0 = [
    130.0, 150.0, 200.0, 250.0, 300.0, 400.0, 600.0, 750.0, 
    900.0, 1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1250.0, 1300.0, 1350.0, 
    1400.0, 1450.0, 1500.0, 1550.0, 1600.0, 1650.0, 1700.0, 1750.0, 1800.0,
    1850.0, 1950.0
]

# TEST SEEDS (FASTER)
#const SEED_β0 = [
#    130.0, 150.0, 200.0, 300.0, 400.0, 600.0, 750.0, 
#    900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0,
#    1600.0, 1700.0, 1800.0, 1950.0
#]

#const SEED_β0 = [
#    130.0, 200.0, 350.0, 500.0, 750.0, 900.0, 1100.0, 1300.0, 1450.0, 1600.0, 1750.0, 1900.0
#]


# grid parameters for collocation
# aka: we discretize the periodic orbit into 15 intervals using degree-3 polynomials
const N_MESH = 15
const M_DEG  = 3


# SOLVER OPTIONS --

# options for the 1-parameter continuation (varying epsilon)
const OPTS_SCAN = BK.ContinuationPar(
    p_min = ε_MIN, p_max = ε_MAX,
    ds = 0.001, dsmin = 1e-6, dsmax = 0.01,
    max_steps = 500,
    newton_options = BK.NewtonPar(tol = 1e-9, max_iterations = 20),
    detect_bifurcation = 3, # monitors eigenvalues at every step 
    n_inversion = 6, # bisection levels for locating bifurcation points
    tol_stability = 1e-3)

# options for 2-parameter period-doubling continuation
const OPTS_PD = BK.ContinuationPar(
    p_min = 125.0, p_max = β_MAX,
    ds = 5.0, dsmin = 1e-4, dsmax = 50.0,
    max_steps = 800,
    newton_options = BK.NewtonPar(tol = 1e-8, max_iterations = 25),
    detect_bifurcation = 0)

# options for 2-parameter fold (tangent) continuation
const OPTS_FOLD = BK.ContinuationPar(
    p_min = 125.0, p_max = β_MAX,
    ds = 5.0, dsmin = 1e-6, dsmax = 50.0,
    max_steps = 800,
    newton_options = BK.NewtonPar(tol = 1e-7, max_iterations = 30),
    detect_bifurcation = 0)


# 1-PARAMETER CONTINUATION (HORIZONTAL SWEEP) --
# Given a fixed beta0 value, this function:
# 1. Runs a long ODE simulation (500 years) to burn in onto the attractor
# 2. Runs a 3-year simulation to extract a clean limit cycle guess
# 3. Sets up a Collocation problem and continues the periodic orbit as ε varies
function horizontal_scan(β0_val; par = PAR_BASE, u0 = U0_DEFAULT)
    p = @set par.β0 = β0_val

    # run ODE solver to damp out transients
    sol_burnin = DE.solve(
        DE.ODEProblem(sirc!, u0, (0.0, 500.0), p),
        DE.AutoTsit5(DE.Rosenbrock23());
        abstol = 1e-12, reltol = 1e-12, maxiters = 10^7)

    # record one clean period-1 orbit
    sol_lap = DE.solve(
        DE.ODEProblem(sirc!, sol_burnin(499.0), (0.0, 3.0), p),
        DE.AutoTsit5(DE.Rosenbrock23());
        abstol = 1e-12, reltol = 1e-10)

    # define the bifurcation problem for periodic orbits
    prob_bif = BK.ODEBifProblem(sirc!, sol_lap(0.0), p, (@optic _.ε);
        record_from_solution = (x,p;k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))

    # set up collocation boundary value solver
    pc, ci = BK.generate_ci_problem(BK.Collocation(N_MESH, M_DEG),
                                    prob_bif, sol_lap, 1.0)

    # continue the periodic orbit along the ε axis
    return BK.continuation(pc, ci, BK.PALC(), OPTS_SCAN;
                           verbosity = 0, normC = BK.norminf)
end


# 2-PARAMETER CONTINUATION (BOTH DIRECTIONS) --
function _continue_both_dirs(br, ind, lens2, opts_base, kw)
    branches = Any[]
    for sgn in (+1.0, -1.0)
        opts = @set opts_base.ds = sgn * abs(opts_base.ds)
        try
            brc = BK.continuation(br, ind, lens2, opts; kw...)
            push!(branches, brc)
        catch e
        end
    end
    return branches
end

# continuation wrapper for PD curve
function continue_pd_curve(br, ind, lens2, opts)
    _continue_both_dirs(br, ind, lens2, opts, (
        alg = BK.PALC(tangent = BK.Bordered()),
        verbosity = 0, normC = BK.norminf,
        jacobian_ma = BK.MinAug(),
        start_with_eigen = true,
        detect_codim2_bifurcation = 2,
        bothside = false))
end

# cntinuation wrapper for tangent curve
function continue_fold_curve(br, ind, lens2, opts)
    _continue_both_dirs(br, ind, lens2, opts, (
        alg = BK.PALC(tangent = BK.Bordered()),
        verbosity = 0, normC = BK.norminf,
        jacobian_ma = BK.MinAug(),
        start_with_eigen = false,  
        detect_codim2_bifurcation = 2,
        usehessian = true,
        bothside = false))
end


# SWEEP PIPELINE --
# struct to save the sweep data
struct BifDiagram
    pd_curves      :: Vector{Any}
    fold_curves    :: Vector{Any}
    codim2_points  :: Vector{NamedTuple}
    scans          :: Vector{Any}
end

# runs the complete bifurcation sweep over the horizontal seeds
function run_bifurcation_sweep(β0_seeds = SEED_β0)
    pd_curves     = Any[]
    fold_curves   = Any[]
    codim2_pts    = NamedTuple[]
    scans         = Any[]

    for β0v in β0_seeds
        println("─── Horizontal scan at β₀ = $β0v ───")
        br = try
            horizontal_scan(β0v)
        catch e
            @warn "Horizontal scan failed at β₀ = $β0v" exception=e
            continue
        end
        push!(scans, br)
        
        # look for PD and BP indices
        pd_idx = [i for (i, sp) in enumerate(br.specialpoint) if sp.type == :pd]
        bp_idx = [i for (i, sp) in enumerate(br.specialpoint) if sp.type == :bp]
        println("  PDs: $(length(pd_idx))   Folds (BPs): $(length(bp_idx))")

        # trace PD curves by varying epsilon and beta0
        for i in pd_idx
            sp = br.specialpoint[i]
            # skip disease-free state bifurcation points (which have near-zero oscillations)
            br.branch.amplitude[sp.step] > 1e-3 || continue
            
            println("  · PD at ε = $(round(sp.param; digits=5)): continuing...")
            brcs = continue_pd_curve(br, i, (@optic _.β0), OPTS_PD)
            append!(pd_curves, brcs)
            for brc in brcs
                # record generalized period-doublings (gpd)
                _collect_codim2!(store = codim2_pts, brc = brc, source = :pd, types = (:gpd,))
            end
        end
        
        # trace BP curves by varying epsilon and beta0
        for i in bp_idx
            sp = br.specialpoint[i]
            # skip disease-free state bifurcation points (which have near-zero oscillations)
            br.branch.amplitude[sp.step] > 1e-3 || continue
            
            println("  · Fold at ε = $(round(sp.param; digits=5)): continuing...")
            brcs = continue_fold_curve(br, i, (@optic _.β0), OPTS_FOLD)
            append!(fold_curves, brcs)
            for brc in brcs
                # record fold cusps 
                _collect_codim2!(store = codim2_pts, brc = brc, source = :fold, types = (:cusp, :R1))
            end
        end
    end

    println("\nSweep complete: $(length(pd_curves)) PD, $(length(fold_curves)) Folds, $(length(codim2_pts)) Codim-2")
    return BifDiagram(pd_curves, fold_curves, codim2_pts, scans)
end

# collect codimension-2 points (gpd / cusp) located along the curves
function _collect_codim2!(; store, brc, source, types)
    for sp in brc.specialpoint
        sp.type in types || continue
        # ignore starting point detections (which are generic codim-1 seeds)
        sp.step > 5 || continue
        
        step = clamp(sp.step, 1, length(brc.branch))
        push!(store, (type = sp.type,
                      ε    = brc.branch.ε[step],
                      β    = brc.branch.β0[step],
                      source = source))
    end
end


# PLOTTING --

# extract and filter parameter coordinates inside the bounds
function branch_coords(brc)
    ε = collect(brc.branch.ε)
    β = collect(brc.branch.β0)
    mask = @. (ε ≥ ε_MIN) & (ε ≤ ε_MAX) & (β ≥ β_MIN) & (β ≤ β_MAX)
    return ε[mask], β[mask]
end

n_points_in_box(brc) = length(branch_coords(brc)[1])

# deduplicate codim-2 markers close to each other
function deduplicate_codim2(pts; εtol = 0.005, βtol = 20.0)
    kept = eltype(pts)[]
    for p in pts
        (ε_MIN ≤ p.ε ≤ ε_MAX) && (β_MIN ≤ p.β ≤ β_MAX) || continue
        is_dup = any(kept) do q
            (q.type == p.type) && (q.source == p.source) &&
            (abs(q.ε - p.ε) < εtol) && (abs(q.β - p.β) < βtol)
        end
        is_dup || push!(kept, p)
    end
    return kept
end

# plot PD
function plot_f_curves(diag::BifDiagram; min_points = 5)
    plt = plot(xlabel = "ε (degree of seasonality)",
               ylabel = "β₀ (baseline transmission)",
               title  = "f₁⁽¹⁾ — period-doubling of period-1",
               xlims  = (ε_MIN, ε_MAX), ylims = (β_MIN, β_MAX),
               size   = (700, 600), legend = :topleft)
    labeled = false
    for brc in diag.pd_curves
        n_points_in_box(brc) ≥ min_points || continue
        ε, β = branch_coords(brc)
        scatter!(plt, ε, β; ms = 1.0, mc = :black, msw = 0.0, label = labeled ? false : "f₁⁽¹¹⁾")
        labeled = true
    end
    hline!(plt, [PAR_BASE.μ + PAR_BASE.α]; ls = :dot, lc = :gray, label = "R₀ = 1")
    savefig(plt, "f_curves.png")
    return plt
end

# plot BP
function plot_t_curves(diag::BifDiagram; min_points = 5)
    plt = plot(xlabel = "ε (degree of seasonality)",
               ylabel = "β₀ (baseline transmission)",
               title  = "t⁽¹⁾ — fold of period-1",
               xlims  = (ε_MIN, ε_MAX), ylims = (β_MIN, β_MAX),
               size   = (700, 600), legend = :topleft)
    labeled = false
    for brc in diag.fold_curves
        len = length(brc.branch)
        max_β = maximum(brc.branch.β0)
        # we filter to keep long main folds (>= 40 steps) 
        (len >= 40 || (len >= 15 && max_β >= 1300.0)) || continue
        n_points_in_box(brc) ≥ min_points || continue
        
        ε, β = branch_coords(brc)
        scatter!(plt, ε, β; ms = 1.0, mc = :black, msw = 0.0, label = labeled ? false : "t⁽¹⁾")
        labeled = true
    end
    hline!(plt, [PAR_BASE.μ + PAR_BASE.α]; ls = :dot, lc = :gray, label = "R₀ = 1")
    savefig(plt, "t_curves.png")
    return plt
end

# EXECUTION --
# run the automated sweep to compute the bifurcation diagrams
println("Running automated bifurcation sweep...")
diag = run_bifurcation_sweep()

# save the computed sweep data as a JLS database
diag_file = joinpath(@__DIR__, "diag_clean.jls")
serialize(diag_file, diag)
println("Saved sweep to: $diag_file")

# generate and save all plot images
plot_f_curves(diag)
plot_t_curves(diag)
println("All tasks finished successfully!")
