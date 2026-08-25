# ═════════════════════════════════════════════════════════════════════════════
#  Automated Bifurcation Diagram for the Seasonally-Forced SIRC Influenza Model
#  
#  This script reproduces Figure 3(a) from Casagrandi et al. (2006): a 2-parameter 
#  bifurcation diagram in the (ε, β₀) plane. 
#  - ε: Amplitude of seasonal forcing (degree of seasonality)
#  - β₀: Baseline transmission rate
#
#  We focus on the Period-1 attractor's boundaries:
#  - f₁⁽¹⁾: Period-doubling (flip) bifurcations, marked with white diamonds (gpd).
#  - t⁽¹⁾: Fold (tangent) bifurcations, marked with black squares (cusps).
# ═════════════════════════════════════════════════════════════════════════════

using Logging
using Serialization

# ─── WARNING FILTER ──────────────────────────────────────────────────────────
# Suppress benign, repetitive warnings from BifurcationKit's internal solvers 
# (e.g. Floquet coefficient warnings during bisection refinement).
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

# Tell Plots to use a headless display to run without opening windows
ENV["GKSwstype"] = "100"


# ─── SECTION 1: THE MATHEMATICAL MODEL ────────────────────────────────────────
# In a seasonally-forced model, the transmission rate varies as:
#      β(t) = β₀ * (1 + ε * cos(2π * t))
# Because this system depends explicitly on time, it is non-autonomous. 
# BifurcationKit requires an autonomous system to trace periodic orbits.
#
# We solve this by coupling the SIRC equations to a Stuart-Landau oscillator 
# (w₁, w₂) which lives on a stable limit cycle of radius 1 and period 1:
#      dw₁/dt = w₁ - 2π*w₂ - (w₁² + w₂²)*w₁
#      dw₂/dt = 2π*w₁ + w₂ - (w₁² + w₂²)*w₂
# Under this oscillator, w₁(t) = cos(2π * t). Thus, we can rewrite:
#      β(t) = β₀ * (1 + ε * w₁)
# Making the combined 6-dimensional system completely autonomous.
#
function sirc!(du, u, p, t = 0)
    S, I, R, C, w1, w2 = u
    β = p.β0 * (1.0 + p.ε * w1)
    
    # SIRC equations
    du[1] = p.μ*(1 - S) - β*S*I + p.γ*C                     # dS/dt
    du[2] = β*S*I + p.σ*β*C*I - (p.μ + p.α)*I               # dI/dt
    du[3] = (1 - p.σ)*β*C*I + p.α*I - (p.μ + p.δ)*R         # dR/dt
    du[4] = p.δ*R - β*C*I - (p.μ + p.γ)*C                   # dC/dt
    
    # Stuart-Landau oscillator equations (periodic forcing generator)
    du[5] = w1 - 2π*w2 - (w1^2 + w2^2)*w1
    du[6] = 2π*w1 + w2 - (w1^2 + w2^2)*w2
    du
end

# Default baseline parameters matching Casagrandi et al. (2006)
const PAR_BASE = (μ = 0.02, α = 365.0/3, δ = 1.0/1.61, γ = 0.35,
                  σ = 0.07874, β0 = 600.0, ε = 0.01)

# Default initial conditions for simulation burn-in
const U0_DEFAULT = [0.3, 1e-3, 0.4, 0.299, 1.0, 0.0]


# ─── SECTION 2: GRID SEEDS & AXIS BOUNDS ──────────────────────────────────────
# The bounding box for our 2-parameter bifurcation diagrams.
const ε_MIN, ε_MAX = 0.0, 0.35
const β_MIN, β_MAX = 0.0, 2000.0

# Finer horizontal scans in β₀.
# Instead of doing vertical sweeps, we run horizontal scans at small intervals of β₀.
# This finds the bifurcation points at many different levels, letting us seed
# all curves (including the upper-middle folds and the lower-limit tips).
const SEED_β0 = [
    130.0, 150.0, 200.0, 250.0, 300.0, 400.0, 600.0, 750.0, 
    900.0, 1050.0, 1200.0, 1350.0, 1500.0, 1650.0, 1750.0, 1850.0
]

# Grid parameters for Orthogonal Collocation
# We discretize the periodic orbit into 15 intervals using degree-3 polynomials
const N_MESH = 15
const M_DEG  = 3


# ─── SECTION 3: SOLVER OPTIONS ────────────────────────────────────────────────
# 1. Options for the 1-Parameter continuation (Horizontal scans varying ε)
const OPTS_SCAN = BK.ContinuationPar(
    p_min = ε_MIN, p_max = ε_MAX,
    ds = 0.001, dsmin = 1e-6, dsmax = 0.01,
    max_steps = 500,
    newton_options = BK.NewtonPar(tol = 1e-9, max_iterations = 15),
    detect_bifurcation = 3,       # Enable bisection-refined event detection
    n_inversion = 6,              # Bisection levels for locating bifurcation points
    tol_stability = 1e-3)

# 2. Options for 2-Parameter Period-Doubling (Flip) Continuation
const OPTS_PD = BK.ContinuationPar(
    p_min = 125.0, p_max = β_MAX,
    ds = 5.0, dsmin = 1e-4, dsmax = 50.0,
    max_steps = 800,
    newton_options = BK.NewtonPar(tol = 1e-8, max_iterations = 25),
    detect_bifurcation = 0)       # Set to 0 because we only record codim-2 crossings along the curve

# 3. Options for 2-Parameter Fold (Tangent) Continuation
const OPTS_FOLD = BK.ContinuationPar(
    p_min = 125.0, p_max = β_MAX,
    ds = 5.0, dsmin = 1e-5, dsmax = 50.0,
    max_steps = 800,
    newton_options = BK.NewtonPar(tol = 1e-7, max_iterations = 30),
    detect_bifurcation = 0)

# Helper to extract the maximum/minimum infected amplitude and period at each step.
# Useful for verifying that the solver is tracking the correct periodic orbit.
const ARGS_PO = (record_from_solution = (x, p; k...) -> begin
        xtt = BK.get_periodic_orbit(p.prob, x, p.p)
        return (I_max  = maximum(xtt[2,:]),
                I_min  = minimum(xtt[2,:]),
                period = BK.getperiod(p.prob, x, p.p))
    end,)


# ─── SECTION 4: 1-PARAMETER CONTINUATION (HORIZONTAL SWEEP) ──────────────────
# Given a fixed β₀ value, this function:
# 1. Runs a long ODE simulation (500 years) to burn in onto the attractor.
# 2. Runs a 3-year simulation to extract a clean limit cycle guess.
# 3. Sets up a Collocation problem and continues the periodic orbit as ε varies.
#
function horizontal_scan(β0_val; par = PAR_BASE, u0 = U0_DEFAULT)
    p = @set par.β0 = β0_val

    # Run ODE solver to damp out transients
    sol_burnin = DE.solve(
        DE.ODEProblem(sirc!, u0, (0.0, 500.0), p),
        DE.AutoTsit5(DE.Rosenbrock23());
        abstol = 1e-12, reltol = 1e-12, maxiters = 10^7)

    # Record one clean period-1 orbit
    sol_lap = DE.solve(
        DE.ODEProblem(sirc!, sol_burnin(499.0), (0.0, 3.0), p),
        DE.AutoTsit5(DE.Rosenbrock23());
        abstol = 1e-12, reltol = 1e-10)

    # Define the bifurcation problem for periodic orbits
    prob_bif = BK.ODEBifProblem(sirc!, sol_lap(0.0), p, (@optic _.ε);
        record_from_solution = (x,p;k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))

    # Set up collocation boundary value solver
    pc, ci = BK.generate_ci_problem(BK.Collocation(N_MESH, M_DEG),
                                    prob_bif, sol_lap, 1.0)

    # Continue the periodic orbit along the ε axis
    return BK.continuation(pc, ci, BK.PALC(), OPTS_SCAN;
                           verbosity = 0, normC = BK.norminf, ARGS_PO...)
end


# ─── SECTION 5: 2-PARAMETER CONTINUATION (BOTH DIRECTIONS) ───────────────────
# To prevent solvers from stalling due to numerical orientation issues, 
# we run the 2-parameter continuation in BOTH directions (positive and negative ds) 
# as separate, independent calls.
#
function _continue_both_dirs(br, ind, lens2, opts_base, kw)
    branches = Any[]
    for sgn in (+1.0, -1.0)
        opts = @set opts_base.ds = sgn * abs(opts_base.ds)
        try
            brc = BK.continuation(br, ind, lens2, opts; kw...)
            push!(branches, brc)
        catch e
            # Log stall silently; stalling is normal when the curve exits the bounding box
        end
    end
    return branches
end

# Continuation wrapper for Period-Doubling curve
function continue_pd_curve(br, ind, lens2, opts)
    _continue_both_dirs(br, ind, lens2, opts, (
        alg = BK.PALC(tangent = BK.Bordered()),
        verbosity = 0, normC = BK.norminf,
        jacobian_ma = BK.MinAug(),
        start_with_eigen = true,
        detect_codim2_bifurcation = 2,
        bothside = false))
end

# Continuation wrapper for Fold (Tangent) curve
function continue_fold_curve(br, ind, lens2, opts)
    _continue_both_dirs(br, ind, lens2, opts, (
        alg = BK.PALC(tangent = BK.Bordered()),
        verbosity = 0, normC = BK.norminf,
        jacobian_ma = BK.MinAug(),
        start_with_eigen = false,       # Must be false for FloquetCollocation
        detect_codim2_bifurcation = 2,
        usehessian = true,
        bothside = false))
end


# ─── SECTION 6: SWEEP PIPELINE ────────────────────────────────────────────────
# Struct to serialize the completed sweep data
struct BifDiagram
    pd_curves      :: Vector{Any}
    fold_curves    :: Vector{Any}
    codim2_points  :: Vector{NamedTuple}
    scans          :: Vector{Any}
end

# Runs the complete bifurcation sweep over the horizontal seeds
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
        
        # Look for PD (period-doubling) and BP (bifurcation point/fold candidate) indices
        pd_idx = [i for (i, sp) in enumerate(br.specialpoint) if sp.type == :pd]
        bp_idx = [i for (i, sp) in enumerate(br.specialpoint) if sp.type == :bp]
        println("  PDs: $(length(pd_idx))   Folds (BPs): $(length(bp_idx))")

        # Trace Period-Doubling curves in 2D parameter space (varying ε and β₀)
        for i in pd_idx
            sp = br.specialpoint[i]
            println("  · PD at ε = $(round(sp.param; digits=5)): continuing...")
            brcs = continue_pd_curve(br, i, (@optic _.β0), OPTS_PD)
            append!(pd_curves, brcs)
            for brc in brcs
                # Record generalized period-doublings (gpd)
                _collect_codim2!(store = codim2_pts, brc = brc, source = :pd, types = (:gpd,))
            end
        end
        
        # Trace Fold curves in 2D parameter space (varying ε and β₀)
        for i in bp_idx
            sp = br.specialpoint[i]
            println("  · Fold at ε = $(round(sp.param; digits=5)): continuing...")
            brcs = continue_fold_curve(br, i, (@optic _.β0), OPTS_FOLD)
            append!(fold_curves, brcs)
            for brc in brcs
                # Record fold cusps (cusp and R1 resonance points represent the same fold cusp)
                _collect_codim2!(store = codim2_pts, brc = brc, source = :fold, types = (:cusp, :R1))
            end
        end
    end

    println("\nSweep complete: $(length(pd_curves)) PD, $(length(fold_curves)) Folds, $(length(codim2_pts)) Codim-2")
    return BifDiagram(pd_curves, fold_curves, codim2_pts, scans)
end

# Collect codimension-2 points (gpd / cusp) located along the curves
function _collect_codim2!(; store, brc, source, types)
    for sp in brc.specialpoint
        sp.type in types || continue
        # Ignore starting point detections (which are generic codim-1 seeds)
        sp.step > 5 || continue
        
        step = clamp(sp.step, 1, length(brc.branch))
        push!(store, (type = sp.type,
                      ε    = brc.branch.ε[step],
                      β    = brc.branch.β0[step],
                      source = source))
    end
end


# ─── SECTION 7: PLOTTING & FILTERING ──────────────────────────────────────────

# Extract and filter parameter coordinates inside our bounding box
function branch_coords(brc)
    ε = collect(brc.branch.ε)
    β = collect(brc.branch.β0)
    mask = @. (ε ≥ ε_MIN) & (ε ≤ ε_MAX) & (β ≥ β_MIN) & (β ≤ β_MAX)
    return ε[mask], β[mask]
end

n_points_in_box(brc) = length(branch_coords(brc)[1])

# Deduplicate codim-2 markers close to each other
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

# Plot Flip (PD) bifurcation curves
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
    for c2 in deduplicate_codim2(diag.codim2_points)
        c2.source == :pd && c2.type == :gpd || continue
        scatter!(plt, [c2.ε], [c2.β]; ms = 6, m = :diamond, mc = :white, msc = :black, msw = 1.2, label = false)
    end
    hline!(plt, [PAR_BASE.μ + PAR_BASE.α]; ls = :dot, lc = :gray, label = "R₀ = 1")
    savefig(plt, "f_curves.png")
    return plt
end

# Plot Fold (Tangent) bifurcation curves
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
        # Hybrid Filter: 
        # - Keep long curves (>= 50 steps) representing main lower folds.
        # - Keep short curves (>= 8 steps) at high baseline transmission (>= 1300.0) 
        #   representing the upper-middle components that stall early.
        (len >= 50 || (len >= 8 && max_β >= 1300.0)) || continue
        n_points_in_box(brc) ≥ min_points || continue
        
        ε, β = branch_coords(brc)
        scatter!(plt, ε, β; ms = 1.0, mc = :blue, msw = 0.0, label = labeled ? false : "t⁽¹⁾")
        labeled = true
    end
    for c2 in deduplicate_codim2(diag.codim2_points)
        c2.source == :fold && (c2.type == :cusp || c2.type == :R1) || continue
        scatter!(plt, [c2.ε], [c2.β]; ms = 6, m = :square, mc = :black, msc = :black, label = false)
    end
    hline!(plt, [PAR_BASE.μ + PAR_BASE.α]; ls = :dot, lc = :gray, label = "R₀ = 1")
    savefig(plt, "t_curves.png")
    return plt
end

# Plot Combined Bifurcation Diagram (Overlaying Flip & Fold curves)
function plot_combined_diagram(diag::BifDiagram; min_points = 5)
    plt = plot(xlabel = "ε (degree of seasonality)",
               ylabel = "β₀ (baseline transmission)",
               title  = "Seasonally Forced SIRC Bifurcation Diagram (Period-1 Curves)",
               xlims  = (ε_MIN, ε_MAX), ylims = (β_MIN, β_MAX),
               size   = (700, 600), legend = :topright)
    
    # 1. R0 = 1 floor
    hline!(plt, [PAR_BASE.μ + PAR_BASE.α]; ls = :dash, lc = :gray, label = "R₀ = 1 floor")
    
    # 2. PD curves (black scatter)
    pd_labeled = false
    for brc in diag.pd_curves
        n_points_in_box(brc) ≥ min_points || continue
        ε, β = branch_coords(brc)
        scatter!(plt, ε, β; ms = 1.0, mc = :black, msw = 0.0, label = pd_labeled ? false : "f₁⁽¹⁾ (Period Doubling)")
        pd_labeled = true
    end
    
    # 3. Fold curves (blue scatter with Hybrid Filter)
    fold_labeled = false
    for brc in diag.fold_curves
        len = length(brc.branch)
        max_β = maximum(brc.branch.β0)
        (len >= 50 || (len >= 8 && max_β >= 1300.0)) || continue
        n_points_in_box(brc) ≥ min_points || continue
        ε, β = branch_coords(brc)
        scatter!(plt, ε, β; ms = 1.0, mc = :blue, msw = 0.0, label = fold_labeled ? false : "t⁽¹⁾ (Fold / Tangent)")
        fold_labeled = true
    end
    
    # 4. Codim-2 bifurcation markers
    c2_points = deduplicate_codim2(diag.codim2_points)
    gpd_labeled = false
    cusp_labeled = false
    for c2 in c2_points
        if c2.type == :gpd
            scatter!(plt, [c2.ε], [c2.β]; ms = 7, m = :diamond, mc = :white, msc = :black, msw = 1.2,
                     label = gpd_labeled ? false : "gpd (generalized PD)")
            gpd_labeled = true
        elseif c2.type == :cusp || c2.type == :R1
            scatter!(plt, [c2.ε], [c2.β]; ms = 6, m = :square, mc = :black, msc = :black, msw = 0.0,
                     label = cusp_labeled ? false : "cusp (fold cusp)")
            cusp_labeled = true
        end
    end
    
    output_path = "/Users/danielarenee/Desktop/Infection-Modeling-with-Mutation/bifurcation_map_SIRC.png"
    savefig(plt, output_path)
    println("Saved combined bifurcation map to: $output_path")
    return plt
end


# ─── SECTION 8: EXECUTION ENTRYPOINT ──────────────────────────────────────────
# Check if a pre-computed sweep database exists. If yes, deserialize and plot.
# If not, run a new sweep and save the database to `diag_clean.jls`.
diag_file = joinpath(@__DIR__, "diag_clean.jls")

diag = if isfile(diag_file)
    println("Loading pre-computed sweep from: $diag_file")
    deserialize(diag_file)
else
    println("No pre-computed sweep found. Running automated sweep...")
    d = run_bifurcation_sweep()
    serialize(diag_file, d)
    println("Saved sweep to: $diag_file")
    d
end

# Generate and save all plot images
plot_f_curves(diag)
plot_t_curves(diag)
plot_combined_diagram(diag)
println("All tasks finished successfully!")
