# ═════════════════════════════════════════════════════════════════════════════
#  Bifurcation Sweep for the Autonomous SIRCmw Model (No Seasonal Forcing)
#  
#  This script sweeps the 2D parameter space (tilde_eps, β₀) of the 
#  autonomous SIRCmw model (prevalence-dependent mutation/waning, η = 0).
#  It uses the finite-difference Trapezoid Method to trace flip (PD) and LP curves.
# ═════════════════════════════════════════════════════════════════════════════

using Logging
using Serialization
using DelimitedFiles

# ─── WARNING FILTER ──────────────────────────────────────────────────────────
# Suppress benign, repetitive warnings from BifurcationKit's internal solvers
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

# ─── SECTION 1: THE SIRCmw MODEL ──────────────────────────────────────────────
# Autonomous 4D model with prevalence-dependent cross-immunity erosion (eps = tilde_eps / SI_0)
# We add a regularization term (1e-4) to the infected class (du[2]) to prevent
# unphysical excursions to extremely small scales (~10^-29) which crash the solvers.
const REG = 1e-4

function sircmw!(du, u, p, t = 0)
    S, I, R, C = u
    b = p.β0
    eps = p.tilde_eps / 0.000178 # SI_0 is 0.000178
    
    du[1] = p.μ*(1 - S) - b*S*I + (1.0 + eps*S*I)*p.γ*C
    du[2] = b*S*I + p.σ*b*C*I - (p.μ + p.α)*I + REG
    du[3] = (1.0 - p.σ)*b*C*I + p.α*I - p.μ*R - (1.0 + eps*S*I)*p.δ*R
    du[4] = (1.0 + eps*S*I)*p.δ*R - b*C*I - p.μ*C - (1.0 + eps*S*I)*p.γ*C
    du
end

# Default baseline parameters (calibrated to the endemic state)
const PAR_BASE = (μ = 0.02, α = 365.0/3, δ = 1.0/1.61, γ = 0.35,
                  σ = 0.07874, β0 = 750.0, tilde_eps = 2.0)

const U0_DEFAULT = [0.2, 0.001, 0.499, 0.3]

# ─── SECTION 2: GRID SEEDS & AXIS BOUNDS ──────────────────────────────────────
const tilde_eps_MIN, tilde_eps_MAX = 0.0, 2.0
const β_MIN, β_MAX = 0.0, 2000.0

# baseline transmission rates for horizontal scans where oscillations are active
const SEED_β0 = [750.0, 900.0, 1150.0, 1300.0, 1450.0, 1600.0]

# Trapezoid grid size
const M_GRID = 150

# ─── SECTION 3: SOLVER OPTIONS ────────────────────────────────────────────────
# 1-parameter horizontal scan options (sweeping backwards from tilde_eps = 2.0 to 0.0)
const OPTS_SCAN = BK.ContinuationPar(
    p_min = tilde_eps_MIN, p_max = tilde_eps_MAX,
    ds = -0.002, dsmin = 1e-6, dsmax = 0.03,
    max_steps = 1000,
    newton_options = BK.NewtonPar(tol = 1e-9, max_iterations = 25, linesearch = true),
    detect_bifurcation = 3,
    n_inversion = 6,
    tol_stability = 1e-3,
    nev = 6)

# 2-parameter period-doubling continuation options
const OPTS_PD = BK.ContinuationPar(
    p_min = 125.0, p_max = β_MAX,
    ds = 5.0, dsmin = 1e-5, dsmax = 50.0,
    max_steps = 1000,
    newton_options = BK.NewtonPar(tol = 1e-8, max_iterations = 25, linesearch = true),
    detect_bifurcation = 0,
    nev = 6)

# 2-parameter fold (tangent) continuation options
const OPTS_FOLD = BK.ContinuationPar(
    p_min = 125.0, p_max = β_MAX,
    ds = 5.0, dsmin = 1e-6, dsmax = 50.0,
    max_steps = 1000,
    newton_options = BK.NewtonPar(tol = 1e-7, max_iterations = 35, linesearch = true),
    detect_bifurcation = 0,
    nev = 6)

# Recorder to collect orbit stats
const ARGS_PO = (record_from_solution = (x, p; k...) -> begin
    xtt = BK.get_periodic_orbit(p.prob, x, p.p)
    return (I_max     = maximum(xtt[2,:]),
            I_min     = minimum(xtt[2,:]),
            amplitude = maximum(xtt[2,:]) - minimum(xtt[2,:]),
            period    = BK.getperiod(p.prob, x, p.p))
end,)

# ─── SECTION 4: 1-PARAMETER CONTINUATION ──────────────────────────────────────
function horizontal_scan(β0_val; par = PAR_BASE, u0 = U0_DEFAULT)
    # Set beta and start the scan from tilde_eps = 2.0 where limit cycles are active
    p = @set par.β0 = β0_val
    p = @set p.tilde_eps = 2.0

    # 1. Burn-in to settle onto the attractor
    sol_burnin = DE.solve(
        DE.ODEProblem(sircmw!, u0, (0.0, 500.0), p),
        DE.AutoTsit5(DE.Rosenbrock23());
        abstol = 1e-12, reltol = 1e-12, maxiters = 10^7)

    # 2. Integrate a short lap to locate peaks
    sol_lap = DE.solve(
        DE.ODEProblem(sircmw!, sol_burnin(499.0), (0.0, 5.0), p),
        DE.AutoTsit5(DE.Rosenbrock23());
        abstol = 1e-12, reltol = 1e-12)

    # Find peaks to estimate period and starting point
    ts = LinRange(0.0, 5.0, 5000)
    vals = [sol_lap(t)[2] for t in ts]
    peaks = Int[]
    for i in 2:(length(vals)-1)
        if vals[i] > vals[i-1] && vals[i] > vals[i+1] && vals[i] > 1e-4
            push!(peaks, i)
        end
    end

    if length(peaks) < 2
        error("No oscillations detected at β₀ = $β0_val; cannot seed periodic orbit.")
    end

    period = ts[peaks[end]] - ts[peaks[end-1]]
    t_start = ts[peaks[end-1]]

    # 3. Resolve a clean 1-period lap starting at a peak
    sol_one_period = DE.solve(
        DE.ODEProblem(sircmw!, sol_lap(t_start), (0.0, period), p),
        DE.AutoTsit5(DE.Rosenbrock23());
        abstol = 1e-12, reltol = 1e-12)

    # 4. Set up periodic orbit problem using Trapezoid method
    prob_bif = BK.ODEBifProblem(sircmw!, sol_one_period(0.0), p, (@optic _.tilde_eps);
        record_from_solution = (x,p;k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))

    pc, ci = BK.generate_ci_problem(BK.Trapeze(M = M_GRID),
                                    prob_bif, sol_one_period, period)

    return BK.continuation(pc, ci, BK.PALC(), OPTS_SCAN;
                           verbosity = 0, normC = BK.norminf, ARGS_PO...)
end

# ─── SECTION 5: 2-PARAMETER CONTINUATION ──────────────────────────────────────
function _continue_both_dirs(br, ind, lens2, opts_base, kw)
    branches = Any[]
    for sgn in (+1.0, -1.0)
        opts = @set opts_base.ds = sgn * abs(opts_base.ds)
        try
            brc = BK.continuation(br, ind, lens2, opts; kw...)
            push!(branches, brc)
        catch e
            @debug "continuation direction stalled" ind=ind sgn=sgn exception=e
        end
    end
    return branches
end

function continue_pd_curve(br, ind, lens2, opts)
    _continue_both_dirs(br, ind, lens2, opts, (
        alg = BK.PALC(tangent = BK.Bordered()),
        verbosity = 0, normC = BK.norminf,
        jacobian_ma = BK.MinAug(),
        start_with_eigen = false,
        detect_codim2_bifurcation = 0,
        bothside = false))
end

function continue_fold_curve(br, ind, lens2, opts)
    _continue_both_dirs(br, ind, lens2, opts, (
        alg = BK.PALC(tangent = BK.Bordered()),
        verbosity = 0, normC = BK.norminf,
        jacobian_ma = BK.MinAug(),
        start_with_eigen = false,  
        detect_codim2_bifurcation = 0,
        bothside = false))
end

# ─── SECTION 6: SWEEP PIPELINE ────────────────────────────────────────────────
struct BifDiagram
    pd_curves      :: Vector{Any}
    fold_curves    :: Vector{Any}
    codim2_points  :: Vector{NamedTuple}
    scans          :: Vector{Any}
end

# Adaptive coverage seeder helper
function is_covered(tilde_eps_val, β_val, curves; εtol = 0.03, βtol = 50.0)
    for brc in curves
        for (e, b) in zip(brc.branch.tilde_eps, brc.branch.β0)
            if abs(e - tilde_eps_val) < εtol && abs(b - β_val) < βtol
                return true
            end
        end
    end
    return false
end

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
        
        pd_idx = [i for (i, sp) in enumerate(br.specialpoint) if sp.type == :pd]
        bp_idx = [i for (i, sp) in enumerate(br.specialpoint) if sp.type == :bp]
        println("  PDs: $(length(pd_idx))   Folds (BPs): $(length(bp_idx))")

        for i in pd_idx
            sp = br.specialpoint[i]
            # Filter for endemic limit cycles
            br.branch.amplitude[sp.step] > 1e-3 || continue
            
            if is_covered(sp.param, β0v, pd_curves; εtol = 0.03, βtol = 50.0)
                println("  · PD at tilde_eps = $(round(sp.param; digits=5)) is already covered; skipping.")
                continue
            end
            
            println("  · PD at tilde_eps = $(round(sp.param; digits=5)): continuing...")
            brcs = continue_pd_curve(br, i, (@optic _.β0), OPTS_PD)
            append!(pd_curves, brcs)
        end
        
        for i in bp_idx
            sp = br.specialpoint[i]
            # Filter for endemic limit cycles
            br.branch.amplitude[sp.step] > 1e-3 || continue
            
            if is_covered(sp.param, β0v, fold_curves; εtol = 0.03, βtol = 50.0)
                println("  · Fold at tilde_eps = $(round(sp.param; digits=5)) is already covered; skipping.")
                continue
            end
            
            println("  · Fold at tilde_eps = $(round(sp.param; digits=5)): continuing...")
            brcs = continue_fold_curve(br, i, (@optic _.β0), OPTS_FOLD)
            append!(fold_curves, brcs)
        end
    end

    println("\nSweep complete: $(length(pd_curves)) PD, $(length(fold_curves)) Folds")
    return BifDiagram(pd_curves, fold_curves, codim2_pts, scans)
end

# ─── SECTION 7: PLOTTING & EXPORT ─────────────────────────────────────────────
function branch_coords(brc)
    tilde_eps = collect(brc.branch.tilde_eps)
    β = collect(brc.branch.β0)
    mask = @. (tilde_eps ≥ tilde_eps_MIN) & (tilde_eps ≤ tilde_eps_MAX) & (β ≥ β_MIN) & (β ≤ β_MAX)
    return tilde_eps[mask], β[mask]
end

n_points_in_box(brc) = length(branch_coords(brc)[1])

function plot_f_curves(diag::BifDiagram; min_points = 20)
    plt = plot(xlabel = "tilde_eps (scaled erosion rate)",
               ylabel = "β₀ (baseline transmission)",
               title  = "f₁⁽¹⁾ — period-doubling of period-1 (SIRCmw)",
               xlims  = (tilde_eps_MIN, tilde_eps_MAX), ylims = (β_MIN, β_MAX),
               size   = (700, 600), legend = :topleft)
    labeled = false
    for brc in diag.pd_curves
        n_points_in_box(brc) ≥ min_points || continue
        ε, β = branch_coords(brc)
        scatter!(plt, ε, β; ms = 1.0, mc = :black, msw = 0.0, label = labeled ? false : "f₁⁽¹⁾")
        labeled = true
    end
    hline!(plt, [PAR_BASE.μ + PAR_BASE.α]; ls = :dot, lc = :gray, label = "R₀ = 1")
    try
        savefig(plt, "SIRCmw/f_curves.png")
    catch e
        @warn "Could not save f_curves.png" exception=e
    end
    return plt
end

function plot_t_curves(diag::BifDiagram; min_points = 20)
    plt = plot(xlabel = "tilde_eps (scaled erosion rate)",
               ylabel = "β₀ (baseline transmission)",
               title  = "t⁽¹⁾ — fold of period-1 (SIRCmw)",
               xlims  = (tilde_eps_MIN, tilde_eps_MAX), ylims = (β_MIN, β_MAX),
               size   = (700, 600), legend = :topleft)
    labeled = false
    for brc in diag.fold_curves
        n_points_in_box(brc) ≥ min_points || continue
        ε, β = branch_coords(brc)
        scatter!(plt, ε, β; ms = 1.0, mc = :black, msw = 0.0, label = labeled ? false : "t⁽¹⁾")
        labeled = true
    end
    hline!(plt, [PAR_BASE.μ + PAR_BASE.α]; ls = :dot, lc = :gray, label = "R₀ = 1")
    try
        savefig(plt, "SIRCmw/t_curves.png")
    catch e
        @warn "Could not save t_curves.png" exception=e
    end
    return plt
end

function export_to_csv(diag::BifDiagram)
    # Export Folds
    for (idx, brc) in enumerate(diag.fold_curves)
        ε, β = branch_coords(brc)
        if length(ε) > 0
            writedlm("SIRCmw/fold_curve_$idx.csv", hcat(ε, β), ',')
        end
    end
    # Export PDs
    for (idx, brc) in enumerate(diag.pd_curves)
        ε, β = branch_coords(brc)
        if length(ε) > 0
            writedlm("SIRCmw/pd_curve_$idx.csv", hcat(ε, β), ',')
        end
    end
    println("Saved CSV curves to SIRCmw/")
end

# ─── SECTION 8: EXECUTION ─────────────────────────────────────────────────────
println("Running SIRCmw automated bifurcation sweep...")
diag = run_bifurcation_sweep()

# save sweep database
serialize("SIRCmw/diag_clean.jls", diag)
println("Saved sweep to: SIRCmw/diag_clean.jls")

# Export CSVs first as a safety measure
export_to_csv(diag)

# plot results
plot_f_curves(diag)
plot_t_curves(diag)
println("All SIRCmw tasks finished successfully!")
