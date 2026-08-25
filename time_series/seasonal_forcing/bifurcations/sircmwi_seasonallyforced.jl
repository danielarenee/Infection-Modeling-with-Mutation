# Bifurcation diagram for SIRCmw_I (seasonally forced, ε₁ = ε₂ = ε)
# Replication of the Casagrandi figure-3 methodology for the I-feedback model.
#
# Traces:
#   f-curves  — period-doubling bifurcations of the period-1 orbit
#   t-curves  — fold (tangent) bifurcations of the period-1 orbit
#
# in the (η, β₀) plane, where η is the seasonal forcing amplitude.
# The I-feedback strength ε̃ is a fixed parameter set in the CONFIG block.

using Logging
using Serialization

# suppress benign BifurcationKit solver warnings
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

# ============================================================================
#                           *** CONFIG ***
# ============================================================================

# I-feedback strength (in terms of ε̃; set to 0 to recover standard SIRC)
const TILDE_EPS = 0.3       # ε̃
const I_0       = 0.001     # reference prevalence scale
# Physical feedback coefficient (derived, do not edit):
const EPS_PHYS  = TILDE_EPS / I_0

# Casagrandi biological baseline parameters
const PAR_BASE = (
    μ   = 0.02,
    α   = 365.0 / 3.0,
    δ   = 1.0 / 1.61,
    γ   = 0.35,
    σ   = 0.07874,
    β0  = 600.0,
    η   = 0.01,         # seasonal forcing amplitude (this is the swept parameter)
    eps = EPS_PHYS      # I-feedback coefficient (fixed for this diagram)
)

# Default initial condition (S, I, R, C, w1, w2)
# w1, w2 are the autonomous oscillator variables initialised on the limit cycle
const U0_DEFAULT = [0.3, 1e-3, 0.4, 0.299, 1.0, 0.0]

# --- Axis bounds for the (η, β₀) diagram ------------------------------------
const η_MIN, η_MAX = 0.0, 0.35
const β_MIN, β_MAX = 0.0, 2000.0

# --- β₀ seeds for horizontal scans (diagnostic: 10 seeds) ------------------
const SEED_β0 = [
    150.0, 300.0, 500.0, 700.0, 900.0,
    1100.0, 1300.0, 1500.0, 1700.0, 1950.0
]

# --- Collocation mesh --------------------------------------------------------
const N_MESH = 15
const M_DEG  = 3

# ============================================================================


# SIRCmw_I ODE WITH AUTONOMOUS OSCILLATOR
# The oscillator (w1, w2) produces w1 → cos(2πt) on its limit cycle, giving
# the seasonal forcing β(t) = β₀(1 + η·cos(2πt)) without explicit time.
# eps1 = eps2 = eps  (symmetric I-feedback)
function sircmwi!(du, u, p, t = 0)
    S, I, R, C, w1, w2 = u
    β   = p.β0 * (1.0 + p.η * w1)
    eps = p.eps

    du[1] = p.μ*(1 - S) - β*S*I + (1 + eps*I)*p.γ*C
    du[2] = β*S*I + p.σ*β*C*I - (p.μ + p.α)*I
    du[3] = (1 - p.σ)*β*C*I + p.α*I - p.μ*R - (1 + eps*I)*p.δ*R
    du[4] = (1 + eps*I)*p.δ*R - β*C*I - p.μ*C - (1 + eps*I)*p.γ*C
    # autonomous oscillator
    du[5] = w1 - 2π*w2 - (w1^2 + w2^2)*w1
    du[6] = 2π*w1 + w2 - (w1^2 + w2^2)*w2
    du
end


# SOLVER OPTIONS

# 1-parameter scan (vary η at fixed β₀)
const OPTS_SCAN = BK.ContinuationPar(
    p_min = η_MIN, p_max = η_MAX,
    ds = 0.001, dsmin = 1e-6, dsmax = 0.01,
    max_steps = 500,
    newton_options = BK.NewtonPar(tol = 1e-9, max_iterations = 20),
    detect_bifurcation = 3,
    n_inversion = 6,
    tol_stability = 1e-3)

# 2-parameter continuation: period-doubling curves (vary β₀)
const OPTS_PD = BK.ContinuationPar(
    p_min = 125.0, p_max = β_MAX,
    ds = 5.0, dsmin = 1e-4, dsmax = 50.0,
    max_steps = 800,
    newton_options = BK.NewtonPar(tol = 1e-8, max_iterations = 25),
    detect_bifurcation = 0)

# 2-parameter continuation: fold curves (vary β₀)
const OPTS_FOLD = BK.ContinuationPar(
    p_min = 125.0, p_max = β_MAX,
    ds = 5.0, dsmin = 1e-6, dsmax = 50.0,
    max_steps = 800,
    newton_options = BK.NewtonPar(tol = 1e-7, max_iterations = 30),
    detect_bifurcation = 0)


# 1-PARAMETER CONTINUATION (horizontal scan at fixed β₀)
# 1. Long burn-in ODE (500 yr) to settle on attractor
# 1. Long burn-in ODE (100 yr) to settle on attractor
# 2. Clean 3-yr lap to seed collocation
# 3. Continue periodic orbit as η varies
function horizontal_scan(β0_val; par = PAR_BASE, u0 = U0_DEFAULT)
    p = @set par.β0 = β0_val

    sol_burnin = DE.solve(
        DE.ODEProblem(sircmwi!, u0, (0.0, 100.0), p),
        DE.AutoTsit5(DE.Rosenbrock23());
        abstol = 1e-12, reltol = 1e-12, maxiters = 10^7)

    sol_lap = DE.solve(
        DE.ODEProblem(sircmwi!, sol_burnin(99.0), (0.0, 3.0), p),
        DE.AutoTsit5(DE.Rosenbrock23());
        abstol = 1e-12, reltol = 1e-10)

    prob_bif = BK.ODEBifProblem(sircmwi!, sol_lap(0.0), p, (@optic _.η);
        record_from_solution = (x, p; k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))

    pc, ci = BK.generate_ci_problem(BK.Collocation(N_MESH, M_DEG),
                                    prob_bif, sol_lap, 1.0)

    return BK.continuation(pc, ci, BK.PALC(), OPTS_SCAN;
                           verbosity = 0, normC = BK.norminf)
end


# 2-PARAMETER CONTINUATION HELPERS

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

function continue_pd_curve(br, ind, lens2, opts)
    _continue_both_dirs(br, ind, lens2, opts, (
        alg = BK.PALC(tangent = BK.Bordered()),
        verbosity = 0, normC = BK.norminf,
        jacobian_ma = BK.MinAug(),
        start_with_eigen = true,
        detect_codim2_bifurcation = 2,
        bothside = false))
end

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


# SWEEP DATA STRUCT AND PIPELINE

struct BifDiagram
    pd_curves     :: Vector{Any}
    fold_curves   :: Vector{Any}
    codim2_points :: Vector{NamedTuple}
    scans         :: Vector{Any}
end

function _collect_codim2!(; store, brc, source, types)
    for sp in brc.specialpoint
        sp.type in types || continue
        sp.step > 5 || continue
        step = clamp(sp.step, 1, length(brc.branch))
        push!(store, (type   = sp.type,
                      η      = brc.branch.η[step],
                      β      = brc.branch.β0[step],
                      source = source))
    end
end

function run_bifurcation_sweep(β0_seeds = SEED_β0)
    pd_curves   = Any[]
    fold_curves = Any[]
    codim2_pts  = NamedTuple[]
    scans       = Any[]

    for β0v in β0_seeds
        println("─── Horizontal scan at β₀ = $β0v ───")
        br = try
            horizontal_scan(β0v)
        catch e
            @warn "Horizontal scan failed at β₀ = $β0v" exception = e
            continue
        end
        push!(scans, br)

        pd_idx = [i for (i, sp) in enumerate(br.specialpoint) if sp.type == :pd]
        bp_idx = [i for (i, sp) in enumerate(br.specialpoint) if sp.type == :bp]
        println("  PDs: $(length(pd_idx))   Folds (BPs): $(length(bp_idx))")

        for i in pd_idx
            sp = br.specialpoint[i]
            br.branch.amplitude[sp.step] > 1e-3 || continue
            println("  · PD at η = $(round(sp.param; digits=5)): continuing...")
            brcs = continue_pd_curve(br, i, (@optic _.β0), OPTS_PD)
            append!(pd_curves, brcs)
            for brc in brcs
                _collect_codim2!(store = codim2_pts, brc = brc, source = :pd, types = (:gpd,))
            end
        end

        for i in bp_idx
            sp = br.specialpoint[i]
            br.branch.amplitude[sp.step] > 1e-3 || continue
            println("  · Fold at η = $(round(sp.param; digits=5)): continuing...")
            brcs = continue_fold_curve(br, i, (@optic _.β0), OPTS_FOLD)
            append!(fold_curves, brcs)
            for brc in brcs
                _collect_codim2!(store = codim2_pts, brc = brc, source = :fold, types = (:cusp, :R1))
            end
        end
    end

    println("\nSweep complete: $(length(pd_curves)) PD, $(length(fold_curves)) Folds, $(length(codim2_pts)) Codim-2")
    return BifDiagram(pd_curves, fold_curves, codim2_pts, scans)
end


# PLOTTING

function branch_coords(brc)
    η = collect(brc.branch.η)
    β = collect(brc.branch.β0)
    mask = @. (η >= η_MIN) & (η <= η_MAX) & (β >= β_MIN) & (β <= β_MAX)
    return η[mask], β[mask]
end

n_points_in_box(brc) = length(branch_coords(brc)[1])

function plot_f_curves(diag::BifDiagram; min_points = 5)
    plt = plot(xlabel = "η  (seasonal forcing amplitude)",
               ylabel = "β₀  (baseline transmission rate)",
               title  = "f₁⁽¹⁾ — period-doubling of period-1  (ε̃ = $(TILDE_EPS))",
               xlims  = (η_MIN, η_MAX),
               ylims  = (β_MIN, β_MAX),
               size   = (700, 600),
               legend = :topleft)
    labeled = false
    for brc in diag.pd_curves
        n_points_in_box(brc) >= min_points || continue
        η, β = branch_coords(brc)
        scatter!(plt, η, β; ms = 1.0, mc = :black, msw = 0.0,
                 label = labeled ? false : "f₁⁽¹¹⁾")
        labeled = true
    end
    hline!(plt, [PAR_BASE.μ + PAR_BASE.α]; ls = :dot, lc = :gray, label = "R₀ = 1")
    out = joinpath(@__DIR__, "f_curves_eps$(TILDE_EPS)")
    savefig(plt, out * ".png")
    savefig(plt, out * ".pdf")
    println("Saved f-curves → $out.png / .pdf")
    return plt
end

function plot_t_curves(diag::BifDiagram; min_points = 5)
    plt = plot(xlabel = "η  (seasonal forcing amplitude)",
               ylabel = "β₀  (baseline transmission rate)",
               title  = "t⁽¹⁾ — fold of period-1  (ε̃ = $(TILDE_EPS))",
               xlims  = (η_MIN, η_MAX),
               ylims  = (β_MIN, β_MAX),
               size   = (700, 600),
               legend = :topleft)
    labeled = false
    for brc in diag.fold_curves
        len   = length(brc.branch)
        max_β = maximum(brc.branch.β0)
        (len >= 40 || (len >= 15 && max_β >= 1300.0)) || continue
        n_points_in_box(brc) >= min_points || continue
        η, β = branch_coords(brc)
        scatter!(plt, η, β; ms = 1.0, mc = :black, msw = 0.0,
                 label = labeled ? false : "t⁽¹⁾")
        labeled = true
    end
    hline!(plt, [PAR_BASE.μ + PAR_BASE.α]; ls = :dot, lc = :gray, label = "R₀ = 1")
    out = joinpath(@__DIR__, "t_curves_eps$(TILDE_EPS)")
    savefig(plt, out * ".png")
    savefig(plt, out * ".pdf")
    println("Saved t-curves → $out.png / .pdf")
    return plt
end


# EXECUTION
println("Running bifurcation sweep for SIRCmw_I  (ε̃ = $(TILDE_EPS), eps = $(EPS_PHYS))...")
diag = run_bifurcation_sweep()

diag_file = joinpath(@__DIR__, "diag_sircmwi_eps$(TILDE_EPS).jls")
serialize(diag_file, diag)
println("Saved sweep data → $diag_file")

plot_f_curves(diag)
plot_t_curves(diag)
println("All tasks finished successfully!")
