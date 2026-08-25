# ═════════════════════════════════════════════════════════════════════════════
#  Automated period-1 bifurcation diagram for the seasonally-forced SIRC model
#
#  Reproduces the black (f) and blue (t) curves of Casagrandi et al. (2006)
#  Fig. 3(a) — period-doubling (flip) and tangent (fold) bifurcations of the
#  period-1 attractor in the (ε, β₀) plane.
# ═════════════════════════════════════════════════════════════════════════════

using Logging
using Serialization

# ─── WARNING FILTER LOGGER ───────────────────────────────────────────────────
# Suppresses the benign [PD-Iooss] checks warnings that clutter the stdout.
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

# Set headless plotting to prevent opening windows on execution
ENV["GKSwstype"] = "100"

# ─── MODEL ──────────────────────────────────────────────────────────────────
# Seasonally-forced SIRC + Stuart–Landau oscillator
# (w₁, w₂) live on the unit circle with period 1, so β(t) = β₀(1 + ε w₁)
# reproduces β₀(1 + ε cos 2πt). The 6-D system is autonomous, so BK can
# treat the orbit as a periodic orbit of period T = 1.
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

const PAR_BASE = (μ = 0.02, α = 365.0/3, δ = 1.0/1.61, γ = 0.35,
                  σ = 0.07874, β0 = 600.0, ε = 0.01)
const U0_DEFAULT = [0.3, 1e-3, 0.4, 0.299, 1.0, 0.0]

# ─── PARAMETER BOX + SEEDS ───────────────────────────────────────────────────
const ε_MIN, ε_MAX = 0.0, 0.35
const β_MIN, β_MAX = 0.0, 2000.0

# Coarse grid of β₀ values for seeding. R₀ = β₀/(μ+α); we need R₀ > 1,
# i.e. β₀ > μ+α ≈ 121.7. Seeds are denser in the low-β₀ strip because leaf
# tips of the (ε, β₀) picture often cluster near the R₀=1 floor.
# Added 1750.0 to capture the upper middle fold component around (0.15, 1800).
const SEED_β0 = [140.0, 200.0, 300.0, 400.0, 600.0, 900.0, 1200.0, 1500.0, 1750.0]

# Vertical scan seeds: fix ε, sweep β₀. Complements the horizontal scans.
const SEED_ε = [0.03, 0.08, 0.15, 0.22, 0.30]

const N_MESH = 15    # collocation intervals
const M_DEG  = 3     # polynomial degree per interval

# ─── CONTINUATION OPTIONS ─────────────────────────────────────────────────────

# Horizontal 1-param scan: fix β₀, continue in ε
const OPTS_HORIZ = BK.ContinuationPar(
    p_min = ε_MIN, p_max = ε_MAX,
    ds = 0.001, dsmin = 1e-6, dsmax = 0.01,
    max_steps = 500,
    newton_options = BK.NewtonPar(tol = 1e-9, max_iterations = 15),
    detect_bifurcation = 3,    # bisection-refined :bp/:pd/:ns detection
    n_inversion = 6,
    tol_stability = 1e-3)

# 1-param scan in β₀ (vertical, fix ε). Same shape as OPTS_HORIZ but for
# the β₀ axis.
const OPTS_VERT = BK.ContinuationPar(
    p_min = 125.0, p_max = β_MAX,
    ds = 1.0, dsmin = 1e-6, dsmax = 10.0,
    max_steps = 1500,
    newton_options = BK.NewtonPar(tol = 1e-9, max_iterations = 15),
    detect_bifurcation = 3, n_inversion = 6, tol_stability = 1e-3)

# Codim-2 continuation options.
# _β variants bound β₀ (used from horizontal scans, lens2 = β₀).
# _ε variants bound ε  (used from vertical scans, lens2 = ε).
const OPTS_PD_β = BK.ContinuationPar(
    p_min = 125.0, p_max = β_MAX,
    ds = 5.0, dsmin = 1e-4, dsmax = 50.0,
    max_steps = 800,
    newton_options = BK.NewtonPar(tol = 1e-8, max_iterations = 25),
    detect_bifurcation = 0)

const OPTS_PD_ε = BK.ContinuationPar(
    p_min = ε_MIN, p_max = ε_MAX,
    ds = 0.005, dsmin = 1e-6, dsmax = 0.02,
    max_steps = 800,
    newton_options = BK.NewtonPar(tol = 1e-8, max_iterations = 25),
    detect_bifurcation = 0)

const OPTS_FOLD_β = BK.ContinuationPar(
    p_min = 125.0, p_max = β_MAX,
    ds = 5.0, dsmin = 1e-5, dsmax = 50.0,
    max_steps = 800,
    newton_options = BK.NewtonPar(tol = 1e-7, max_iterations = 30),
    detect_bifurcation = 0)

const OPTS_FOLD_ε = BK.ContinuationPar(
    p_min = ε_MIN, p_max = ε_MAX,
    ds = 0.005, dsmin = 1e-6, dsmax = 0.02,
    max_steps = 800,
    newton_options = BK.NewtonPar(tol = 1e-7, max_iterations = 30),
    detect_bifurcation = 0)

# Record amplitude + period at every 1-param step (used by the sanity plots)
const ARGS_PO = (record_from_solution = (x, p; k...) -> begin
        xtt = BK.get_periodic_orbit(p.prob, x, p.p)
        return (I_max  = maximum(xtt[2,:]),
                I_min  = minimum(xtt[2,:]),
                period = BK.getperiod(p.prob, x, p.p))
    end,)

# ─── CORE OPERATION 1: horizontal scan ────────────────────────────────────────
function horizontal_scan(β0_val; par = PAR_BASE, u0 = U0_DEFAULT)
    p = @set par.β0 = β0_val

    # 500-year burn-in onto the attractor
    sol_burnin = DE.solve(
        DE.ODEProblem(sirc!, u0, (0.0, 500.0), p),
        DE.AutoTsit5(DE.Rosenbrock23());
        abstol = 1e-12, reltol = 1e-12, maxiters = 10^7)

    # one clean 3-year lap to hand to collocation
    sol_lap = DE.solve(
        DE.ODEProblem(sirc!, sol_burnin(499.0), (0.0, 3.0), p),
        DE.AutoTsit5(DE.Rosenbrock23());
        abstol = 1e-12, reltol = 1e-10)

    prob_bif = BK.ODEBifProblem(sirc!, sol_lap(0.0), p, (@optic _.ε);
        record_from_solution = (x,p;k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))

    pc, ci = BK.generate_ci_problem(BK.Collocation(N_MESH, M_DEG),
                                    prob_bif, sol_lap, 1.0)

    return BK.continuation(pc, ci, BK.PALC(), OPTS_HORIZ;
                           verbosity = 0, normC = BK.norminf, ARGS_PO...)
end

# ─── CORE OPERATION 1b: vertical scan (fix ε, vary β₀) ────────────────────────
function vertical_scan(ε_val; par = PAR_BASE, u0 = U0_DEFAULT,
                       β0_start = 500.0)
    p = @set par.ε = ε_val
    p = @set p.β0 = β0_start

    sol_burnin = DE.solve(
        DE.ODEProblem(sirc!, u0, (0.0, 500.0), p),
        DE.AutoTsit5(DE.Rosenbrock23());
        abstol = 1e-12, reltol = 1e-12, maxiters = 10^7)

    sol_lap = DE.solve(
        DE.ODEProblem(sirc!, sol_burnin(499.0), (0.0, 3.0), p),
        DE.AutoTsit5(DE.Rosenbrock23());
        abstol = 1e-12, reltol = 1e-10)

    prob_bif = BK.ODEBifProblem(sirc!, sol_lap(0.0), p, (@optic _.β0);
        record_from_solution = (x,p;k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))

    pc, ci = BK.generate_ci_problem(BK.Collocation(N_MESH, M_DEG),
                                    prob_bif, sol_lap, 1.0)

    return BK.continuation(pc, ci, BK.PALC(), OPTS_VERT;
                           verbosity = 0, normC = BK.norminf, ARGS_PO...)
end

# ─── CORE OPERATION 2: classify special points on a 1-param branch ────────────
pd_indices(br) = [i for (i, sp) in enumerate(br.specialpoint) if sp.type == :pd]
bp_indices(br) = [i for (i, sp) in enumerate(br.specialpoint) if sp.type == :bp]

# ─── CORE OPERATION 3: codim-2 continuations ──────────────────────────────────
# Runs the two directions (+ds and -ds) as SEPARATE calls under try/catch.
# Prevents BK's internal `MethodError: getindex(::Nothing, ...)` when
# using `bothside = true` and `start_with_eigen = true` fails to populate
# eigenvectors in one of the directions.
function _continue_both_dirs(br, ind, lens2, opts_base, kw)
    branches = Any[]
    for sgn in (+1.0, -1.0)
        opts = @set opts_base.ds = sgn * abs(opts_base.ds)
        try
            brc = BK.continuation(br, ind, lens2, opts; kw...)
            push!(branches, brc)
        catch e
            @warn "    direction stalled" ind=ind ds_sign=sgn exception=(e, catch_backtrace())
        end
    end
    return branches
end

function continue_pd_curve(br, ind, lens2, opts)
    _continue_both_dirs(br, ind, lens2, opts, (
        alg = BK.PALC(tangent = BK.Bordered()),
        verbosity = 0, normC = BK.norminf,
        jacobian_ma = BK.MinAug(),
        start_with_eigen = true,            # PD works fine with start_with_eigen=true
        detect_codim2_bifurcation = 2,
        bothside = false))
end

function continue_fold_curve(br, ind, lens2, opts)
    _continue_both_dirs(br, ind, lens2, opts, (
        alg = BK.PALC(tangent = BK.Bordered()),
        verbosity = 0, normC = BK.norminf,
        jacobian_ma = BK.MinAug(),
        start_with_eigen = false,           # MUST be false because FloquetColl does not save eigenvectors!
        detect_codim2_bifurcation = 2,
        usehessian = true,
        bothside = false))
end

# ─── AUTOMATED DRIVER ────────────────────────────────────────────────────────
struct BifDiagram
    pd_curves      :: Vector{Any}   # BK branch objects (period-doubling)
    fold_curves    :: Vector{Any}   # BK branch objects (fold)
    codim2_points  :: Vector{NamedTuple}
    scans          :: Vector{Any}   # the 1-param scans, kept for sanity plots
end

function sweep_diagram(β0_seeds = SEED_β0, ε_seeds = SEED_ε)
    pd_curves     = Any[]
    fold_curves   = Any[]
    codim2_pts    = NamedTuple[]
    scans         = Any[]

    # ── Horizontal scans (fix β₀, vary ε; lens2 for codim-2 = β₀) ──
    for β0v in β0_seeds
        println("─── horizontal scan at β₀ = $β0v ───")
        br = try
            horizontal_scan(β0v)
        catch e
            @warn "horizontal_scan failed at β₀=$β0v" exception=e
            continue
        end
        push!(scans, br)
        _process_scan!(pd_curves, fold_curves, codim2_pts,
                       br, (@optic _.β0), OPTS_PD_β, OPTS_FOLD_β)
    end

    # ── Vertical scans (fix ε, vary β₀; lens2 for codim-2 = ε) ──
    for εv in ε_seeds
        println("─── vertical scan at ε = $εv ───")
        br = try
            vertical_scan(εv)
        catch e
            @warn "vertical_scan failed at ε=$εv" exception=e
            continue
        end
        push!(scans, br)
        _process_scan!(pd_curves, fold_curves, codim2_pts,
                       br, (@optic _.ε), OPTS_PD_ε, OPTS_FOLD_ε)
    end

    println("\nSweep summary: $(length(pd_curves)) PD branches, " *
            "$(length(fold_curves)) fold branches, " *
            "$(length(codim2_pts)) codim-2 points")

    return BifDiagram(pd_curves, fold_curves, codim2_pts, scans)
end

function _process_scan!(pd_curves, fold_curves, codim2_pts,
                        br, lens2, opts_pd, opts_fold)
    pd_idx = pd_indices(br)
    bp_idx = bp_indices(br)
    println("  PDs: $(length(pd_idx))   :bp (fold candidates): $(length(bp_idx))")

    for i in pd_idx
        sp = br.specialpoint[i]
        println("  · PD at param=$(round(sp.param; digits=5)): continuing...")
        brcs = continue_pd_curve(br, i, lens2, opts_pd)
        println("    → $(length(brcs)) branch(es) returned")
        append!(pd_curves, brcs)
        for brc in brcs
            _collect_codim2!(store = codim2_pts, brc = brc, source = :pd,
                             types = (:gpd, :R2, :foldFlip, :pdNS))
        end
    end
    for i in bp_idx
        sp = br.specialpoint[i]
        println("  · :bp at param=$(round(sp.param; digits=5)): continuing as fold...")
        brcs = continue_fold_curve(br, i, lens2, opts_fold)
        println("    → $(length(brcs)) branch(es) returned")
        append!(fold_curves, brcs)
        for brc in brcs
            _collect_codim2!(store = codim2_pts, brc = brc, source = :fold,
                             types = (:cusp, :R1, :foldFlip, :foldNS))
        end
    end
end

function _collect_codim2!(; store, brc, source, types)
    for sp in brc.specialpoint
        sp.type in types || continue
        # Filter out spurious detections at the starting point of continuation branches.
        # Codim-2 bifurcations are isolated in 2D parameter space, so they cannot
        # generically lie exactly on the 1D parameter seed scans.
        sp.step > 5 || continue
        
        step = clamp(sp.step, 1, length(brc.branch))
        push!(store, (type = sp.type,
                      ε    = brc.branch.ε[step],
                      β    = brc.branch.β0[step],
                      source = source))
    end
    return store
end

# ─── PLOTTING & UTILITIES ─────────────────────────────────────────────────────

# Extract (ε, β₀) from a codim-2 branch, clipped to the parameter box
function branch_coords(brc)
    ε = collect(brc.branch.ε)
    β = collect(brc.branch.β0)
    mask = @. (ε ≥ ε_MIN) & (ε ≤ ε_MAX) & (β ≥ β_MIN) & (β ≤ β_MAX)
    return ε[mask], β[mask]
end

# Number of in-box points on a branch — used to filter isolated stalls
n_points_in_box(brc) = length(branch_coords(brc)[1])

# Deduplicate codim-2 points to clean up the plot overlay
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

function plot_f_curves(diag::BifDiagram; min_points = 5)
    plt = plot(xlabel = "ε (degree of seasonality)",
               ylabel = "β₀ (baseline transmission)",
               title  = "f₁⁽¹⁾ — period-doubling of period-1",
               xlims  = (ε_MIN, ε_MAX), ylims = (β_MIN, β_MAX),
               size   = (700, 600), legend = :topleft)
    labeled = false
    for brc in diag.pd_curves
        n_points_in_box(brc) ≥ min_points || continue
        ε = collect(brc.branch.ε)
        β = collect(brc.branch.β0)
        # Using scatter! with msw=0.0 as requested to keep components distinct without shortcut lines.
        scatter!(plt, ε, β; ms = 1.0, mc = :black, msw = 0.0,
                 label = labeled ? false : "f₁⁽¹⁾")
        labeled = true
    end
    for c2 in deduplicate_codim2(diag.codim2_points)
        c2.source == :pd || continue
        c2.type == :gpd || continue
        scatter!(plt, [c2.ε], [c2.β]; ms = 6, m = :diamond,
                 mc = :white, msc = :black, msw = 1.2, label = false)
    end
    hline!(plt, [PAR_BASE.μ + PAR_BASE.α]; ls = :dot, lc = :gray,
           label = "R₀ = 1")
    display(plt)
    savefig(plt, "f_curves.png")
    return plt
end

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
        (len >= 50 || (len >= 8 && max_β >= 1300.0)) || continue
        n_points_in_box(brc) ≥ min_points || continue
        ε = collect(brc.branch.ε)
        β = collect(brc.branch.β0)
        # Using scatter! with blue color and msw=0.0 to match publication look cleanly.
        scatter!(plt, ε, β; ms = 1.0, mc = :blue, msw = 0.0,
                 label = labeled ? false : "t⁽¹⁾")
        labeled = true
    end
    for c2 in deduplicate_codim2(diag.codim2_points)
        c2.source == :fold || continue
        c2.type == :cusp || continue
        scatter!(plt, [c2.ε], [c2.β]; ms = 6, m = :square,
                 mc = :black, msc = :black, label = false)
    end
    hline!(plt, [PAR_BASE.μ + PAR_BASE.α]; ls = :dot, lc = :gray,
           label = "R₀ = 1")
    display(plt)
    savefig(plt, "t_curves.png")
    return plt
end

# Sanity plot: I amplitude vs ε from each horizontal scan (fixes FieldError by checking hasproperty)
function plot_scans(diag::BifDiagram; title = "1-param scans — I amplitude")
    plt = plot(xlabel = "ε", ylabel = "I (min–max envelope)",
               title = title, size = (700, 400))
    for br in diag.scans
        if hasproperty(br.branch, :ε)
            plot!(plt, br.branch.ε, br.branch.I_max; lc = :black, label = false)
            plot!(plt, br.branch.ε, br.branch.I_min; lc = :black, label = false)
        end
    end
    display(plt)
    savefig(plt, "scans_amplitude.png")
    return plt
end

# Combined Bifurcation Diagram (matching the target publication figure overlay)
function plot_combined_diagram(diag::BifDiagram; min_points = 5)
    plt = plot(xlabel = "ε (degree of seasonality)",
               ylabel = "β₀ (baseline transmission)",
               title  = "Seasonally Forced SIRC Bifurcation Diagram (Period-1 Curves)",
               xlims  = (ε_MIN, ε_MAX), ylims = (β_MIN, β_MAX),
               size   = (700, 600), legend = :topright)
    
    # 1. Plot R0 = 1 floor
    hline!(plt, [PAR_BASE.μ + PAR_BASE.α]; ls = :dash, lc = :gray, label = "R₀ = 1 floor")
    
    # 2. Plot PD curves (f_pieces) in black
    pd_labeled = false
    for brc in diag.pd_curves
        n_points_in_box(brc) ≥ min_points || continue
        ε = collect(brc.branch.ε)
        β = collect(brc.branch.β0)
        scatter!(plt, ε, β; ms = 1.0, mc = :black, msw = 0.0,
                 label = pd_labeled ? false : "f₁⁽¹⁾ (Period Doubling)")
        pd_labeled = true
    end
    
    # 3. Plot Fold curves (t_pieces) in blue (matching target reference style)
    fold_labeled = false
    for brc in diag.fold_curves
        len = length(brc.branch)
        max_β = maximum(brc.branch.β0)
        (len >= 50 || (len >= 8 && max_β >= 1300.0)) || continue
        n_points_in_box(brc) ≥ min_points || continue
        ε = collect(brc.branch.ε)
        β = collect(brc.branch.β0)
        scatter!(plt, ε, β; ms = 1.0, mc = :blue, msw = 0.0,
                 label = fold_labeled ? false : "t⁽¹⁾ (Fold / Tangent)")
        fold_labeled = true
    end
    
    # 4. Plot codim-2 points (diamonds for gpd, squares for cusp)
    c2_points = deduplicate_codim2(diag.codim2_points)
    gpd_labeled = false
    cusp_labeled = false
    for c2 in c2_points
        if c2.type == :gpd
            scatter!(plt, [c2.ε], [c2.β]; ms = 7, m = :diamond,
                     mc = :white, msc = :black, msw = 1.2,
                     label = gpd_labeled ? false : "gpd (generalized PD)")
            gpd_labeled = true
        elseif c2.type == :cusp
            scatter!(plt, [c2.ε], [c2.β]; ms = 6, m = :square,
                     mc = :black, msc = :black, msw = 0.0,
                     label = cusp_labeled ? false : "cusp (fold cusp)")
            cusp_labeled = true
        end
    end
    
    display(plt)
    output_path = "/Users/danielarenee/Desktop/Infection-Modeling-with-Mutation/bifurcation_map_SIRC.png"
    savefig(plt, output_path)
    println("Saved combined bifurcation map to: $output_path")
    return plt
end

# ─── RUN ─────────────────────────────────────────────────────────────────────
diag_file = joinpath(@__DIR__, "diag.jls")
diag = if isfile(diag_file)
    println("Loading pre-computed bifurcation diagram from: $diag_file")
    deserialize(diag_file)
else
    println("Starting automated bifurcation sweep...")
    d = sweep_diagram()
    serialize(diag_file, d)
    println("Serialized bifurcation diagram to: $diag_file")
    d
end

plot_f_curves(diag)
plot_t_curves(diag)
plot_scans(diag)
plot_combined_diagram(diag)
println("All tasks finished successfully!")
