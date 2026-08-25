using Plots
import BifurcationKit as BK
import BifurcationKit: @optic, @set
import OrdinaryDiffEq as DE

# ── MODEL ──────────────────────────────────────────────────────────────────────
function sirc!(du, u, p, t = 0)
    S, I, R, C, w1, w2 = u
    β = p.β0 * (1.0 + p.ε * w1)
    du[1] = p.μ*(1-S) - β*S*I + p.γ*C
    du[2] = β*S*I + p.σ*β*C*I - (p.μ + p.α)*I
    du[3] = (1-p.σ)*β*C*I + p.α*I - (p.μ + p.δ)*R
    du[4] = p.δ*R - β*C*I - (p.μ + p.γ)*C
    du[5] = w1 - 2π*w2 - (w1^2 + w2^2)*w1
    du[6] = 2π*w1 + w2 - (w1^2 + w2^2)*w2
    du
end
par = (μ=0.02, α=365.0/3, δ=1.0/1.61, γ=0.35, σ=0.07874, β0=600.0, ε=0.01)
u0  = [0.3, 1e-3, 0.4, 0.299, 1.0, 0.0]
N_mesh = 15

# ── SHARED SETTINGS ────────────────────────────────────────────────────────────
# 1-param options for horizontal scans (primary = ε)
opts_h = BK.ContinuationPar(p_min=0.0, p_max=0.35,
    ds=0.001, dsmin=1e-6, dsmax=0.01, max_steps=500,
    newton_options=BK.NewtonPar(tol=1e-9, max_iterations=15),
    detect_bifurcation=3, n_inversion=6, tol_stability=1e-3)

# 1-param options for vertical scans (primary = β₀)
opts_v = BK.ContinuationPar(p_min=125.0, p_max=2000.0,
    ds=1.0, dsmin=1e-6, dsmax=10.0, max_steps=1500,
    newton_options=BK.NewtonPar(tol=1e-9, max_iterations=15),
    detect_bifurcation=3, n_inversion=6, tol_stability=1e-3)

argspo = (record_from_solution = (x, p; k...) -> begin
        xtt = BK.get_periodic_orbit(p.prob, x, p.p)
        isnothing(xtt) && return (I_max=NaN, I_min=NaN, period=NaN)
        return (I_max   = maximum(xtt[2,:]),
                I_min   = minimum(xtt[2,:]),
                period  = BK.getperiod(p.prob, x, p.p))
    end,)

clip(ε, β) = let m = (ε .>= 0.0) .& (ε .<= 0.35); (ε[m], β[m]); end

# ── SCAN FUNCTIONS ─────────────────────────────────────────────────────────────
# Fix β₀, sweep ε (horizontal scan). Detects :pd and :bp along the branch.
function scan_at_β0(β0_val)
    par_loc  = @set par.β0 = β0_val
    prob_de  = DE.ODEProblem(sirc!, u0, (0.0, 500.0), par_loc)
    sol      = DE.solve(prob_de, DE.AutoTsit5(DE.Rosenbrock23()),
                   abstol=1e-12, reltol=1e-12, maxiters=10_000_000)
    prob_de2 = DE.ODEProblem(sirc!, sol(499.0), (0.0, 3.0), par_loc,
                   reltol=1e-10, abstol=1e-12)
    sol2     = DE.solve(prob_de2, DE.AutoTsit5(DE.Rosenbrock23()))
    prob_bif = BK.ODEBifProblem(sirc!, sol2(0.0), par_loc, (@optic _.ε);
        record_from_solution=(x,p;k...)->(S=x[1],I=x[2],R=x[3],C=x[4]))
    pc, ci   = BK.generate_ci_problem(BK.Collocation(N_mesh,3), prob_bif, sol2, 1.0)
    return BK.continuation(pc, ci, BK.PALC(), opts_h; verbosity=0, normC=BK.norminf, argspo...)
end

# Fix ε, sweep β₀ (vertical scan).
# T_spin: integration time to spin onto the attractor; increase for high ε.
function scan_at_ε(ε_val, β0_start; T_spin=500.0)
    par_loc  = @set par.ε = ε_val
    par_loc  = @set par_loc.β0 = β0_start
    prob_de  = DE.ODEProblem(sirc!, u0, (0.0, T_spin), par_loc)
    sol      = DE.solve(prob_de, DE.AutoTsit5(DE.Rosenbrock23()),
                   abstol=1e-12, reltol=1e-12, maxiters=50_000_000)
    prob_de2 = DE.ODEProblem(sirc!, sol(T_spin-1.0), (0.0, 3.0), par_loc,
                   reltol=1e-10, abstol=1e-12)
    sol2     = DE.solve(prob_de2, DE.AutoTsit5(DE.Rosenbrock23()))
    prob_bif = BK.ODEBifProblem(sirc!, sol2(0.0), par_loc, (@optic _.β0);
        record_from_solution=(x,p;k...)->(S=x[1],I=x[2],R=x[3],C=x[4]))
    pc, ci   = BK.generate_ci_problem(BK.Collocation(N_mesh,3), prob_bif, sol2, 1.0)
    return BK.continuation(pc, ci, BK.PALC(), opts_v; verbosity=0, normC=BK.norminf, argspo...)
end

# ── 2-PARAM CONTINUATION ───────────────────────────────────────────────────────
# Period-doubling curve; seeded from horizontal scan (lens1=ε), lens2=β₀
function cont2_pd(br, idx; ds=5.0, dsmin=1e-4, max_steps=600)
    opts = BK.ContinuationPar(p_min=125.0, p_max=2000.0,
        ds=ds, dsmin=dsmin, dsmax=50.0, max_steps=max_steps,
        newton_options=BK.NewtonPar(tol=1e-8, max_iterations=25),
        detect_bifurcation=0)
    return BK.continuation(br, idx, (@optic _.β0), opts;
        alg=BK.PALC(tangent=BK.Bordered()), verbosity=1, normC=BK.norminf,
        jacobian_ma=BK.MinAug(), start_with_eigen=true,
        detect_codim2_bifurcation=0, bothside=false)
end

# Fold curve; seeded from horizontal scan (lens1=ε), lens2=β₀
function cont2_fold_h(br, idx; ds=5.0, dsmin=1e-5, max_steps=600)
    opts = BK.ContinuationPar(p_min=125.0, p_max=2000.0,
        ds=ds, dsmin=dsmin, dsmax=50.0, max_steps=max_steps,
        newton_options=BK.NewtonPar(tol=1e-7, max_iterations=30),
        detect_bifurcation=0)
    return BK.continuation(br, idx, (@optic _.β0), opts;
        alg=BK.PALC(tangent=BK.Bordered()), verbosity=1, normC=BK.norminf,
        jacobian_ma=BK.MinAug(), start_with_eigen=true,
        detect_codim2_bifurcation=2, usehessian=true, bothside=false)
end

# Fold curve; seeded from vertical scan (lens1=β₀), lens2=ε
function cont2_fold_v(br, idx; ds=0.01, dsmin=1e-6, max_steps=800)
    opts = BK.ContinuationPar(p_min=0.0, p_max=0.35,
        ds=ds, dsmin=dsmin, dsmax=0.1, max_steps=max_steps,
        newton_options=BK.NewtonPar(tol=1e-7, max_iterations=30),
        detect_bifurcation=0)
    return BK.continuation(br, idx, (@optic _.ε), opts;
        alg=BK.PALC(tangent=BK.Bordered()), verbosity=1, normC=BK.norminf,
        jacobian_ma=BK.MinAug(), start_with_eigen=true,
        detect_codim2_bifurcation=2, usehessian=true, bothside=true)
end

# ══════════════════════════════════════════════════════════════════════════════
# f CURVES  (period-doubling)
# ══════════════════════════════════════════════════════════════════════════════
br_h1200 = scan_at_β0(1200.0)
br_h400  = scan_at_β0(400.0)
br_h140  = scan_at_β0(140.0)

# f₁⁽¹⁾ upper piece — seeded from β₀=1200, pd[1] and pd[2]
br_fA1_up = cont2_pd(br_h1200, 1; ds= 5.0)
br_fA1_dn = cont2_pd(br_h1200, 1; ds=-5.0)
br_fA2_up = cont2_pd(br_h1200, 2; ds= 5.0)
br_fA2_dn = cont2_pd(br_h1200, 2; ds=-5.0)
ε_fA1 = vcat(br_fA1_dn.branch.ε, br_fA1_up.branch.ε)
β_fA1 = vcat(br_fA1_dn.branch.β0, br_fA1_up.branch.β0)
ε_fA2 = vcat(br_fA2_dn.branch.ε, br_fA2_up.branch.ε)
β_fA2 = vcat(br_fA2_dn.branch.β0, br_fA2_up.branch.β0)

# f₁⁽¹⁾ lower-mid piece — seeded from β₀=400
pd_400  = findfirst(sp -> sp.type == :pd, br_h400.specialpoint)
br_fB_dn = cont2_pd(br_h400, pd_400; ds=-2.0, dsmin=1e-5, max_steps=800)
br_fB_up = cont2_pd(br_h400, pd_400; ds= 2.0, dsmin=1e-5, max_steps=800)
ε_fB = vcat(br_fB_dn.branch.ε, br_fB_up.branch.ε)
β_fB = vcat(br_fB_dn.branch.β0, br_fB_up.branch.β0)

# f₁⁽¹⁾ bottom piece — seeded from β₀=140
pd_140  = findfirst(sp -> sp.type == :pd, br_h140.specialpoint)
br_fC_dn = cont2_pd(br_h140, pd_140; ds=-2.0, dsmin=1e-5, max_steps=800)
br_fC_up = cont2_pd(br_h140, pd_140; ds= 2.0, dsmin=1e-5, max_steps=800)
ε_fC = vcat(br_fC_dn.branch.ε, br_fC_up.branch.ε)
β_fC = vcat(br_fC_dn.branch.β0, br_fC_up.branch.β0)

plt_f = plot(xlabel="ε", ylabel="β₀", title="f curves (period-doubling)",
    xlims=(0,0.35), ylims=(0,2000), size=(600,600))
for (e,b) in [(ε_fA1,β_fA1),(ε_fA2,β_fA2),(ε_fB,β_fB),(ε_fC,β_fC)]
    ev,bv = clip(e,b); scatter!(plt_f, ev, bv; ms=1.5, mc=:black, label=false)
end
hline!(plt_f, [par.μ+par.α]; ls=:dot, lc=:gray, label="R₀=1")
display(plt_f)

# ══════════════════════════════════════════════════════════════════════════════
# t CURVES  (folds of periodic orbits)
# ══════════════════════════════════════════════════════════════════════════════

br_h600  = scan_at_β0(600.0)    # bp[1]≈ε 0.081, bp[2]≈ε 0.017
br_h1300 = scan_at_β0(1300.0)   # bp[3]≈ε 0.343, bp[4]≈ε 0.177
br_v003  = scan_at_ε(0.03, 500.0)

# t₁⁽¹⁾ — br_h1300, bp[3] at ε≈0.343
br_t1_up = cont2_fold_h(br_h1300, 3; ds= 5.0, dsmin=1e-6)
br_t1_dn = cont2_fold_h(br_h1300, 3; ds=-5.0, dsmin=1e-6)
ε_t1 = vcat(br_t1_dn.branch.ε, br_t1_up.branch.ε)
β_t1 = vcat(br_t1_dn.branch.β0, br_t1_up.branch.β0)

# t₂⁽¹⁾ — br_h600, bp[2] at ε≈0.017, extended via vertical scan at ε=0.03
br_t2_up   = cont2_fold_h(br_h600, 2; ds= 5.0, dsmin=1e-6)
br_t2_dn   = cont2_fold_h(br_h600, 2; ds=-5.0, dsmin=1e-6)
br_t2_vert = cont2_fold_v(br_v003, 2)
ε_t2  = vcat(br_t2_dn.branch.ε, br_t2_up.branch.ε)
β_t2  = vcat(br_t2_dn.branch.β0, br_t2_up.branch.β0)
ε_t2v = br_t2_vert.branch.ε
β_t2v = br_t2_vert.branch.β0

plt_t = plot(xlabel="ε", ylabel="β₀", title="t curves (folds of periodic orbits)",
    xlims=(0,0.35), ylims=(0,2000), size=(600,600))
e,b = clip(ε_t1, β_t1);  scatter!(plt_t, e, b; ms=1.5, mc=:black, label="t₁⁽¹⁾")
e,b = clip(ε_t2, β_t2);  scatter!(plt_t, e, b; ms=1.0, mc=:blue,  label="t₂⁽¹⁾")
e,b = clip(ε_t2v, β_t2v);scatter!(plt_t, e, b; ms=1.0, mc=:blue,  label=false)
hline!(plt_t, [par.μ+par.α]; ls=:dot, lc=:gray, label="R₀=1")
display(plt_t)

# ══════════════════════════════════════════════════════════════════════════════
# COMBINED  (f + t curves)
# ══════════════════════════════════════════════════════════════════════════════
plt_all = plot(xlabel="ε", ylabel="β₀", title="f and t curves",
    xlims=(0,0.35), ylims=(0,2000), size=(600,600))

for (e,b) in [(ε_fA1,β_fA1),(ε_fA2,β_fA2),(ε_fB,β_fB),(ε_fC,β_fC)]
    ev,bv = clip(e,b); scatter!(plt_all, ev, bv; ms=1.5, mc=:steelblue, label=false)
end
scatter!(plt_all, [NaN], [NaN]; mc=:steelblue, ms=3, label="f (period-doubling)")

for (e,b) in [(ε_t1,β_t1),(ε_t2,β_t2),(ε_t2v,β_t2v)]
    ev,bv = clip(e,b); scatter!(plt_all, ev, bv; ms=1.5, mc=:crimson, label=false)
end
scatter!(plt_all, [NaN], [NaN]; mc=:crimson, ms=3, label="t (folds)")

hline!(plt_all, [par.μ+par.α]; ls=:dot, lc=:gray, label="R₀=1")
display(plt_all)

