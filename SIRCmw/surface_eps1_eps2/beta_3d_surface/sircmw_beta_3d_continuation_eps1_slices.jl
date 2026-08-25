"""
Computes the Hopf bifurcation curves in the (eps2, beta0) plane for 61 fixed values of eps1
up to 3.0 using BifurcationKit's 2-parameter continuation in Julia.
Uses adaptive smaller step sizes for eps1 < 0.5 to safely trace near the BT point.
"""

using LinearAlgebra
using DelimitedFiles
import BifurcationKit as BK
import BifurcationKit: @optic, @set

include(joinpath(@__DIR__, "..", "..", "sircmw_utils.jl"))

const μ = PAR_BASE.μ
const α = PAR_BASE.α
const δ = PAR_BASE.δ
const γ = PAR_BASE.γ
const σ = PAR_BASE.σ

function sircmw!(du, u, p, t = 0)
    S, I, R, C = u
    eps1 = p.tilde_eps1 / SI_0
    eps2 = p.tilde_eps2 / SI_0
    b = p.beta0
    
    du[1] = μ*(1.0 - S) - b*S*I + (1.0 + eps2*S*I)*γ*C
    du[2] = b*S*I + σ*b*C*I - (μ + α)*I
    du[3] = (1.0 - σ)*b*C*I + α*I - μ*R - (1.0 + eps1*S*I)*δ*R
    du[4] = (1.0 + eps1*S*I)*δ*R - b*C*I - μ*C - (1.0 + eps2*S*I)*γ*C
    du
end

# Trace a 2D Hopf bifurcation curve in the (eps2, beta0) plane for a fixed eps1
function trace_hopf_for_eps1(fixed_eps1)
    # Scan beta0 and eps2 to find an unstable endemic equilibrium seed point
    local seed = nothing
    for b_test in range(200.0, 1800.0, length=9)
        for e2_test in range(0.1, 2.9, length=15)
            roots = get_endemic_roots(fixed_eps1 / SI_0, e2_test / SI_0, b_test)
            if !isempty(roots)
                S, I, R, C = roots[1]
                b = b_test
                eps1 = fixed_eps1 / SI_0
                eps2 = e2_test / SI_0
                J = [
                    -μ - b*I + eps2*I*γ*C     -b*S + eps2*S*γ*C                       0.0                            γ*(1.0 + eps2*S*I);
                    b*I                        b*S + σ*b*C - (μ + α)                  0.0                            σ*b*I;
                    -eps1*I*δ*R                (1.0 - σ)*b*C + α - eps1*S*δ*R         -(μ + δ*(1.0 + eps1*S*I))      (1.0 - σ)*b*I;
                    eps1*I*δ*R - eps2*I*γ*C    eps1*S*δ*R - b*C - eps2*S*γ*C          δ*(1.0 + eps1*S*I)             -(b*I + μ + γ*(1.0 + eps2*S*I))
                ]
                max_re = maximum(real(eigvals(J)))
                if max_re > 0.0
                    seed = (beta0 = b_test, eps2 = e2_test, u0 = [S, I, R, C])
                    break
                end
            end
        end
        if seed !== nothing
            break
        end
    end
    
    if seed === nothing
        return nothing
    end
    
    p_start = (μ = μ, α = α, δ = δ, γ = γ, σ = σ, beta0 = seed.beta0, tilde_eps1 = fixed_eps1, tilde_eps2 = seed.eps2)
    u0_eq = seed.u0
    
    # 1. Continue in tilde_eps2 to locate the Hopf crossing point
    prob_eq = BK.ODEBifProblem(sircmw!, u0_eq, p_start, (@optic _.tilde_eps2);
        record_from_solution = (x, p; k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))
        
    opts_eq = BK.ContinuationPar(
        p_min = 0.0, p_max = 3.0,
        ds = -0.01, dsmin = 1e-6, dsmax = 0.05, max_steps = 500,
        newton_options = BK.NewtonPar(tol = 1e-9, max_iterations = 25, linesearch = true),
        detect_bifurcation = 3, n_inversion = 6, nev = 4)
        
    br_eq = BK.continuation(prob_eq, BK.PALC(), opts_eq; verbosity = 0)
    hopf_idx = findall(sp -> sp.type == :hopf, br_eq.specialpoint)
    if isempty(hopf_idx)
        # Try forward continuation if backward didn't hit it
        opts_eq = @set opts_eq.ds = 0.01
        br_eq = BK.continuation(prob_eq, BK.PALC(), opts_eq; verbosity = 0)
        hopf_idx = findall(sp -> sp.type == :hopf, br_eq.specialpoint)
        if isempty(hopf_idx)
            return nothing
        end
    end
    h_idx = hopf_idx[1]
    
    # 2. Run 2-parameter continuation in (tilde_eps2, beta0)
    # Use smaller, tighter step bounds when eps1 is small to handle stiffness near the BT point
    ds_val = fixed_eps1 < 0.5 ? 0.25 : 1.0
    dsmax_val = fixed_eps1 < 0.5 ? 2.5 : 10.0
    dsmin_val = fixed_eps1 < 0.5 ? 1e-7 : 1e-6
    
    opts_hopf2p = BK.ContinuationPar(
        p_min = 125.0, p_max = 2000.0,
        ds = ds_val, dsmin = dsmin_val, dsmax = dsmax_val, max_steps = 2500,
        newton_options = BK.NewtonPar(tol = 1e-8, max_iterations = 25, linesearch = true),
        detect_bifurcation = 1, nev = 4)
        
    all_branch_points = Tuple{Float64,Float64,Float64,String}[]
    
    # Trace in both directions along the second parameter beta0
    for (label, ds_sign) in (("Forward", 1.0), ("Backward", -1.0))
        try
            opts_dir = @set opts_hopf2p.ds = ds_sign * ds_val
            br = BK.continuation(br_eq, h_idx, (@optic _.beta0), opts_dir;
                detect_codim2_bifurcation = 2, start_with_eigen = true, verbosity = 0,
                bdlinsolver = BK.MatrixBLS())
                
            branch_pts = Tuple{Float64,Float64,Float64,String}[]
            for (te2, b0) in zip(br.branch.tilde_eps2, br.branch.beta0)
                if 0.0 <= te2 <= 3.0 && 100.0 <= b0 <= 2000.0
                    push!(branch_pts, (fixed_eps1, te2, b0, lowercase(label)))
                end
            end
            
            if ds_sign == 1.0
                append!(all_branch_points, reverse(branch_pts))
            else
                if !isempty(all_branch_points) && !isempty(branch_pts)
                    append!(all_branch_points, branch_pts[2:end])
                else
                    append!(all_branch_points, branch_pts)
                end
            end
        catch e
        end
    end
    
    return all_branch_points
end

function run_eps1_slices()
    eps1_slices = range(0.0, 3.0, length=121)
    
    all_points_thread = Vector{Vector{Tuple{Float64,Float64,Float64,String}}}(undef, length(eps1_slices))
    
    println("Tracing Hopf curves across $(length(eps1_slices)) eps1 slices in parallel...")
    Threads.@threads for k in 1:length(eps1_slices)
        e1 = eps1_slices[k]
        println("  Starting eps1 slice: $(round(e1, digits=2))")
        pts = trace_hopf_for_eps1(e1)
        if pts !== nothing
            all_points_thread[k] = pts
            println("  Success: traced $(length(pts)) points for eps1 = $(round(e1, digits=2))")
        else
            all_points_thread[k] = Tuple{Float64,Float64,Float64,String}[]
            println("  Warning: no Hopf curve found for eps1 = $(round(e1, digits=2))")
        end
    end
    
    # Combine results and inject NaN separators between slices
    all_points = Tuple{Float64,Float64,Float64,String}[]
    for k in 1:length(eps1_slices)
        pts = all_points_thread[k]
        if !isempty(pts)
            if !isempty(all_points)
                push!(all_points, (NaN, NaN, NaN, ""))
            end
            append!(all_points, pts)
        end
    end
    
    output_path = joinpath(@__DIR__, "hopf_slices_eps1_indexed.csv")
    open(output_path, "w") do io
        println(io, "eps1,eps2,beta0,branch")
        for (te1, te2, bv, br_name) in all_points
            println(io, "$te1,$te2,$bv,$br_name")
        end
    end
    println("Exported continuation results to: $output_path")
end

run_eps1_slices()
