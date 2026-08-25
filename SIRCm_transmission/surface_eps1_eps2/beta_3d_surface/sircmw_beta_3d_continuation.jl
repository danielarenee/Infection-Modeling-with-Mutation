"""
Computes the Hopf bifurcation curves in the (eps1, eps2) plane at various beta0 
using BifurcationKit's 2-parameter continuation in Julia
Its output is used by sircm_beta_3d_sweep.py
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

# for each beta val...
function trace_hopf_curve_at_beta(beta_val)
    grid_vals = range(0.0, 3.0, length=10) # grid 
    
    # evaluate stability at all grid points
    grid_data = Matrix{Any}(nothing, 10, 10)
    for (j, e2) in enumerate(grid_vals)
        for (i, e1) in enumerate(grid_vals)
            # find the endemic equilibrium root algebraically 
            roots = get_endemic_roots(e1 / SI_0, e2 / SI_0, beta_val)
            if !isempty(roots)
                S, I, R, C = roots[1]
                eps1 = e1 / SI_0
                eps2 = e2 / SI_0
                b = beta_val
                # construct jacobian
                J = [
                    -μ - b*I + eps2*I*γ*C     -b*S + eps2*S*γ*C                       0.0                            γ*(1.0 + eps2*S*I);
                    b*I                        b*S + σ*b*C - (μ + α)                  0.0                            σ*b*I;
                    -eps1*I*δ*R                (1.0 - σ)*b*C + α - eps1*S*δ*R         -(μ + δ*(1.0 + eps1*S*I))      (1.0 - σ)*b*I;
                    eps1*I*δ*R - eps2*I*γ*C    eps1*S*δ*R - b*C - eps2*S*γ*C          δ*(1.0 + eps1*S*I)             -(b*I + μ + γ*(1.0 + eps2*S*I))
                ]
                # evaluate eigenvals and save stability
                max_re = maximum(real(eigvals(J)))
                sgn = max_re < 0.0 ? 1 : -1
                grid_data[i, j] = (e1, e2, S, I, R, C, sgn)
            else
                grid_data[i, j] = (e1, e2, 0.0, 0.0, 0.0, 0.0, 0)
            end
        end
    end
    
    # find crossing segments
    segments = []
    # horizontal segments (where stability changes)
    for j in 1:10
        for i in 1:9
            pt1 = grid_data[i, j]
            pt2 = grid_data[i+1, j]
            if pt1[7] != 0 && pt2[7] != 0 && pt1[7] != pt2[7]
                push!(segments, (pt1, pt2))
            end
        end
    end
    # vertical segments
    for i in 1:10
        for j in 1:9
            pt1 = grid_data[i, j]
            pt2 = grid_data[i, j+1]
            if pt1[7] != 0 && pt2[7] != 0 && pt1[7] != pt2[7]
                push!(segments, (pt1, pt2))
            end
        end
    end
    
    if isempty(segments)
        return nothing
    end
    
    all_traced_points = Tuple{Float64,Float64,Float64}[]
    
    # computes euclidean distance of a candidate to all traced points 
    # to avoid re-tracing the same curve
    function is_near_existing(e1, e2)
        for (te1, te2, _) in all_traced_points
            if isnan(te1) || isnan(te2)
                continue
            end
            if sqrt((te1 - e1)^2 + (te2 - e2)^2) < 0.2
                return true
            end
        end
        return false
    end
    
    # for each segment find Hopf and continue
    for (pt1, pt2) in segments
        e1_s, e2_s, S_s, I_s, R_s, C_s, sgn1 = pt1
        e1_u, e2_u, _, _, _, _, sgn2 = pt2
        
        # ensure that pt1 always represents the stable endpoint 
        if sgn1 == -1
            e1_s, e1_u = e1_u, e1_s
            e2_s, e2_u = e2_u, e2_s
            S_s, I_s, R_s, C_s = pt2[3], pt2[4], pt2[5], pt2[6]
        end
        
        # deduplication check
        if is_near_existing(e1_s, e2_s) && is_near_existing(e1_u, e2_u)
            continue
        end
        
        p = (μ = μ, α = α, δ = δ, γ = γ, σ = σ, beta0 = beta_val,
             s = 0.0, e1_s = e1_s, e1_u = e1_u, e2_s = e2_s, e2_u = e2_u)
        u0_eq = [S_s, I_s, R_s, C_s]
        
        # vector field parameterized by continuation parameter 's'
        # this one is written so that p.s is a dial for eps1 and eps2
        function sircmw_s!(du, u, p, t = 0)
            S, I, R, C = u
            eps1_rel = (1.0 - p.s)*p.e1_s + p.s*p.e1_u
            eps2_rel = (1.0 - p.s)*p.e2_s + p.s*p.e2_u
            eps1 = eps1_rel / SI_0
            eps2 = eps2_rel / SI_0
            b = p.beta0
            
            du[1] = μ*(1.0 - S) - b*S*I + (1.0 + eps2*S*I)*γ*C
            du[2] = b*S*I + σ*b*C*I - (μ + α)*I
            du[3] = (1.0 - σ)*b*C*I + α*I - μ*R - (1.0 + eps1*S*I)*δ*R
            du[4] = (1.0 + eps1*S*I)*δ*R - b*C*I - μ*C - (1.0 + eps2*S*I)*γ*C
            du
        end
        
        # 1D continuation along the segment 
        prob_eq = BK.ODEBifProblem(sircmw_s!, u0_eq, p, (@optic _.s);
            record_from_solution = (x, p; k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))
            
        opts_eq = BK.ContinuationPar(
            p_min = 0.0, p_max = 2.0,
            ds = 0.01, dsmin = 1e-6, dsmax = 0.05, max_steps = 500,
            newton_options = BK.NewtonPar(tol = 1e-9, max_iterations = 25, linesearch = true),
            detect_bifurcation = 3, n_inversion = 6, nev = 4)
            
        br_eq = BK.continuation(prob_eq, BK.PALC(), opts_eq; verbosity = 0)
        hopf_idx = findall(sp -> sp.type == :hopf, br_eq.specialpoint)
        if isempty(hopf_idx)
            continue
        end
        s_Hopf = br_eq.specialpoint[hopf_idx[1]].param
        
        # map the found s_Hopf parameter back to the corresponding (eps1, eps2) coordinate space
        hopf_eps1 = (1.0 - s_Hopf)*e1_s + s_Hopf*e1_u
        hopf_eps2 = (1.0 - s_Hopf)*e2_s + s_Hopf*e2_u
        
        # skip if the located Hopf point is already close to an existing trace
        if is_near_existing(hopf_eps1, hopf_eps2)
            continue
        end
        
        # solve algebraically for the exact endemic equilibrium states at the Hopf crossing
        roots_hopf = get_endemic_roots(hopf_eps1 / SI_0, hopf_eps2 / SI_0, beta_val)
        if isempty(roots_hopf)
            continue
        end
        S_h, I_h, R_h, C_h = roots_hopf[1]
        
        # construct the 2-parameter bifurcation problem with active parameters eps1 and eps2
        p_2p = (μ = μ, α = α, δ = δ, γ = γ, σ = σ, beta0 = beta_val, tilde_eps1 = hopf_eps1, tilde_eps2 = hopf_eps2)
        u0_h = [S_h, I_h, R_h, C_h]
        
        prob_eq_2p = BK.ODEBifProblem(sircmw!, u0_h, p_2p, (@optic _.tilde_eps1);
            record_from_solution = (x, p; k...) -> (S=x[1], I=x[2], R=x[3], C=x[4]))
            
        opts_eq_2p = BK.ContinuationPar(
            p_min = 0.0, p_max = 3.0,
            ds = 0.005, dsmin = 1e-6, dsmax = 0.05, max_steps = 500,
            newton_options = BK.NewtonPar(tol = 1e-9, max_iterations = 25, linesearch = true),
            detect_bifurcation = 3, n_inversion = 6, nev = 4)
            
        br_eq_2p = BK.continuation(prob_eq_2p, BK.PALC(), opts_eq_2p; verbosity = 0)
        hopf_idx_2p = findall(sp -> sp.type == :hopf, br_eq_2p.specialpoint)
        if isempty(hopf_idx_2p)
            continue
        end
        h_idx_2p = hopf_idx_2p[1]
        
        # adjust step sizes for continuation to handle stiffness at beta >= 800
        ds_val = beta_val >= 800.0 ? 0.001 : 0.01
        dsmax_val = beta_val >= 800.0 ? 0.01 : 0.05
        
        opts_hopf2p = BK.ContinuationPar(
            p_min = 0.0, p_max = 3.0,
            ds = ds_val, dsmin = 1e-6, dsmax = dsmax_val, max_steps = 2000,
            newton_options = BK.NewtonPar(tol = 1e-8, max_iterations = 25, linesearch = true),
            detect_bifurcation = 1, nev = 4)
            
        forward_points = Tuple{Float64,Float64,Float64}[]
        backward_points = Tuple{Float64,Float64,Float64}[]
        
        # trace the bifurcation curve in the positive parameter step direction
        try
            opts_dir = @set opts_hopf2p.ds = 1.0 * ds_val
            br = BK.continuation(br_eq_2p, h_idx_2p, (@optic _.tilde_eps2), opts_dir;
                detect_codim2_bifurcation = 2, start_with_eigen = true, verbosity = 0,
                bdlinsolver = BK.MatrixBLS())
                
            for (idx, (te1, te2)) in enumerate(zip(br.branch.tilde_eps1, br.branch.tilde_eps2))
                if 0.0 <= te1 <= 3.0 && 0.0 <= te2 <= 3.0
                    dist = sqrt((te1 - br.branch.tilde_eps1[1])^2 + (te2 - br.branch.tilde_eps2[1])^2)
                    if idx > 30 && dist < 0.05
                        break
                    end
                    push!(forward_points, (te1, te2, beta_val))
                else
                    break
                end
            end
        catch e
        end
        
        # Backward branch
        # trace the bifurcation curve in the negative parameter step direction
        try
            opts_dir = @set opts_hopf2p.ds = -1.0 * ds_val
            br = BK.continuation(br_eq_2p, h_idx_2p, (@optic _.tilde_eps2), opts_dir;
                detect_codim2_bifurcation = 2, start_with_eigen = true, verbosity = 0,
                bdlinsolver = BK.MatrixBLS())
                
            for (idx, (te1, te2)) in enumerate(zip(br.branch.tilde_eps1, br.branch.tilde_eps2))
                if 0.0 <= te1 <= 3.0 && 0.0 <= te2 <= 3.0
                    dist = sqrt((te1 - br.branch.tilde_eps1[1])^2 + (te2 - br.branch.tilde_eps2[1])^2)
                    if idx > 30 && dist < 0.05
                        break
                    end
                    push!(backward_points, (te1, te2, beta_val))
                else
                    break
                end
            end
        catch e
        end
        
        # combine
        combined = Tuple{Float64,Float64,Float64}[]
        if !isempty(forward_points)
            append!(combined, reverse(forward_points))
        end
        if !isempty(backward_points)
            start_idx = !isempty(forward_points) ? 2 : 1
            if length(backward_points) >= start_idx
                append!(combined, backward_points[start_idx:end])
            end
        end
        
        if !isempty(combined)
            if !isempty(all_traced_points)
                push!(all_traced_points, (NaN, NaN, beta_val))
            end
            append!(all_traced_points, combined)
        end
    end
    return all_traced_points
end

function export_slices()
    # arrays of betas
    betas_low = range(100.0, 400.0, length=5)
    betas_mid = range(400.0, 1000.0, length=13)
    betas_high = range(1000.0, 2000.0, length=6)
    slice_betas = unique(vcat(betas_low, betas_mid, betas_high))
    
    all_points = Tuple{Float64,Float64,Float64}[]
    
    println("Tracing Hopf curves across $(length(slice_betas)) beta slices...")
    for (k, bv) in enumerate(slice_betas)
        println("[$k/$(length(slice_betas))] Beta0 = $(round(bv, digits=1))")
        pts = trace_hopf_curve_at_beta(bv)
        if pts === nothing || isempty(pts)
            println("  no Hopf curve found - skipping")
            continue
        end
        append!(all_points, pts)
        println("  Success: traced $(length(pts)) points")
    end
    
    output_path = joinpath(@__DIR__, "hopf_slices_eps1_eps2_beta.csv")
    open(output_path, "w") do io
        println(io, "eps1,eps2,beta0")
        for (te1, te2, bv) in all_points
            println(io, "$te1,$te2,$bv")
        end
    end
    println("Exported continuation results to: $output_path")
end

export_slices()
