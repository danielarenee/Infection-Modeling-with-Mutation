import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.integrate import solve_ivp

# Add SIRCmw directory to path to import sircmw_utils
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent.parent))

from sircmw_I_utils import (
    sircmw,
    integrate_with_reseeding,
    get_algebraic_equilibria,
    MU as mu,
    ALPHA as alpha,
    DELTA as delta,
    GAMMA as gamma,
    SIGMA as sigma,
    BETA0 as beta0
)

TILDE_EPS_LIST = [0.0, 1.0, 2.0]

# Calculate endemic equilibrium at tilde_eps = 0 to define exact scaling factor
p_base = {
    'beta0': beta0,
    'sigma': sigma,
    'mu': mu,
    'alpha': alpha,
    'delta': delta,
    'gamma': gamma,
    'si_0': 1.0
}
base_eqs = get_algebraic_equilibria(0.0, p_base)
endemic_base_eq = [eq for eq in base_eqs if eq[1] > 1e-5][0]
I_star_base = endemic_base_eq[1]
scale_factor = I_star_base

# List of 5 diverse initial conditions to show complete flow behavior
Y0_LISTS = [
    np.array([0.2, 0.001, 0.499, 0.3]),
    np.array([0.1, 0.2, 0.4, 0.3]),
    np.array([0.4, 0.1, 0.25, 0.25]),
    np.array([0.6, 0.05, 0.15, 0.2]),
    np.array([0.15, 0.5, 0.15, 0.2])
]

# Time span for the main simulations
T_SPAN = (0.0, 150.0)

# Output filename base
OUTPUT_FILENAME_BASE = "sircmw_phase_portraits"

# Cache file path
CACHE_FILE = SCRIPT_DIR.parent / "sircmw_I_phase_portraits_cache.npz"

# Buffered boundary event to stop integrations when they go significantly out-of-bounds
def simplex_boundary_event(t, y, p_val):
    S, I, R, C = y
    # Small buffer to prevent immediate cutoff at t=0 when starting near borders
    return min(S + 1e-4, I + 1e-4, 1.0001 - S - I)

simplex_boundary_event.terminal = True
simplex_boundary_event.direction = -1

# Matplotlib publication settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 15,
    'legend.fontsize': 9
})

def run_simulation(tilde_eps, y0, t_span):
    eps = tilde_eps / scale_factor
    p = {
        'beta0': beta0,
        'sigma': sigma,
        'mu': mu,
        'alpha': alpha,
        'delta': delta,
        'gamma': gamma,
        'eps': eps
    }
    t_arr, Y_arr, n_ev = integrate_with_reseeding(
        sircmw, t_span, y0, p,
        method='Radau', rtol=1e-7, atol=1e-9
    )
    return t_arr, Y_arr

def load_or_compute_data():
    if CACHE_FILE.exists():
        print(f"Loading cached simulations from {CACHE_FILE}. Delete this file to force recalculation.")
        try:
            cache_data = np.load(CACHE_FILE, allow_pickle=True)
            solutions = {}
            manual_flows = {}
            for te in TILDE_EPS_LIST:
                solutions[te] = []
                for idx in range(len(Y0_LISTS)):
                    solutions[te].append((cache_data[f"t_arr_{te}_{idx}"], cache_data[f"Y_arr_{te}_{idx}"]))
                manual_flows[te] = cache_data[f"manual_flows_{te}"].tolist()
            return solutions, manual_flows
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing simulations...")
            
    solutions = {}
    manual_flows = {}
    
    y0_ref = Y0_LISTS[0]
    R_start_ratio = y0_ref[2] / (y0_ref[2] + y0_ref[3])
    C_start_ratio = y0_ref[3] / (y0_ref[2] + y0_ref[3])
    
    # Strategic starting points covering boundaries exactly (forward-only flow to attractor)
    starts = []
    N_boundary = 18
    # 1. Left boundary: S = 0.001, vary I
    for i0 in np.linspace(0.001, 0.99, N_boundary):
        starts.append((0.001, i0))
    # 2. Bottom boundary: I = 0.001, vary S
    for s0 in np.linspace(0.001, 0.99, N_boundary):
        starts.append((s0, 0.001))
    # 3. Diagonal boundary: S + I = 0.995
    for s0 in np.linspace(0.005, 0.99, N_boundary):
        starts.append((s0, 0.995 - s0))
                
    cache_dict = {}
    
    for te in TILDE_EPS_LIST:
        print(f"  Computing main trajectories for tilde_eps = {te}...")
        solutions[te] = []
        for idx, y0 in enumerate(Y0_LISTS):
            t_arr, Y_arr = run_simulation(te, y0, T_SPAN)
            solutions[te].append((t_arr, Y_arr))
            cache_dict[f"t_arr_{te}_{idx}"] = t_arr
            cache_dict[f"Y_arr_{te}_{idx}"] = Y_arr
        
        print(f"  Computing manual flow lines for tilde_eps = {te}...")
        eps = te / scale_factor
        p_sim = {
            'beta0': beta0,
            'sigma': sigma,
            'mu': mu,
            'alpha': alpha,
            'delta': delta,
            'gamma': gamma,
            'eps': eps
        }
        
        flows = []
        for s0, i0 in starts:
            r0 = (1.0 - s0 - i0) * R_start_ratio
            c0 = (1.0 - s0 - i0) * C_start_ratio
            y_init = np.array([s0, i0, r0, c0])
            
            # High-precision forward-only integration for full duration
            sol = solve_ivp(
                sircmw, (0.0, 150.0), y_init, args=(p_sim,),
                method='Radau', rtol=1e-7, atol=1e-9,
                events=simplex_boundary_event
            )
            flows.append(np.column_stack([sol.y[0], sol.y[1]]))
            
        manual_flows[te] = flows
        cache_dict[f"manual_flows_{te}"] = np.array(flows, dtype=object)
        
    np.savez_compressed(CACHE_FILE, **cache_dict)
    print(f"Saved computed simulations cache to {CACHE_FILE}")
    
    return solutions, manual_flows

def generate_plots(mode, solutions, manual_flows):
    print(f"Generating phase portraits ({mode} mode)...")
    fig, axes = plt.subplots(1, len(TILDE_EPS_LIST), figsize=(15, 5), sharey=True)
    
    for i, te in enumerate(TILDE_EPS_LIST):
        eps = te / scale_factor
        ax1 = axes[i]
        
        # Get analytical endemic equilibrium
        p_eq = {
            'beta0': beta0,
            'sigma': sigma,
            'mu': mu,
            'alpha': alpha,
            'delta': delta,
            'gamma': gamma,
            'si_0': scale_factor
        }
        eqs = get_algebraic_equilibria(te, p_eq)
        endemic_eqs = [eq for eq in eqs if eq[1] > 1e-5]
        if len(endemic_eqs) > 0:
            S_eq, I_eq, R_eq, C_eq = endemic_eqs[0]
        else:
            S_eq, I_eq, R_eq, C_eq = eqs[0]
            
        if mode == 'manual':
            flow_color = mcolors.to_rgba('C0', alpha=0.35)
            flows = manual_flows[te]
            for flow in flows:
                ax1.plot(flow[:, 0], flow[:, 1], color=flow_color, lw=0.8)
                
            # Plot ONLY the single original simulation trajectory
            t_arr, Y_arr = solutions[te][0]
            S_traj, I_traj, _, _ = Y_arr
            ax1.plot(S_traj, I_traj, color='red', ls='--', lw=1.6, label='Trajectory' if i == 0 else None)
            ax1.scatter([S_traj[0]], [I_traj[0]], color='green', s=45, edgecolor='k', zorder=6, label='Initial condition' if i == 0 else None)
            
            # Plot analytical equilibrium points (red stars)
            for eq_idx, eq in enumerate(eqs):
                curr_S_eq, curr_I_eq, _, _ = eq
                lbl = 'Equilibrium' if (i == 0 and eq_idx == 0) else None
                ax1.scatter([curr_S_eq], [curr_I_eq], color='red', marker='*', s=140, edgecolor='k', zorder=8, label=lbl)
                
            ax1.set_xlabel('S (Susceptible)')
            if i == 0:
                ax1.set_ylabel('I (Infected)')
            else:
                ax1.set_ylabel('')
                ax1.tick_params(labelleft=False)
                
            ax1.set_xlim(0.0, 1.0)
            ax1.set_ylim(0.0, 1.0)
            
        elif mode == 'SI_streamplot':
            S_grid = np.linspace(0.0, 1.0, 50)
            I_grid = np.linspace(0.0, 1.0, 50)
            S_mesh, I_mesh = np.meshgrid(S_grid, I_grid)
            rem1 = np.maximum(1.0 - I_mesh - S_mesh, 0.0)
            
            # Analytical equilibrium scaling (preserves concentric limit cycle shape alignment!)
            sum_RC = R_eq + C_eq
            if sum_RC > 1e-12:
                R_mesh = rem1 * (R_eq / sum_RC)
                C_mesh = rem1 * (C_eq / sum_RC)
            else:
                R_mesh = rem1 / 2.0
                C_mesh = rem1 / 2.0
            
            dS_mesh = mu * (1.0 - S_mesh) - beta0 * S_mesh * I_mesh + (1.0 + eps * I_mesh) * gamma * C_mesh
            dI_mesh = beta0 * S_mesh * I_mesh + sigma * beta0 * C_mesh * I_mesh - (mu + alpha) * I_mesh
            
            dS_mesh[rem1 <= 0] = np.nan
            dI_mesh[rem1 <= 0] = np.nan
            
            flow_color = mcolors.to_rgba('C0', alpha=0.55)
            ax1.streamplot(S_grid, I_grid, dS_mesh, dI_mesh, broken_streamlines=False, density=0.8, color=flow_color)
            
            # Plot all simulation trajectories
            for idx, (t_arr, Y_arr) in enumerate(solutions[te]):
                S_traj, I_traj, _, _ = Y_arr
                lbl = 'Trajectory' if (i == 0 and idx == 0) else None
                ax1.plot(S_traj, I_traj, color='red', ls='--', lw=1.6, label=lbl)
                
                lbl_ic = 'Initial condition' if (i == 0 and idx == 0) else None
                ax1.scatter([S_traj[0]], [I_traj[0]], color='green', s=45, edgecolor='k', zorder=6, label=lbl_ic)
            
            for eq_idx, eq in enumerate(eqs):
                curr_S_eq, curr_I_eq, _, _ = eq
                lbl = 'Equilibrium' if (i == 0 and eq_idx == 0) else None
                ax1.scatter([curr_S_eq], [curr_I_eq], color='red', marker='*', s=140, edgecolor='k', zorder=8, label=lbl)
                
            ax1.set_xlabel('S (Susceptible)')
            if i == 0:
                ax1.set_ylabel('I (Infected)')
            else:
                ax1.set_ylabel('')
                ax1.tick_params(labelleft=False)
                
            ax1.set_xlim(0.0, 1.0)
            ax1.set_ylim(0.0, 1.0)
            
        ax1.grid(True, alpha=0.2)
        
        # Add solid diagonal simplex boundary line (S+I=1 or R+I=1) for visual cleanliness
        ax1.plot([0.0, 1.0], [1.0, 0.0], color='C0', alpha=0.6, lw=1.2, ls='-', zorder=4)
        
        # Avoid overlapping x-tick labels between adjacent subplots using FuncFormatter
        def make_formatter(index):
            def formatter(x, pos):
                if abs(x - 0.0) < 1e-5:
                    return '' if index > 0 else '0.0'
                elif abs(x - 1.0) < 1e-5:
                    return '' if index < len(TILDE_EPS_LIST) - 1 else '1.0'
                else:
                    return f'{x:.1f}'
            return formatter
        ax1.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(make_formatter(i)))
        
        # Text box in the right top corner for epsilon tilde
        ax1.text(0.95, 0.95, f'$\\tilde{{\\epsilon}} = {te:.1f}$', transform=ax1.transAxes, 
                 ha='right', va='top', fontsize=13,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='lightgray', alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, top=0.86)
    
    # Legend outside the boxes, on top of the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=True, fontsize=13.5, markerscale=1.2, handlelength=2.2)
    
    out_filename = f"{OUTPUT_FILENAME_BASE}_{mode}"
    if mode != 'manual':
        fig_save_path_png = SCRIPT_DIR.parent / f"{out_filename}.png"
        plt.savefig(fig_save_path_png, dpi=300, bbox_inches='tight')
        print(f"Successfully generated and saved clean 2D phase portraits PNG to: {fig_save_path_png}")
    
    fig_save_path_pdf = SCRIPT_DIR.parent / f"{out_filename}.pdf"
    plt.savefig(fig_save_path_pdf, dpi=300, bbox_inches='tight')
    print(f"Successfully generated and saved clean 2D phase portraits PDF to: {fig_save_path_pdf}")
    plt.close()

def main():
    print(f"Exact scaling factor: {scale_factor}")
    solutions, manual_flows = load_or_compute_data()
    generate_plots('SI_streamplot', solutions, manual_flows)
    generate_plots('manual', solutions, manual_flows)

if __name__ == "__main__":
    main()
