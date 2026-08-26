"""
Generates phase portraits (SI plane) of the SIRCmw transmission model
for multiple values of tilde_eps, using integrated flow lines.

Extracted from sircmw_phase_portrait.ipynb.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.integrate import solve_ivp

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent.parent))

from sircmw_utils import (
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

# Scaling: SI_ref = S0_ref * I0_ref (from original notebook)
S0_ref, I0_ref = 0.2, 0.001
scale_factor = S0_ref * I0_ref  # = 0.0002

TILDE_EPS_LIST = [0.5, 1.5, 1.7]

Y0_LISTS = [
    np.array([0.2, 0.001, 0.499, 0.3]),
    np.array([0.1, 0.2, 0.4, 0.3]),
    np.array([0.4, 0.1, 0.25, 0.25]),
    np.array([0.6, 0.05, 0.15, 0.2]),
    np.array([0.15, 0.5, 0.15, 0.2])
]

T_SPAN = (0.0, 150.0)
CACHE_FILE = SCRIPT_DIR / "sircmw_phase_portraits_cache.npz"

# Stop flow integrations when trajectory leaves simplex
def simplex_boundary_event(t, y, p_val):
    S, I, R, C = y
    return min(S + 1e-4, I + 1e-4, 1.0001 - S - I)

simplex_boundary_event.terminal = True
simplex_boundary_event.direction = -1

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
    p = {'beta0': beta0, 'sigma': sigma, 'mu': mu, 'alpha': alpha,
         'delta': delta, 'gamma': gamma, 'eps': eps}
    t_arr, Y_arr, _ = integrate_with_reseeding(
        sircmw, t_span, y0, p, method='Radau', rtol=1e-7, atol=1e-9
    )
    return t_arr, Y_arr

def load_or_compute_data():
    if CACHE_FILE.exists():
        print(f"Loading cached simulations from {CACHE_FILE}.")
        try:
            cache_data = np.load(CACHE_FILE, allow_pickle=True)
            solutions = {te: [(cache_data[f"t_arr_{te}_{idx}"], cache_data[f"Y_arr_{te}_{idx}"])
                               for idx in range(len(Y0_LISTS))]
                         for te in TILDE_EPS_LIST}
            manual_flows = {te: cache_data[f"manual_flows_{te}"].tolist() for te in TILDE_EPS_LIST}
            return solutions, manual_flows
        except Exception as e:
            print(f"Failed to load cache: {e}. Recomputing...")

    solutions, manual_flows, cache_dict = {}, {}, {}

    y0_ref = Y0_LISTS[0]
    R_ratio = y0_ref[2] / (y0_ref[2] + y0_ref[3])
    C_ratio = y0_ref[3] / (y0_ref[2] + y0_ref[3])

    # Boundary starting points to cover the full simplex
    starts = []
    N = 18
    for i0 in np.linspace(0.001, 0.99, N):
        starts.append((0.001, i0))
    for s0 in np.linspace(0.001, 0.99, N):
        starts.append((s0, 0.001))
    for s0 in np.linspace(0.005, 0.99, N):
        starts.append((s0, 0.995 - s0))

    for te in TILDE_EPS_LIST:
        print(f"  Computing trajectories for tilde_eps = {te}...")
        solutions[te] = []
        for idx, y0 in enumerate(Y0_LISTS):
            t_arr, Y_arr = run_simulation(te, y0, T_SPAN)
            solutions[te].append((t_arr, Y_arr))
            cache_dict[f"t_arr_{te}_{idx}"] = t_arr
            cache_dict[f"Y_arr_{te}_{idx}"] = Y_arr

        print(f"  Computing flow lines for tilde_eps = {te}...")
        eps = te / scale_factor
        p_sim = {'beta0': beta0, 'sigma': sigma, 'mu': mu, 'alpha': alpha,
                 'delta': delta, 'gamma': gamma, 'eps': eps}
        flows = []
        for s0, i0 in starts:
            r0 = (1.0 - s0 - i0) * R_ratio
            c0 = (1.0 - s0 - i0) * C_ratio
            sol = solve_ivp(sircmw, (0.0, 150.0), [s0, i0, r0, c0], args=(p_sim,),
                            method='Radau', rtol=1e-7, atol=1e-9, events=simplex_boundary_event)
            flows.append(np.column_stack([sol.y[0], sol.y[1]]))
        manual_flows[te] = flows
        cache_dict[f"manual_flows_{te}"] = np.array(flows, dtype=object)

    np.savez_compressed(CACHE_FILE, **cache_dict)
    print(f"Saved cache to {CACHE_FILE}")
    return solutions, manual_flows

def generate_plots(solutions, manual_flows):
    fig, axes = plt.subplots(1, len(TILDE_EPS_LIST), figsize=(15, 5), sharey=True)
    flow_color = mcolors.to_rgba('C0', alpha=0.35)

    for i, te in enumerate(TILDE_EPS_LIST):
        ax = axes[i]

        p_eq = {'beta0': beta0, 'sigma': sigma, 'mu': mu, 'alpha': alpha,
                'delta': delta, 'gamma': gamma, 'si_0': scale_factor}
        eqs = get_algebraic_equilibria(te, p_eq)

        # Flow lines
        for flow in manual_flows[te]:
            ax.plot(flow[:, 0], flow[:, 1], color=flow_color, lw=0.8)

        # Main trajectory
        t_arr, Y_arr = solutions[te][0]
        S_traj, I_traj, _, _ = Y_arr
        ax.plot(S_traj, I_traj, color='red', ls='--', lw=1.6, label='Trajectory' if i == 0 else None)
        ax.scatter([S_traj[0]], [I_traj[0]], color='green', s=45, edgecolor='k', zorder=6,
                   label='Initial condition' if i == 0 else None)

        # Equilibria
        for eq_idx, eq in enumerate(eqs):
            S_eq, I_eq, _, _ = eq
            lbl = 'Equilibrium' if (i == 0 and eq_idx == 0) else None
            ax.scatter([S_eq], [I_eq], color='red', marker='*', s=140, edgecolor='k', zorder=8, label=lbl)

        ax.set_xlabel('S (Susceptible)')
        ax.set_ylabel('I (Infected)' if i == 0 else '')
        if i > 0:
            ax.tick_params(labelleft=False)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.2)
        ax.plot([0.0, 1.0], [1.0, 0.0], color='C0', alpha=0.6, lw=1.2, ls='-', zorder=4)

        def make_formatter(index):
            def formatter(x, pos):
                if abs(x - 0.0) < 1e-5:
                    return '' if index > 0 else '0.0'
                elif abs(x - 1.0) < 1e-5:
                    return '' if index < len(TILDE_EPS_LIST) - 1 else '1.0'
                return f'{x:.1f}'
            return formatter
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(make_formatter(i)))
        ax.text(0.95, 0.95, f'$\\tilde{{\\epsilon}} = {te:.1f}$', transform=ax.transAxes,
                ha='right', va='top', fontsize=13,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='lightgray', alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, top=0.86)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=True,
               fontsize=13.5, markerscale=1.2, handlelength=2.2)

    save_path = SCRIPT_DIR / "07_phase_portraits.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved PNG to: {save_path}")
    plt.close()

def main():
    print(f"Scaling factor (SI_ref): {scale_factor}")
    solutions, manual_flows = load_or_compute_data()
    generate_plots(solutions, manual_flows)

if __name__ == "__main__":
    main()
