"""
Generates 3D trajectory plots (S, I, R axes) of the SIRCmw transmission model
for multiple values of tilde_eps.

Extracted from sircmw_3d_trajectory.ipynb.
Equivalent to SIRCm_prevalence/equilibria_and_stability/plotting/plot_3d_trajectories.py
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add SIRCm_transmission root to path
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

# Scaling factor: SI_ref = S0_ref * I0_ref (as used in the original notebook)
S0_ref, I0_ref = 0.2, 0.001
scale_factor = S0_ref * I0_ref  # = 0.0002

TILDE_EPS_LIST = [0.5, 1.5, 1.7]

# Initial conditions for each tilde_eps value
Y0_LIST = [
    np.array([0.2, 0.001, 0.499, 0.3]),  # for tilde_eps = 0.5
    np.array([0.2, 0.001, 0.499, 0.3]),  # for tilde_eps = 1.5
    np.array([0.2, 0.001, 0.499, 0.3])   # for tilde_eps = 1.7
]

T_SPAN = (0.0, 100.0)
OUTPUT_FILENAME = "06_3d_trajectories.png"

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
        method='Radau', rtol=1e-6, atol=1e-9
    )
    return t_arr, Y_arr, n_ev

def main():
    print(f"Running simulations for tilde_eps: {TILDE_EPS_LIST}...")
    solutions = {}
    for te, y0 in zip(TILDE_EPS_LIST, Y0_LIST):
        t_arr, Y_arr, n_ev = run_simulation(te, y0, T_SPAN)
        solutions[te] = Y_arr
        print(f"  tilde_eps = {te} done. Reseeding events: {n_ev}")

    fig = plt.figure(figsize=(18, 6.0))
    axes = []

    for i, te in enumerate(TILDE_EPS_LIST):
        ax = fig.add_subplot(1, len(TILDE_EPS_LIST), i + 1, projection='3d')
        axes.append(ax)
        S, I, R, C = solutions[te]

        ax.plot(S, I, R, color='tab:blue', lw=1.5, alpha=0.85, label='Trajectory' if i == 0 else None)
        ax.scatter([S[0]], [I[0]], [R[0]], color='green', s=60, edgecolor='k', zorder=6, label='Initial condition' if i == 0 else None)

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
        for eq_idx, eq in enumerate(eqs):
            S_eq, I_eq, R_eq, _ = eq
            lbl = 'Equilibrium' if (i == 0 and eq_idx == 0) else None
            ax.scatter([S_eq], [I_eq], [R_eq], color='red', marker='*', s=140, edgecolor='k', zorder=8, label=lbl)

        ax.set_xlabel('S (Susceptible)', labelpad=12)
        ax.set_ylabel('I (Infected)', labelpad=12)
        ax.set_zlabel('R (Recovered)', labelpad=12)

        S_range = S.max() - S.min()
        I_range = I.max() - I.min()
        R_range = R.max() - R.min()
        S_pad = 0.05 * S_range if S_range > 1e-6 else 0.05
        I_pad = 0.05 * I_range if I_range > 1e-6 else 0.05
        R_pad = 0.05 * R_range if R_range > 1e-6 else 0.05

        ax.set_xlim(S.min() - S_pad, S.max() + S_pad)
        ax.set_ylim(I.min() - I_pad, I.max() + I_pad)
        ax.set_zlim(R.min() - R_pad, R.max() + R_pad)

        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.zaxis.set_major_locator(plt.MaxNLocator(4))

        ax.view_init(elev=22, azim=-45)

        ax.text2D(0.95, 0.95, f'$\\tilde{{\\epsilon}} = {te}$', transform=ax.transAxes,
                  ha='right', va='top', fontsize=13,
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='lightgray', alpha=0.9))

        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35, left=0.05, right=0.95, top=0.85, bottom=0.1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=True, fontsize=13.5, markerscale=1.2, handlelength=2.2)

    fig_save_path_png = SCRIPT_DIR / OUTPUT_FILENAME
    plt.savefig(fig_save_path_png, dpi=300, bbox_inches='tight')
    print(f"Saved PNG to: {fig_save_path_png}")
    plt.close()

if __name__ == "__main__":
    main()
