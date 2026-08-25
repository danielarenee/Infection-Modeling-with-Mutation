"""
Tracks the real and imaginary components of the eigenvalues of the Jacobian 
 of the SIRCmw model at the endemic equilibrium for a range of tilde_eps
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp

sys.path.append(str(Path(__file__).parent.parent))

from sircmw_utils import (
    sircmw,
    sircmw_jacobian,
    MU as mu,
    ALPHA as alpha,
    DELTA as delta,
    GAMMA as gamma,
    SIGMA as sigma,
    BETA0 as beta0,
    SI_0,
    infection_eigvec,
    integrate_with_reseeding,
    get_algebraic_equilibria
)

# sweep to find polynomium equilibria
tilde_eps_vals = np.linspace(0, 2, 5000)
rows = []

for te in tilde_eps_vals:
    eqs = get_algebraic_equilibria(te)
    for eq in eqs:
        rows.append((te, *eq))

data    = np.array(rows)
te_plot = data[:, 0]
n_eq    = len(data)

# eigenvalues
eigvals = np.empty((n_eq, 4), dtype=complex)

for i, row in enumerate(data):
    te  = row[0]
    eq  = row[1:]
    eps = te / SI_0
    J   = sircmw_jacobian(eq, eps)
    eigs = np.linalg.eigvals(J)
    idx  = np.lexsort((-eigs.imag, -eigs.real))
    eigvals[i] = eigs[idx]

# plot (full picture)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
fig1, ax1 = plt.subplots(figsize=(11, 6))
ax1.axhline(0, color='gray', linewidth=0.8, linestyle='--', zorder=0)

for k in range(4):
    ax1.plot(te_plot, eigvals[:, k].real, color=colors[k], linestyle='-',  linewidth=1.5, label=fr'Re($\lambda_{k+1}$)')
    ax1.plot(te_plot, eigvals[:, k].imag, color=colors[k], linestyle=':', linewidth=1.5, label=fr'Im($\lambda_{k+1}$)')

ax1.set_xlabel(r'Common $\tilde{\varepsilon}$', fontsize=12)
ax1.set_ylabel('Eigenvalue component', fontsize=12)
ax1.set_title(fr'SIRCmw eigenvalues at endemic equilibrium ($\beta_0={beta0}$)', fontsize=13)
ax1.legend(fontsize=9, ncol=4)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "sircmw_eigenvalues_full.png", dpi=150)
plt.show()

#%%
# plot (zoomed)
fig2, ax2 = plt.subplots(figsize=(11, 5))
ax2.axhline(0, color='gray', linewidth=0.8, linestyle='--', zorder=0)

for k in range(4):
    ax2.plot(te_plot, eigvals[:, k].real, color=colors[k], linestyle='-',  linewidth=1.5, label=fr'Re($\lambda_{k+1}$)')
    ax2.plot(te_plot, eigvals[:, k].imag, color=colors[k], linestyle=':', linewidth=1.5, label=fr'Im($\lambda_{k+1}$)')

ax2.set_ylim(-0.1, 0.1)
ax2.set_xlabel(r'Common $\tilde{\varepsilon}$', fontsize=12)
ax2.set_ylabel('Eigenvalue component', fontsize=12)
ax2.set_title('zoom on y (from -0.1 to 0.1)', fontsize=12)
ax2.legend(fontsize=9, ncol=4)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "sircmw_eigenvalues_zoom.png", dpi=150)
plt.show()

# test simulations

# first we find E0, which is the tilde_eps where Re(λ₂) is maximum
re_lam2 = eigvals[:, 1].real
i0 = int(np.argmax(re_lam2))
E0 = te_plot[i0]
eps_E0  = E0 / SI_0
print(f"\nE0 = {E0:.4f}  →  eps = {eps_E0:.2f}   (Re(λ₂) max = {re_lam2[i0]:.6f})")

# get equilibrium XE0 at E0
XE0 = np.array(get_algebraic_equilibria(E0)[0], dtype=float)
print(f"XE0 = [S={XE0[0]:.6f}, I={XE0[1]:.6f}, R={XE0[2]:.6f}, C={XE0[3]:.6f}]")

#  eigenvectors at XE0 
J_E0 = sircmw_jacobian(XE0, eps_E0) # evaluate jacobian at xe0
vals_E0, vecs_E0  = np.linalg.eig(J_E0) # computes eigenvals and eigenvect
idx_E0 = np.lexsort((-vals_E0.imag, -vals_E0.real)) # sort by descending real part and then imaginary
vals_E0 = vals_E0[idx_E0]
vecs_E0 = vecs_E0[:, idx_E0]

print("\nEigenvalues at E0:")
for k, v in enumerate(vals_E0):
    print(f"  λ{k+1} = {v.real:+.4f}  {v.imag:+.4f}i")

# yellow = λ₂ (index 1),  red = λ₄ (index 3)
Vy_raw = vecs_E0[:, 1]
Vr_raw = vecs_E0[:, 3]

def normalize_real(v):
    """unit-normalize the real part of an eigenvector"""
    v_r = np.real(v)
    n = np.linalg.norm(v_r)
    return v_r / n

Vy = normalize_real(Vy_raw)
Vr = normalize_real(Vr_raw)

print(f"\nVy (yellow, normalized) = {Vy}")
print(f"Vr (red, normalized) = {Vr}")


# %%
