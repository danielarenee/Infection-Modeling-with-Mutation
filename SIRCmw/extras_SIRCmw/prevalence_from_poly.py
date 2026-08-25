import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory SIRCmw to path so it can find sircmw_utils
sys.path.append(str(Path(__file__).parent.parent))

from sircmw_utils import get_algebraic_equilibria

# sweep tilde_eps and collect equilibrium points
tilde_eps_vals = np.linspace(0, 2, 1000)
rows = [] # each entry: (tilde_eps, S*, I*, R*, C*)

for te in tilde_eps_vals:
    eqs = get_algebraic_equilibria(te)
    for eq in eqs:
        rows.append((te, *eq))

data = np.array(rows) 

# plot
fig, ax = plt.subplots(figsize=(9, 5))

for col_idx, (color, label) in enumerate(zip(
        ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'],
        ['S*', 'I*', 'R*', 'C*'])):
    ax.scatter(data[:, 0], data[:, 1 + col_idx], s=4, color=color, label=label)

ax.set_xlabel(r'$\tilde{\varepsilon}$', fontsize=12)
ax.set_ylabel('Equilibrium prevalence', fontsize=12)
ax.set_title('SIRCmw equilibrium (S*, I*, R*, C*) from polynomial roots', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%% query a single tilde_eps value 
te_query = 1.5
eqs_q = get_algebraic_equilibria(te_query)

print(f"\ntilde_eps = {te_query}")
for eq in eqs_q:
    S, I, R, C = eq
    print(f"  [S, I, R, C] = [{S:.6f}, {I:.6f}, {R:.6f}, {C:.6f}]  (sum = {S+I+R+C:.6f})")


# %%
