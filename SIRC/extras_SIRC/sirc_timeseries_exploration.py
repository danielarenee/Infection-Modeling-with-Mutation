import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add SIRC parent folder to path so it can find sirc_utils
sys.path.append(str(Path(__file__).parent.parent))

from sirc_utils import solve, sirc, calculate_period

# Fixed Parameters
mu    = 0.02       
alpha = 365/3      
delta = 1/1.61     
gamma = 0.35       
sigma = 0.07874    

# chaos parameters
eps = 0.25
beta0 = 400

params = {
    'mu': mu, 'alpha': alpha, 'delta': delta, 'gamma': gamma,
    'sigma': sigma, 'beta0': beta0, 'eps': eps,
}

# Define multiple initial conditions (S, I, R, C)
y0_list = [
    np.array([0.20, 0.001, 0.499, 0.30]), # Baseline
    np.array([0.10, 0.050, 0.600, 0.25]), # High initial infection
    np.array([0.40, 0.005, 0.200, 0.395]) # High susceptibility
]

# Simulation settings
h = 1/365
sim_years = 200 # Run long enough to clear transients
plot_years = 50 # Only plot the last 50 years

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#3498DB', '#E74C3C', '#2ECC71']

for idx, y0 in enumerate(y0_list):
    # 1. Integrate system
    t_arr, y_arr = solve(sirc, y0, (0, sim_years), h, params)
    
    # 2. Calculate period from the last 50 years of I(t)
    I_ts = y_arr[:, 1]
    period = calculate_period(I_ts)
    period_label = f"Period {period}" if period < 8 else "Chaos (8+)"
    
    print(f"Initial Condition {idx+1} settled into: {period_label}")
    
    # 3. Slice the arrays to only plot the last `plot_years`
    steps_to_plot = plot_years * 365
    t_plot = t_arr[-steps_to_plot:]
    I_plot = I_ts[-steps_to_plot:]
    
    # Plot
    label = f'IC {idx+1} -> {period_label}'
    ax.plot(t_plot, I_plot, color=colors[idx % len(colors)], linewidth=1.5, alpha=0.8, label=label)

ax.set_xlabel('Time (years)', fontsize=12)
ax.set_ylabel('Prevalence I(t)', fontsize=12)
ax.set_title(f'SIRC Last 10 Years ($\\beta_0$ = {beta0}, $\\epsilon$ = {eps})', fontsize=13)

ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / "sirc_timeseries_exploration.png", dpi=150)
plt.show()
plt.close()
