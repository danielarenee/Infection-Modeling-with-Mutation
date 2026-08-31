# Compartmental Epidemiological Models with Infection-Driven Viral Mutation

This repository contains the simulation, bifurcation analysis, numerical continuation, asymptotic series approximations, and plotting code 
---

## Mathematical Models & Organization

The codebase is organized around the two variants of the SIRCm model:

1. **Prevalence-Driven variant ($\mu(t) = 1 + \varepsilon I(t)$)** — [`SIRCm_prevalence/`](SIRCm_prevalence/):  
   Viral mutation supply and immune escape scale with the fraction of actively infected individuals $I(t)$.
2. **Transmission-Driven variant ($\mu(t) = 1 + \varepsilon S(t)I(t)$)** — [`SIRCm_transmission/`](SIRCm_transmission/):  
   Mutation escape events scale with transmission encounters $S(t)I(t)$.
3. **Time Series Dynamics** — [`time_series/`](time_series/):  
   Time series comparisons and seasonally forced dynamics across temperate ($\beta_0 = 1200, \eta = 0.07$) and tropical ($\beta_0 = 400, \eta = 0.18$) regimes.
4. **Asymptotic Series Approximations** — [`series_approximations/`](series_approximations/):  
   Taylor series expansions for small $\varepsilon \to 0$ and perturbation expansions for large $\varepsilon \to \infty$ supporting both model variants.

---

## Figure Mapping

Scripts are organized in numbered folders corresponding to the figures in the paper:

| Fig. | Variant / Topic | Directory | Script | Output |
|:---:|:---|:---|:---|:---|
| **02** | Transmission | [`SIRCm_transmission/02_endemic_prevalence_surface/`](SIRCm_transmission/02_endemic_prevalence_surface/) | `endemic_prevalence_surface.py` | `02_endemic_prevalence_surface.png` |
| **03** | Prevalence | [`SIRCm_prevalence/03_hopf_surface_two_eps/`](SIRCm_prevalence/03_hopf_surface_two_eps/) | `hopf_surface_two_eps.py` | `03_hopf_surface_two_eps.png` |
| **04** | Transmission | [`SIRCm_transmission/04_hopf_surface_two_eps/`](SIRCm_transmission/04_hopf_surface_two_eps/) | `hopf_surface_two_eps.py` | `04_hopf_surface_two_eps.png` |
| **05** | Transmission | [`SIRCm_transmission/05_eigenvalues_exploration/`](SIRCm_transmission/05_eigenvalues_exploration/) | `eigenvalues_exploration.py` | `05_eigenvalues_exploration.png` |
| **06** | Transmission | [`SIRCm_transmission/06_07_trajectories/06_3d_trajectories/`](SIRCm_transmission/06_07_trajectories/06_3d_trajectories/) | `3d_trajectories.py` | `06_3d_trajectories.png` |
| **07** | Transmission | [`SIRCm_transmission/06_07_trajectories/07_phase_portraits/`](SIRCm_transmission/06_07_trajectories/07_phase_portraits/) | `phase_portraits.py` | `07_phase_portraits.png` |
| **08** | Prevalence | [`SIRCm_prevalence/08_09_trajectories/08_3d_trajectories/`](SIRCm_prevalence/08_09_trajectories/08_3d_trajectories/) | `3d_trajectories.py` | `08_3d_trajectories.png` |
| **09** | Prevalence | [`SIRCm_prevalence/08_09_trajectories/09_phase_portraits/`](SIRCm_prevalence/08_09_trajectories/09_phase_portraits/) | `phase_portraits.py` | `09_phase_portraits.png` |
| **10** | Transmission | [`SIRCm_transmission/10_endemic_prevalence/`](SIRCm_transmission/10_endemic_prevalence/) | `endemic_prevalence.jl` | `10_endemic_prevalence.png` |
| **11** | Prevalence | [`SIRCm_prevalence/11_endemic_prevalence/`](SIRCm_prevalence/11_endemic_prevalence/) | `endemic_prevalence.py` | `11_endemic_prevalence.png` |
| **12** | Transmission | [`SIRCm_transmission/12_hopf_surface_one_eps/`](SIRCm_transmission/12_hopf_surface_one_eps/) | `hopf_surface_one_eps.py` | `12_hopf_surface_one_eps.png` |
| **13, 14** | Prevalence / Unforced | [`time_series/13_14_SIRCm_prevalence/`](time_series/13_14_SIRCm_prevalence/) | `13_14_plot_timeseries_unforced_panels.py` | `timeseries_comparison_three_panels.png` |
| **15, 17, 18, 19, 20** | Seasonal Forcing | [`time_series/seasonal_forcing/`](time_series/seasonal_forcing/) | `plot_timeseries_panels.py` | `sircmwI_vs_sirc_comparison.png` |
| **16** | Prevalence | [`SIRCm_prevalence/16_endemic_prevalence_surface/`](SIRCm_prevalence/16_endemic_prevalence_surface/) | `endemic_prevalence_surface.py` | `16_endemic_prevalence_surface.png` |

---

## Shared Utility Modules

Core ODE vector fields, analytical Jacobians, polynomial reductions, and algebraic solvers are encapsulated in modular utility files:

- **Python Utilities**:
  - [`SIRCm_prevalence/sircmw_I_utils.py`](SIRCm_prevalence/sircmw_I_utils.py): Prevalence-driven ODE vector fields, Jacobian, polynomial solvers, and reseeding integrators.
  - [`SIRCm_transmission/sircmw_utils.py`](SIRCm_transmission/sircmw_utils.py): Transmission-driven ODE vector fields, Jacobian, characteristic polynomial solvers, and reseeding integrators.
- **Julia Utilities**:
  - [`SIRCm_prevalence/sircmw_utils.jl`](SIRCm_prevalence/sircmw_utils.jl): In-place vector fields and polynomial solvers for prevalence-driven continuation.
  - [`SIRCm_transmission/sircmw_utils.jl`](SIRCm_transmission/sircmw_utils.jl): In-place vector fields and polynomial solvers for transmission-driven continuation.

---

## Supplementary Notebooks & Analytical Files

- **Interactive Jupyter Notebooks (`.ipynb`)**:
  - [`SIRCm_prevalence/08_09_trajectories/08_3d_trajectories/3d_trajectories_notebook.ipynb`](SIRCm_prevalence/08_09_trajectories/08_3d_trajectories/3d_trajectories_notebook.ipynb)
  - [`SIRCm_transmission/06_07_trajectories/06_3d_trajectories/3d_trajectories_notebook.ipynb`](SIRCm_transmission/06_07_trajectories/06_3d_trajectories/3d_trajectories_notebook.ipynb)

- **Mathematica Analytical Notebooks (`.nb`)**:
  - [`SIRCm_prevalence/sircm_prevalence_equilibria.nb`](SIRCm_prevalence/sircm_prevalence_equilibria.nb): Symbolic reduction and equilibrium algebra for the prevalence variant.
  - [`SIRCm_transmission/sircm_transmission_equilibria.nb`](SIRCm_transmission/sircm_transmission_equilibria.nb): Symbolic reduction and equilibrium algebra for the transmission variant.

---

## Getting Started

### Python Environment
```bash
pip install numpy scipy matplotlib pandas sympy
```

### Julia Environment (for continuation figures)
```julia
using Pkg
Pkg.add(["Plots", "LinearAlgebra", "BifurcationKit", "DelimitedFiles"])
```

### Reproducing Figures
All scripts use relative paths and can be run directly from any directory:
```bash
# Example: Generate Figure 02
python SIRCm_transmission/02_endemic_prevalence_surface/endemic_prevalence_surface.py

```
