# AC-PINN v1.3: additional executed analysis and next-stage development

## Executed experiments

- Three nonlinear/transient studies: increased radius sweep (1.4, 1.6, 1.8, 2.0 where appropriate) at the published coefficient budgets and time steps. Source: `results/extended_radius_sweep_v13.json`.
- New **PDE-only** scale-selection procedure for the symmetric elliptic cases. Criterion: minimize assembled Ritz energy among candidates with scaled stiffness condition <= 2e6. Source: `energy_condition_selection.py`, `results/energy_condition_selection_v13.json`.
- Repeated 48-subdivision ellipse experiments with independently evaluated 80-subdivision L2/energy errors, saved in `results/energy_radius_selection_v13.json`.
- Tests: existing 7 checks, existing 3 radius checks, 3 new selection-unit checks all passed locally.

## Numerical findings (single runs; not a blind external benchmark)

| Case | New scale | Relative L2 | Error at scale 1.5 (v1.2) | FDM at same time step (v1.2) |
|---|---:|---:|---:|---:|
| Allen–Cahn | 1.4 | 0.000861816 | 0.000881945 | 0.000771077 |
| Burgers | 2.0 | 0.001503442 | 0.001686142 | 0.001252869 |
| Transient | 2.0 | 0.000195793 | 0.000239916 | 0.000335300 |

- New-radius time measurements vary by machine and are not an equal-runtime benchmark. The transient adaptive model still takes seconds; FDM is faster.
- For both elliptic examples, the guard cap 2e6 excluded scale 1.8 and selected 1.5 based entirely on Ritz energy and conditioning. In the 48-subdivision pilot, the **interface** relative L2 improved from 0.0093893 (1.0) to 0.0028112 (1.5), and **Poisson** from 6.8903e-05 to 8.9291e-06.
- Choosing the guard from these same examples creates hyperparameter selection bias. This is a pilot, not out-of-distribution validation.

## Next rigorous method improvement

1. Specify a dimensionless, problem-normalized stability criterion and preset it before evaluating held-out problems.
2. Select radii through residual-energy estimates computed on a higher-order **training-only** quadrature, with a separate final evaluation rule.
3. Add local per-level scale choice (instead of one scale for the whole dictionary), with low-rank matrix updates and feature-neighbor search.
4. Repeat at multiple front widths, velocities, interface contrasts, and spatially changing transient shapes.
5. Evaluate equal total stored scalars, end-to-end time including selection, and fixed wall-clock budgets; run the published AB-PINN and sparse-RBF code before any state-of-the-art claim.

No such future experiments should be reported as already performed.
