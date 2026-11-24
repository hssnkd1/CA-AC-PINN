# Adaptive circular physics-informed PDE study

This package contains the full implementation and measured data accompanying
the manuscript. The method is a compact radial Ritz network with a dyadic
dictionary, exact block residual scores for the quadrature-defined elliptic
problem, and energy-guarded pruning. It is not a new activation function.

## Reproduction

Use Python 3.11 or newer in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify.py
python run_study.py --part all --iterations 1500 --seeds 3
python make_figures.py
```

On Windows activate the environment with `.venv\Scripts\activate`.
No GPU or machine learning framework is required. Run the study in one process
and do not run independent benchmark processes concurrently if comparing times.
BLAS is restricted to one thread by the study entry point.

## Files

* `acpinn.py`: circular features, quadrature assembly, nested block selection,
  energy-guarded pruning and exact-solution error integration.
* `problems.py`: manufactured elliptic and transient diffusion problems.
* `classical.py`: independent triangular P1 FEM and rectangular RT0 flux recovery.
* `pinn_baseline.py`: a fully trained 32-unit tanh PINN, RAR variant and a finite
  difference gradient-enhanced PINN variant, with analytic parameter Jacobians.
* `evolution.py`: backward Euler/Newton networks and independent FDM calculations
  for moving Allen-Cahn and Burgers fronts and heterogeneous transient diffusion.
* `verify.py`: focused derivative, boundary, interface, source and energy checks.
* `run_study.py`: the complete experiment driver.
* `results/`: JSON results, saved network coefficients and reproducibility metadata.
* `figures/`: publication-scale figures generated from saved results.
* `manuscript.md`: editable manuscript source with LaTeX mathematical notation.

## Scope and limits

The elliptic energy identities are exact mathematical results. The implementation
uses floating-point Gaussian quadrature. Reported functional majorants are
quadrature-evaluated estimates, not interval-certified numerical bounds.
The nonlinear tests do not establish convergence of the adaptive algorithm.
Geometry is selected from a discrete hierarchy; it is not optimized continuously.
The flat-interface coordinate mapping uses the known diffusion coefficient.
No automatic differentiation library or GPU experiment is hidden in the results.

To rebuild the Word manuscript from its template and saved results, install the
Pandoc command line program and run `python compile_manuscript.py`. This creates
native Word equations. LibreOffice is only needed for independent rendering and
visual review of the resulting DOCX.

The neural baselines have 129 trained scalar parameters. The circular networks
with 129 coefficients additionally store 363 geometry scalars. The study is
coefficient-matched and reports actual time; it is not a comprehensive equal-time,
total-storage-matched or architecture-tuned comparison. P1 FEM is uniform, not
adaptive FEM. Three neural seeds provide descriptive variation, not a powered
statistical study. Physical observations were not used: the tests verify numerical
solutions against manufactured or exact fields.

Authorship, affiliations, funding and competing-interest declarations must be
supplied and confirmed by the authors before submission. The closest 2026 radial
and adaptive-basis methods should receive a direct independent implementation
comparison before any claim of a new state of the art.

## Supplementary radius-sensitivity study (revision)

The original tables and benchmark runs are preserved and **not overwritten**.
A separate sensitivity experiment changes all Wendland radii by a single
multiplicative factor while reselecting 129 active features. For reproducibility:

```bash
python verify_radius.py
python radius_sensitivity.py --scales 1.0 1.2 --cases poisson interface
python radius_sensitivity.py --scales 1.5 --cases interface --output results/radius_interface_1p5.json
```

The independent results in `results/radius_sensitivity.json` and
`results/radius_interface_1p5.json` support Table 9 of the revised manuscript.
The radius increase improves the reported L2 errors but also increases the
selected-matrix condition number. No general accuracy or speed advantage is
claimed beyond the tested configurations. Original run timings remain from the
original environment; comparisons between machines should not be interpreted
as a fair speed experiment. The modified Word manuscript is supplied at the
repository root as `AC_PINN_Manuscript_Revised.docx`.

A public GitHub URL will be added only after publishing and verifying a tag.

## v1.2 executed evolutionary benchmark extension

Run `python evolution_radius_benchmark.py` to repeat the matched-coefficient radius sweep, and `python evolution_refinement_v12.py` to reproduce the full time-step grid at scale 1.5. The manuscript reports the results in Section 8.8 and Table 10. Original outputs are preserved. For experimental limits see BENCHMARK_VALIDATION_v1_2.md.

## Exploratory v1.3 extension (not included in original v1.2 tables)

```bash
OPENBLAS_NUM_THREADS=1 python energy_condition_selection.py
OPENBLAS_NUM_THREADS=1 python -m unittest test_energy_condition_selection.py
```

The additional files `results/extended_radius_sweep_v13.json` and
`results/energy_radius_selection_v13.json` preserve the newly executed studies.
See `VALIDATION_AND_DEVELOPMENT_v1_3.md` for interpretation and limitations.
