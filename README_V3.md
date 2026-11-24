# CA-AC-PINN v3.0 — guarded radius policy experiment

This complete version preserves the v2.1 reproducibility source and extends **symmetric 2D elliptic** cases with an independent-quadrature energy-based selector and an explicit conditioning guard. This is an **exploratory implementation**, not a general nonlinear radius optimizer, certified bound, or equal-time state-of-the-art comparison.

## What's new

- `v3_select.py`: candidate dictionary fit, independent quadrature of physical Ritz energy, selected-system condition cap, uniform fallback for indistinguishable energies. No access to an exact field during radius selection.
- `run_v3.py`: deterministic experiments at fixed coefficient budgets, independent quadrature and JSON outputs.
- `test_v3.py`: selection and energy sanity checks.
- `results/v3_elliptic.json`: three 65-coefficient cases completed with training/check/evaluation subdivisions 20/28/40.
- `results/v3_elliptic_129.json`: two completed 129-coefficient cases with 32/48/64 subdivisions; contrast=5 case did not complete during execution and is excluded.
- `CA_AC_PINN_Manuscript_v3.docx`: retains previous equation objects and introduces Section 8.12 and Table 15.

## Reproduction

In the `project` directory, install dependencies listed in `requirements.txt`; for robust runtime comparisons limit BLAS to 1 thread:

```bash
python -m unittest verify_radius test_energy_condition_selection test_ca_v2 test_v21 test_v3 -q
python verify.py -q
OPENBLAS_NUM_THREADS=1 python run_v3.py --train-n 20 --check-n 28 --evaluation-n 40 --budget 65 --output results/v3_elliptic.json
```

The 129-coefficient run is computationally expensive:

```bash
OPENBLAS_NUM_THREADS=1 python run_v3.py --train-n 32 --check-n 48 --evaluation-n 64 --budget 129 --output results/v3_elliptic_129.json
```

`run_v3.py` writes after each completed problem so interrupted runs retain previously completed cases. Time includes full candidate fitting and screening; it is not directly comparable to one-shot uniform fits without accounting for repeated candidate solves. If you rerun, keep environment and runtime measurements together.

## Scientific scope

The identity J(v)-J(u)=0.5||u-v||_a² applies to the continuous symmetric diffusion energy. Our checked energy uses quadrature and is **not a certified upper bound**. The selector can lower energy and still worsen the measured L2 error. The physical forcing is derived from manufactured exact solutions to define the test problem; the selection algorithm only calls `forcing` and `coefficient`, not the manufactured solution evaluator. Conditions tested are not evidence of generalization to unseen forcing patterns. v2.1 Allen–Cahn, Burgers and transient results are included unchanged and are not v3 results.

No GitHub upload has occurred. No published DOI exists for this package.
