# CA-AC-PINN v2 — experimental elliptic development

This version extends the **original AC-PINN v1.3 project**, retaining its unchanged benchmark data and historical manuscripts. The new modules are `ca_acpinn.py`, `run_ca_v2.py`, and `test_ca_v2.py`.

## Requirements

Use Python 3.11+ and install from `requirements.txt`. The built-in experiments use NumPy/SciPy/Matplotlib and one BLAS thread for repeatable CPU benchmark conditions.

## Tests

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest -v verify verify_radius test_energy_condition_selection test_ca_v2
```

## Reproduce the new pilot

```bash
OPENBLAS_NUM_THREADS=1 python run_ca_v2.py --n 24 --evaluation 48 --budget 65 --output results/ca_v2_pilot_small.json
OPENBLAS_NUM_THREADS=1 python run_ca_v2.py --n 32 --evaluation 64 --budget 129 --output results/ca_v2_pilot_129_recheck.json
```

The second run computes 12 variants (the released `ca_v2_pilot_129.json` records the six actually re-executed rows for Poisson and contrast-10 interface). Timings are computer-dependent. The saved results should be preferred for the exact numbers included in the v2 paper. `build_v2_release.py` creates the manuscript from the separately supplied original and is **not** needed to execute experiments.

## Interpretation

- The new local-radius rule is calculated from forcing variation, not exact error.
- The guard rejects poorly conditioned candidate **blocks**, not globally ill-conditioned selected matrices.
- The CSR assembler matches the dense method but currently still evaluates the full candidate dictionary and does not speed up the dense downstream Ritz solve.
- The nonlinear and transient results in the inherited package were produced by the earlier global radius code, **not by the new v2 local-adaptation method**.
- These are exploratory single-run tests. No claim of superior equal-cost performance or state-of-the-art advantage is supported.
- No files have been uploaded to GitHub.
