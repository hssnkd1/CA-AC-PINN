# AC-PINN v1.2 executed evolutionary benchmark audit

## Executed
- Identical manufactured cases from v1.1, in double precision on CPU with BLAS capped to one thread.
- Sensitivity runs at radius scales 1.0, 1.2, 1.5; both adaptive and fixed networks widened consistently. Front dt = 0.002; transient dt = 0.005.
- Full time-step sensitivity at radius scale 1.5: front dt 0.004, 0.002, 0.001 and transient dt 0.01, 0.005, 0.0025.
- All 27 refinement-grid measurements including FDM comparisons recorded in results/evolution_refinement_v12.json.
- 7 original verification tests and 3 radius verification tests passed.

## Main comparison at matched timestep (L2 relative error)
| Case | Adaptive original radius 1.0 | Adaptive radius 1.5 | Fixed radius 1.5 | FDM | Adaptive s (radius 1.5) | FDM s |
|---|---:|---:|---:|---:|---:|---:|
| Allen-Cahn dt .002 | .001311298 | .000881945 | .001705511 | .000771077 | .217 | .038 |
| Burgers dt .002 | .002417561 | .001686142 | .002887063 | .001252869 | .342 | .095 |
| Transient dt .005 | .000913868 | .000239916 | .008545841 | .000335300 | 6.371 | .033 |

## Limits
- These are **exploratory on-test radius sweeps**: selecting 1.5 using evaluation errors creates a selection bias. Do not claim validated default generalization.
- The time comparisons are *reported*, but there was no equal-wallclock-budget model tuning or repeated independent timing samples. FDM is substantially faster in these executions.
- Equal coefficient count does **not** mean equal full parameter/storage budget. One-dimensional circle nets store ~186 scalars (64 output coefficients + 61 center/radius pairs). Two-dimensional circle nets store ~492 scalars (129 coefficients + 121 triples).
- Actual AB-PINN and Sparse RBF reference implementations were not run on matched PDE data: no valid superiority conclusion over these methods.
- Test code is a numerical research prototype, not a proof of stability or mesh-independent convergence.
- No GitHub publication was performed.

## Contemporary baseline source code
- AB-PINNs: https://github.com/merlresearch/ab-pinns (external code, not run).
- Sparse RBF networks: https://github.com/EddyShao/SparseKernelPDE (external code, not run).

## Reproduce
```bash
python verify.py
python verify_radius.py
python evolution_radius_benchmark.py
python evolution_refinement_v12.py
```
