# CA-AC-PINN v2 implementation and experimental audit

## Verified implementation
- PDE-forcing-gradient-based locally varied Wendland radii (no exact errors used for selection).
- Locally conditioned Schur-complement block selection.
- CSR sparse stiffness/mass quadrature assembly (dense downstream solve remains).
- New baseline/variant runner and 5 new automated checks.
- 18/18 tests pass when legacy and new tests are combined.

## Pilot benchmark protocol
Single CPU BLAS thread; 2D Poisson and discontinuous-diffusion interface at contrasts 10 and 5.
65 coefficient budget, training 24 subdivisions and independent evaluation 48 subdivisions; partial 129-coefficient benchmark with training 32 and evaluation 64.
All timings measure end-to-end construction of each selected model, but **not** fair equal-wall-clock tuning.

## Results: 129 coefficients
| Case | Method | Relative L2 | Scaled condition | Construction seconds |
|---|---|---:|---:|---:|
| poisson | original | 0.0001008486 | 2806 | 2.260 |
| poisson | uniform_1p5 | 2.639751e-05 | 1.1984e+06 | 2.586 |
| poisson | local_weak | 2.596262e-05 | 9.4815e+05 | 2.480 |
| interface | original | 0.01428764 | 3211 | 2.083 |
| interface | uniform_1p5 | 0.007777948 | 3.0448e+05 | 2.379 |
| interface | local_weak | 0.008070116 | 7.6169e+05 | 2.660 |

## Results: 65 coefficients
| Case | Method | Relative L2 | Scaled condition | Construction seconds |
|---|---|---:|---:|---:|
| poisson | original | 0.0001447225 | 1756 | 1.493 |
| poisson | uniform_1p5 | 4.664617e-05 | 50814 | 1.516 |
| poisson | local_weak | 4.629874e-05 | 37603 | 1.555 |
| interface | original | 0.08034494 | 1374.3 | 1.409 |
| interface | uniform_1p5 | 0.03420342 | 42228 | 1.733 |
| interface | local_weak | 0.03914188 | 74065 | 1.709 |
| interface_contrast_5 | original | 0.01523621 | 1233.8 | 1.594 |
| interface_contrast_5 | uniform_1p5 | 0.008085224 | 38186 | 1.639 |
| interface_contrast_5 | local_weak | 0.007900623 | 68729 | 1.691 |

## Limitations
- Local radii do not consistently beat the tuned uniform-radius model; conditioning often worsens versus unscaled baseline.
- Comparisons are coefficient-matched, not equal stored parameters, tuning budgets, or wall-clock.
- New radius algorithm has NOT been applied to nonlinear or transient equations.
- CSR assembly equivalence is tested, but no sparse end-to-end runtime speedup is established.
- No execution of published AB-PINN or sparse RBF baselines; no SOTA superiority is claimed.
- The paper retains earlier experiments as separate historical v1.3 data.
- No GitHub publication has been made.
