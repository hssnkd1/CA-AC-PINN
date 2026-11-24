# CA-AC-PINN v3.0 — validation and audit

## Implemented changes

1. `v3_select.py`: assemble and fit candidate radius policies; calculate physical Ritz energy on an independent quadrature; reject candidate selected matrices with scaled condition above 2e6; prefer uniform reference if no resolvable improvement. No exact solution used in the radius decision.
2. `run_v3.py`: auditable JSON experiments with actual per-candidate timings, error metrics evaluated only after the energy-based selection and selected matrix conditioning.
3. `test_v3.py`: energy-zero, energy-sign, no-exact-use-at-selection (with manufactured forcing provided independently), and stability rejection tests.
4. `plot_v3.py`: draws results from recorded JSON.
5. `CA_AC_PINN_Manuscript_v3.docx`: adds Section 8.12 and Table 15 while preserving previously embedded equations and earlier experimental results.

## Completed benchmarks

Recorded in `results/v3_elliptic.json`: three cases with 65 coefficients and quadrature subdivisions train/check/eval=20/28/40.

| Case | Uniform 1.5 relative L2 | Selected policy | Selected relative L2 | Selected scaled condition |
|---|---:|---|---:|---:|
| Poisson | 8.746e-05 | local strength 0.15 | 8.164e-05 | 6.3346e4 |
| Interface contrast 10 | 3.8174e-02 | local strength 0.15 | 2.7870e-02 | 6.9895e4 |
| Interface contrast 5 | 3.3716e-02 | local strength 0.15 | 3.0849e-02 | 6.6088e4 |

Recorded in `results/v3_elliptic_129.json`: two cases with 129 coefficients; train/check/eval=32/48/64.

| Case | Uniform 1.5 relative L2 | Selected policy | Selected relative L2 | Selected scaled condition |
|---|---:|---|---:|---:|
| Poisson | 2.6398e-05 | local strength 0.15 | 2.5963e-05 | 9.4815e5 |
| Interface contrast 10 | 7.77795e-03 | uniform 1.5 | 7.77795e-03 | 3.0448e5 |

**Unfinished:** the 129-coefficient contrast-5 benchmark was interrupted by the execution timeout; there is no completed result for it in the JSON. No v3 guard evaluation was completed on Allen–Cahn, Burgers or transient diffusion; previous v2.1 records remain unmodified.

## Software checks

- `python -m unittest verify_radius test_energy_condition_selection test_ca_v2 test_v21 test_v3 -q`: **18 passing tests**.
- `python verify.py -q`: **7 passing original tests**.
- These are software-level checks, not independent verification of all prior experimental claims.

## Limits and scientific interpretation

- Candidate fitting is performed four times per problem, so full selection cost is greater than a single uniform solve. The 65-coefficient candidate wall-time totals (approximately 4 s per case) are machine-dependent; they are not evidence of faster numerical solution.
- The selected radius policy is not guaranteed to minimize L2 error; the continuous Ritz energy identity concerns the energy seminorm, and quadrature checks are not rigorous interval enclosures.
- The selector may fall back to a uniform policy; this is an intentional outcome, not an omitted failure.
- Policy strengths and the condition cap were developed on related manufactured examples; held-out families and equal-time/memory benchmarks against AB-PINNs and sparse RBFs remain necessary.
- Full sparse direct solve is not implemented; the previous CSR assembly prototype uses a dense Ritz solution.
- The release has not been uploaded to GitHub, and no DOI has been assigned.
