# CA-AC-PINN v2.1: reproducible nonlinear/transient extension

## Code changes
- `local_radius.py`: local support scaling using prescribed initial-condition gradients, never final-time exact values.
- `evolution.py`: optional `local_strength` and `schur_min_eig` for Allen–Cahn, Burgers and transient diffusion.
- `run_v21_benchmark.py`: deterministic 3-case × 3-variant experiment with persisted JSON.
- `test_v21.py`: three new unit tests; existing tests retained.

## Experimental protocol
One BLAS thread. Allen–Cahn and Burgers use dt=0.002, transient diffusion dt=0.005. 64 coefficients for the 1D cases and 129 for transient diffusion. Uniform scaling 1.5 versus initial-condition-based strength 0.15 or 0.35. Results use manufactured solution for *evaluation only*. The comparison is coefficient-matched, not time-matched.

## Measured results

| Case | Variant | Relative L2 error | Runtime (s) | Coefficients |
|---|---|---:|---:|---:|
| allen_cahn | uniform | 0.000881945491 | 0.520 | 64 |
| allen_cahn | local_weak | 0.000848086755 | 0.484 | 64 |
| allen_cahn | local_strong | 0.000918231109 | 0.411 | 64 |
| burgers | uniform | 0.00168614191 | 0.744 | 64 |
| burgers | local_weak | 0.00192353657 | 0.854 | 64 |
| burgers | local_strong | 0.00202844285 | 0.749 | 64 |
| transient | uniform | 0.00023991555 | 9.237 | 129 |
| transient | local_weak | 0.000280148942 | 10.575 | 129 |
| transient | local_strong | 0.000357208013 | 10.109 | 129 |

## Interpretation
Local adaptation improved Allen–Cahn slightly at strength 0.15, but worsened Burgers and transient diffusion. The method is not demonstrated to dominate uniform scaling, classical solvers, or published adaptive neural solvers. No new comparison with AB-PINNs or sparse RBF methods has been executed. The original v2 elliptic outcomes remain unchanged.

## Reproduction
```
python -m unittest discover -p 'test*.py' -q
python verify.py
python verify_radius.py
python run_v21_benchmark.py --output results/v21_evolution_benchmark.json
```
No GitHub upload was performed.
