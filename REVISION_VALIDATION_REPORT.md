# AC-PINN revised research package — verified change log

## What was executed
- Original `python verify.py`: **7 passed**.
- New `python verify_radius.py`: **3 passed**.
- Repeated the original-scale and widened-radius elliptic experiments in CPU double precision, 64 subdivisions for assembly and independent 80 subdivisions for error assessment.
- Recomputed coefficients and structural selection at each scale, with 129 active coefficients.
- Recorded five cases in `results/radius_sensitivity.json` and `results/radius_interface_1p5.json`.
- Rendered the revised manuscript as 25 pages to inspect formatting, including the new Table 9.

## Measured radius experiment
| Case | Radius multiplier | Relative L2 | Relative energy | Scaled selected-system condition |
|---|---:|---:|---:|---:|
| Poisson | 1.0 | 6.9190e-5 | 7.2698e-4 | 5.0669e3 |
| Poisson | 1.2 | 2.2263e-5 | 2.7311e-4 | 5.4526e4 |
| Interface | 1.0 | 1.00023e-2 | 5.71858e-2 | 4.0401e3 |
| Interface | 1.2 | 5.73707e-3 | 3.29354e-2 | 1.7875e4 |
| Interface | 1.5 | 3.01574e-3 | 2.07672e-2 | 3.7390e5 |

The improvement is **accuracy**, not proven stability or compute speed. Condition numbers increase strongly with radius, so do not assert dominance on all metrics. The radius values were inspected after looking at the test cases; these are exploratory, not an independent hold-out validation of tuned hyperparameters.

## Manuscript edits
- Language repairs in Introduction, Algorithm 1, Section 9.1, and Conclusions.
- New Section 8.7 / Table 9 summarizes the executed sensitivity experiment and the accuracy–conditioning tradeoff.
- Original main benchmark tables and figure records were preserved; no numbers were altered to create favorable comparisons.
- No public GitHub claim or fictitious URL was inserted. Existing code-availability statement accurately refers to accompanying supplementary material.
- Original Word math objects, major figures and existing reference list were retained.

## Important remaining work before claiming publication-ready superiority
1. Repeat full primary benchmarks with consistent improved support-radius policies across elliptic and evolution cases (including all three tanh seeds), and verify reference problems on new test settings. Existing evolution results are unchanged.
2. Evaluate radius choice with **held-out PDE families, noise/perturbations, conditioning thresholds, and repeated timing runs**. The best radius must not be chosen using test-set error if the protocol is intended to be predictive.
3. Compare at equal time and total stored model parameters with tuned adaptive FEM, AB-PINN and sparse-RBF rivals. The original 129-coefficient comparisons do not control total storage.
4. Add robust regularization/pivot safeguards if supporting radii are increased further and prove any new mathematical property before claiming it.
5. Independently check theorem hypotheses, numerical derivatives, full paper citation metadata, funding/author declarations and journal template.
6. Add actual GitHub link and tag to the Word paper **after** final verified release, not before.

## Distinction between old and new results
`results/poisson.json`, `results/interface.json`, other baseline/evolution JSON and saved NPZ are the **original** study. The **new** radius sensitivity records are separate and the manuscript labels them as a supplementary controlled experiment. The full benchmark suite was not rerun after changes.
