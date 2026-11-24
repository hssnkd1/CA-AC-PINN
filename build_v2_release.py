from docx import Document
from docx.shared import Pt,Inches
import json,datetime
from pathlib import Path
P=Path(__file__).parent
source=Path('/mnt/data/AC_PINN_Manuscript_Experimental_v1_3.docx')
target=P/'AC_PINN_Manuscript_Development_v2.docx'
d=Document(source)
anchor=next(p for p in d.paragraphs if p.text.strip().startswith('9 Verification reproducibility'))
def para(text='',style=None):
    p=anchor.insert_paragraph_before(text,style=style)
    return p
para('8.10 Condition-aware local radii and sparse assembly: implemented pilot',style='Heading 2')
para('We extend the computational implementation with a forcing-gradient-based local-radius dictionary, an additional Schur-complement stability filter and sparse quadrature assembly. These are experimental numerical extensions, not a new proof of convergence or an extension of the elliptic energy identity to the nonlinear models.')
para('For circular feature j, let g_j be a finite-difference approximation to the spatial gradient of the manufactured forcing, evaluated at the feature center in physical coordinates (using the known interface mapping in the fitted-coefficient experiment). With q_0.9 the 90th percentile of the forcing-gradient magnitudes across candidates, define theta_j = min(1, |g_j|/max(q_0.9, 10^(-12))). The scale s_j = clip[s_0 {1 + eta (1/2 - theta_j)}, s_min, s_max] produces radius r_j = h_j s_j. The trial variants use s_0 = 1.5 and eta = 0.15 or 0.40. No exact solution values or measured approximation errors enter this radius rule. This is not a guarantee of local optimality: discontinuities, localized sources and quadrature influence its effectiveness.')
para('For a proposed child block C, compute the residual Schur matrix S_C and its normalized counterpart D_C^(-1/2) S_C D_C^(-1/2). Reject the candidate if its minimum eigenvalue is less than 10^(-7). The remaining children are ranked using the residual energy-reduction score of Section 5; this diagnostic only guards local linear dependence, and does not establish a global condition-number bound. Sparse CSR matrix assembly retains quadrature-induced sparsity; the present solver nevertheless converts selected systems to dense form and does not yet deliver sparse end-to-end complexity.')
para('Table 13. New pilot comparison at identical coefficient budgets. Relative L2 error is measured on an independent 48-subdivision evaluation rule for N = 65, and on a 64-subdivision rule for N = 129. All variants include full dictionary assembly in the measured construction time. Only single deterministic runs are reported; these are not equal-wall-clock comparisons or comparisons to published AB-PINN / sparse-RBF implementations.')
rows=json.loads((P/'results/ca_v2_pilot_small.json').read_text())['rows']
rows2=json.loads((P/'results/ca_v2_pilot_129.json').read_text())['rows']
selected=[r for r in rows if r['method'] in ('original','uniform_1p5','local_weak')]
selected+=rows2
from docx import Document
T=d.add_table(rows=1,cols=6);T.style='Table Grid'
for cell,t in zip(T.rows[0].cells,['Case','N','Radius rule','Relative L2','Scaled cond.','Time (s)']):cell.text=t
for r in selected:
    cc=T.add_row().cells
    for cell,t in zip(cc,[r['case'],str(r['size']),r['method'],f"{r['relative_l2']:.4e}",f"{r['scaled_condition']:.3e}",f"{r['construction_seconds']:.3f}"]):cell.text=t
anchor._p.addprevious(T._tbl)
para('At 129 coefficients, the local-radius variant yields 2.5963 × 10^(-5) on smooth Poisson versus 2.6398 × 10^(-5) for the uniform 1.5-radius variant, but is less accurate on the contrast-10 interface (8.0701 × 10^(-3) versus 7.7779 × 10^(-3)). Accordingly, neither local-radius superiority nor computational acceleration is established. The original unit-radius dictionary remains far better conditioned in these runs. The contrast-5 interface at 65 coefficients is a modest additional parameter-contrast test, not an independent physical benchmark.')
para('All 18 automated software tests pass, including agreement between sparse and dense quadrature assembly on a small test. Validation of the new nonlinear and transient solvers with locally changing radii is not yet available: the evolution results in Sections 8.8–8.9 belong to earlier global-radius variants. We therefore retain those results unchanged and do not present the new elliptic implementation as a validated time-dependent method. Independent published-method comparisons, memory profiling, equal-runtime benchmarking, larger-scale sparse solves and repeated experiments remain necessary before any overall state-of-the-art claim.')
# Insert a table/section before conclusions? Current limitations appropriate.
d.save(target)
print(target)
report=['# CA-AC-PINN v2 implementation and experimental audit','','## Verified implementation','- PDE-forcing-gradient-based locally varied Wendland radii (no exact errors used for selection).','- Locally conditioned Schur-complement block selection.','- CSR sparse stiffness/mass quadrature assembly (dense downstream solve remains).','- New baseline/variant runner and 5 new automated checks.','- 18/18 tests pass when legacy and new tests are combined.','','## Pilot benchmark protocol','Single CPU BLAS thread; 2D Poisson and discontinuous-diffusion interface at contrasts 10 and 5.','65 coefficient budget, training 24 subdivisions and independent evaluation 48 subdivisions; partial 129-coefficient benchmark with training 32 and evaluation 64.','All timings measure end-to-end construction of each selected model, but **not** fair equal-wall-clock tuning.','','## Results: 129 coefficients','| Case | Method | Relative L2 | Scaled condition | Construction seconds |','|---|---|---:|---:|---:|']
for r in rows2:report.append(f"| {r['case']} | {r['method']} | {r['relative_l2']:.7g} | {r['scaled_condition']:.5g} | {r['construction_seconds']:.3f} |")
report+=['','## Results: 65 coefficients','| Case | Method | Relative L2 | Scaled condition | Construction seconds |','|---|---|---:|---:|---:|']
for r in selected[:9]:report.append(f"| {r['case']} | {r['method']} | {r['relative_l2']:.7g} | {r['scaled_condition']:.5g} | {r['construction_seconds']:.3f} |")
report+=['','## Limitations','- Local radii do not consistently beat the tuned uniform-radius model; conditioning often worsens versus unscaled baseline.','- Comparisons are coefficient-matched, not equal stored parameters, tuning budgets, or wall-clock.','- New radius algorithm has NOT been applied to nonlinear or transient equations.','- CSR assembly equivalence is tested, but no sparse end-to-end runtime speedup is established.','- No execution of published AB-PINN or sparse RBF baselines; no SOTA superiority is claimed.','- The paper retains earlier experiments as separate historical v1.3 data.','- No GitHub publication has been made.']
(P/'VALIDATION_V2.md').write_text('\n'.join(report)+'\n')
