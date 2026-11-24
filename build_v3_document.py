"""Update the retained v2.1 Word manuscript without rewriting OMML equations."""
from pathlib import Path
import json
from docx import Document
from docx.shared import Inches,Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

root=Path(__file__).resolve().parent
source=root/'CA_AC_PINN_Manuscript_v2_1.docx'
if not source.exists(): source=Path('/mnt/data/CA_AC_PINN_v2_1/CA_AC_PINN_Manuscript_v2_1.docx')
output=root/'CA_AC_PINN_Manuscript_v3.docx'
small=json.loads((root/'results/v3_elliptic.json').read_text())
large=json.loads((root/'results/v3_elliptic_129.json').read_text())
doc=Document(source)
heading=next(p for p in doc.paragraphs if p.text.startswith('9 Verification'))

def before(txt,style=None):
    p=heading.insert_paragraph_before(txt,style=style)
    return p

before('8.12 CA-AC-PINN v3.0: independent-energy guarded radius selection', 'Heading 2')
before('The v3.0 extension selects a complete radius-policy candidate using the discrete Ritz fit followed by independent quadrature of the same physical energy. This addresses an empirical weakness of the initial-condition and forcing-gradient rules: a locally modified radius may worsen accuracy, and adaptation must not be accepted merely because it appears spatially plausible. The v3 selection is restricted to symmetric coercive elliptic diffusion with the homogeneous Dirichlet trace enforced by the trial family. The nonlinear and transient experiments from Section 8.11 are retained as v2.1 results, not presented as v3 validations.')
before('For every candidate radius policy s, we construct a hierarchical circular dictionary, solve its Ritz coefficient system, and compute J_check(v_s) = (1/2) int_Omega a |grad v_s|^2 dx - int_Omega f v_s dx by quadrature distinct from the assembly rule. Let K_s be the selected stiffness matrix and D_s = diag(K_s). The admissible set requires kappa_2(D_s^(-1/2) K_s D_s^(-1/2)) <= kappa_max. A policy with the minimum checked energy is accepted, subject to an energy-difference tolerance; if the estimated improvement over the uniform reference is smaller than that tolerance, the uniform reference is retained. Failed or unstable candidates are recorded, not silently discarded. The selector uses no exact solution; exact manufactured fields enter only the subsequently reported evaluation errors.')
before('For the exact continuous symmetric problem, the identity J(v)-J(u)=0.5 ||v-u||_a^2 implies that selecting lower exact energy among admissible trial solutions would also select lower energy error. Because J_check is evaluated numerically, this is not a certified acceptance guarantee. The stability cap controls the selected-system scaled condition number; it is not a convergence proof or a bound on the full candidate dictionary.')
before('The implementation uses a single-threaded CPU protocol. For the 65-coefficient exploratory runs, assembly uses 20 subdivisions, energy checking 28 subdivisions, and independent error evaluation 40 subdivisions. Four policies are compared: global radius multipliers 1.0, 1.3 and 1.5, and a forcing-gradient-based local rule with base multiplier 1.5 and strength 0.15. A second experiment uses 129 coefficients and subdivisions 32, 48 and 64 for two completed cases. The 129-coefficient contrast-5 case was interrupted by the execution limit and is not reported as a completed result.')
before('Table 15. CA-AC-PINN v3.0 independent-energy selection. Errors are relative L2 norms computed after selecting the radius policy. The uniform reference uses radius multiplier 1.5 in this comparison. All values are from executed JSON records.')
rows=[('65','Poisson',small['rows'][0]),('65','Interface contrast 10',small['rows'][1]),('65','Interface contrast 5',small['rows'][2]),('129','Poisson',large['rows'][0]),('129','Interface contrast 10',large['rows'][1])]
t=doc.add_table(rows=1, cols=6)
t.style='Table Grid'
for cell, value in zip(t.rows[0].cells,['N','Case','Uniform 1.5 L2','Selected policy','Selected L2','Condition no.']):cell.text=value
for n,name,item in rows:
    choice=item['candidates'][item['selected_index']]
    baseline=next(c for c in item['candidates'] if c['kind']=='uniform' and c['parameter']==1.5)
    values=[n,name,f"{baseline['evaluation_metrics']['relative_l2']:.6g}",f"{choice['kind']} ({choice['parameter']:.2f})",f"{choice['evaluation_metrics']['relative_l2']:.6g}",f"{choice['condition']:.3g}"]
    for c,v in zip(t.add_row().cells,values):c.text=v
for row in t.rows:
    for cell in row.cells:
        for pp in cell.paragraphs:
            for run in pp.runs:run.font.size=Pt(8)
heading._p.addprevious(t._tbl)
before('At 65 coefficients, the checked-energy rule selects the local policy in all three completed cases. For the contrast-10 interface, the relative L2 error changes from 0.03817 for uniform radius 1.5 to 0.02787 for the selected local representation. At 129 coefficients, the policy selects local radii for Poisson (2.5963 x 10^(-5)) but falls back to uniform radius 1.5 for the contrast-10 interface (7.7779 x 10^(-3)); the local candidate has a larger error of 8.0701 x 10^(-3). These observations are consistent with an adaptive decision rule, but selected and evaluated cases overlap with the policy-development tests, so out-of-distribution robustness remains unestablished.')
before('The energy checks and condition estimates add candidate-selection cost; the reported results are coefficient-matched, not equal-wall-clock or equal-memory comparisons. The full candidate matrix remains dense in the legacy Ritz stage despite the separate CSR assembly experiment. No direct AB-PINN or sparse-RBF baseline was run in this v3.0 experiment, and no performance claim is made for nonlinear or transient PDEs under the new selector.')
# integrate conclusions with explicit restriction
con=next(p for p in doc.paragraphs if p.text.startswith('The most useful next steps are'))
con.add_run(' The v3.0 guarded selection study provides a PDE-only way to reject an unfavorable local radius policy in tested symmetric elliptic cases; it does not extend the elliptic energy identity to Allen–Cahn or Burgers and is not a verified general speedup.')
abstract=next(p for p in doc.paragraphs if p.text.startswith('Localized solution features'))
abstract.add_run(' In an exploratory v3.0 extension, independent-quadrature energy and a selected-system condition guard select between uniform and local support policies for symmetric elliptic examples; broader robustness and equal-work superiority remain open.')
data=next(p for p in doc.paragraphs if p.text.startswith('The versioned supplementary reproducibility package'))
data.add_run(' The present v3.0 distribution adds v3_select.py, run_v3.py, test_v3.py, executed benchmark JSON records and documentation. A public repository URL will be inserted only after actual publication of the archived code; no DOI is asserted.')
doc.save(output)
print(output)
