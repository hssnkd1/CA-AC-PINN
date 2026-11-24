"""Non-oracular consistency checks on saved v3.1 candidate logs."""
from pathlib import Path
import json, math

def verify(root=None):
    root=Path(root) if root else Path(__file__).resolve().parent
    e=json.loads((root/'results/guarded_evolution_v31.json').read_text())
    assert set(e['cases'])=={'allen_cahn','burgers'}
    for name,block in e['cases'].items():
        rows=block['candidates']
        assert len(rows)==3
        assert not any('failure' in r or r.get('newton_failures') for r in rows)
        best=min(rows,key=lambda r:r['heldout_weak_residual'])
        assert best['variant']==block['selected_variant']
        assert all(math.isfinite(r['relative_l2']) and math.isfinite(r['heldout_weak_residual']) for r in rows)
        assert block['selection_total_seconds']>0
    t=json.loads((root/'results/guarded_transient_v31.json').read_text())
    assert t['selection_complete'] and len(t['rows'])==3
    assert min(t['rows'],key=lambda r:r['heldout_weak_residual'])['variant']==t['selected_variant']
    assert all(r['size']==65 and r['dt']==.005 for r in t['rows'])
    return {'elliptic_previous_v3':'retained','completed_nonlin_cases':2,'completed_transient_pilot':1,'candidate_runs':9}

if __name__=='__main__':print(verify())
