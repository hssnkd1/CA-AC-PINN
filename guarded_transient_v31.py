"""Three-policy 65-coefficient transient pilot. Weak scores select, not exact error."""
import json, time, pathlib
from threadpoolctl import threadpool_limits
from evolution import transient_network
from guarded_evolution_v31 import select

def run(output='results/guarded_transient_v31.json'):
    rows=[];tic=time.perf_counter()
    with threadpool_limits(limits=1):
        for label,strength in [('uniform',0.),('local_weak',.15),('local_strong',.35)]:
            try:
                row,_=transient_network(True,.005,radius_scale=1.5,local_strength=strength,
                        schur_min_eig=1e-8,target=65,quadrature_n=24,evaluation_n=40)
                row['variant']=label
            except Exception as e:row={'case':'transient','variant':label,'failure':repr(e)}
            print(label,row.get('heldout_weak_residual'),row.get('relative_l2'),flush=True)
            rows.append(row)
            pathlib.Path(output).write_text(json.dumps({'rows':rows,'selection_complete':False},indent=2))
    chosen=select([dict(r,newton_failures=0) if 'failure' not in r else r for r in rows])
    result={'case':'transient','test_budget':65,'quadrature_n':24,'evaluation_n':40,
            'dt':.005,'rows':rows,'selected_variant':None if chosen is None else chosen['variant'],
            'selection_total_seconds':time.perf_counter()-tic,
            'selection_complete':True,
            'note':'held-out weak residual heuristic at 65 coefficients; not the 129-coefficient study'}
    pathlib.Path(output).write_text(json.dumps(result,indent=2))
    return result
if __name__=='__main__':run()
