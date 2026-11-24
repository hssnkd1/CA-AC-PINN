"""Front-case guarded policy selection using common held-out weak test functions.

Unlike an exact-error oracle, the criterion uses the final-step discrete PDE
residual against 16 fixed sine test functions. It is a *heuristic*: a smaller
weak residual is not a guarantee of lower L2 or energy error. The full costs
of every trial candidate are included. No selection from the reported errors.
"""
import argparse, json, time, platform
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from evolution import nonlinear_network

def select(rows):
    feasible=[r for r in rows if 'failure' not in r and r.get('newton_failures',1)==0
              and np.isfinite(r.get('heldout_weak_residual',np.nan))]
    if not feasible:return None
    return min(feasible,key=lambda r:r['heldout_weak_residual'])

def run(output, kinds=('allen_cahn','burgers'), dt=.002):
    report={'version':'3.1','method':'held-out weak-form sine test residual; NOT an exact-error selector',
     'protocol':'single-thread CPU double precision, total selection time includes all candidates',
     'python':platform.python_version(),'cases':{}}
    with threadpool_limits(limits=1):
      for kind in kinds:
        rows=[]
        start=time.perf_counter()
        for name,strength in [('uniform',0.),('local_weak',.15),('local_strong',.35)]:
          try:
            row,_=nonlinear_network(kind,True,dt,radius_scale=1.5,local_strength=strength,schur_min_eig=1e-8)
            row['variant']=name
          except Exception as exc:row={'case':kind,'variant':name,'failure':repr(exc)}
          rows.append(row)
          print(kind,name,row.get('heldout_weak_residual'),row.get('relative_l2'),flush=True)
        chosen=select(rows)
        report['cases'][kind]={'candidates':rows,'selected_variant':chosen['variant'] if chosen else None,
          'selection_total_seconds':time.perf_counter()-start}
        Path(output).write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/guarded_evolution_v31.json');p.add_argument('--dt',type=float,default=.002)
    a=p.parse_args();Path(a.output).parent.mkdir(parents=True,exist_ok=True);run(a.output,dt=a.dt)
