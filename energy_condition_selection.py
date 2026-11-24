"""Condition-guarded multiscale Ritz selection on a common quadrature rule.

The criterion uses ONLY assembled PDE energy and matrix conditioning;
independent exact-solution errors are reported AFTER selection for audit.
No guarantee transfers to nonlinear/time-dependent cases.
"""
import argparse, json, time
import numpy as np
from threadpoolctl import threadpool_limits
from acpinn import CircularDictionary, assemble, HierarchicalRitz, metrics
from problems import Poisson, Interface


def candidate(problem, scale, subdivisions, budget, evaluation):
    start=time.perf_counter()
    dictionary=CircularDictionary(dim=2,depth=3)
    dictionary.radii*=scale
    system=assemble(dictionary,problem,n=subdivisions,keep_samples=False)
    solver=HierarchicalRitz(dictionary,system.stiffness,system.rhs)
    solver.grow(budget)
    if len(solver.ids)!=budget:raise RuntimeError('Could not achieve requested coefficient budget')
    coeff=solver.coefficients()
    active=system.stiffness[np.ix_(solver.ids,solver.ids)]
    d=np.sqrt(np.diag(active))
    cond=float(np.linalg.cond(active/d[:,None]/d[None,:]))
    energy=float(solver.energy())
    error=metrics(dictionary,solver.ids,coeff,problem,n=evaluation)
    return {'scale':float(scale),'ritz_energy':energy,'scaled_condition':cond,'seconds':time.perf_counter()-start,**error}


def select(rows,condition_cap):
    eligible=[row for row in rows if np.isfinite(row['ritz_energy']) and row['scaled_condition']<=condition_cap]
    if not eligible:return None
    return min(eligible,key=lambda row:row['ritz_energy'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scales',nargs='+',type=float,default=[1.0,1.5,1.8]);p.add_argument('--condition-cap',type=float,default=2e6)
    p.add_argument('--n',type=int,default=48);p.add_argument('--evaluation',type=int,default=80)
    p.add_argument('--budget',type=int,default=129);p.add_argument('--output',default='results/energy_condition_selection_v13.json')
    a=p.parse_args()
    if any(s<=0 for s in a.scales):p.error('scales must be positive')
    report={'method':'minimum quadrature Ritz energy under condition guard','condition_cap':a.condition_cap,'training_subdivisions':a.n,'evaluation_subdivisions':a.evaluation,'budget':a.budget,'cases':{}}
    with threadpool_limits(limits=1):
        for name,problem in [('poisson',Poisson()),('interface',Interface())]:
            rows=[candidate(problem,scale,a.n,a.budget,a.evaluation) for scale in a.scales]
            chosen=select(rows,a.condition_cap)
            report['cases'][name]={'rows':rows,'selected_scale':None if chosen is None else chosen['scale'],'selected_energy':None if chosen is None else chosen['ritz_energy']}
            print(name,'selected',report['cases'][name]['selected_scale'],'rows',[(r['scale'],r['relative_l2'],r['scaled_condition']) for r in rows],flush=True)
    from pathlib import Path
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
