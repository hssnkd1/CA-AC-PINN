"""Reproducible pilot with independent error quadrature and full construction time."""
import argparse,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ca_acpinn import ConditionAwareRitz,pde_local_radii
from acpinn import CircularDictionary,assemble,HierarchicalRitz,metrics
from problems import Poisson,Interface

def experiment(problem,method,n=32,evaluation=64,budget=129):
    start=time.perf_counter();d=CircularDictionary(2,3)
    diagnostic={}
    if method=='original':pass
    elif method=='uniform_1p5': d.radii*=1.5
    elif method=='local_weak':diagnostic=pde_local_radii(d,problem,1.5,.15)
    elif method=='local_strong':diagnostic=pde_local_radii(d,problem,1.5,.4)
    else:raise ValueError(method)
    system=assemble(d,problem,n=n,keep_samples=False)
    m=(HierarchicalRitz(d,system.stiffness,system.rhs) if method=='original' else
       ConditionAwareRitz(d,system.stiffness,system.rhs,min_schur_eig=1e-7,cost_power=.0))
    m.grow(budget)
    coeff=m.coefficients()
    total=time.perf_counter()-start
    active=system.stiffness[np.ix_(m.ids,m.ids)]
    v=np.sqrt(np.diag(active));cond=np.linalg.cond(active/v[:,None]/v[None,:])
    score=metrics(d,m.ids,coeff,problem,n=evaluation)
    return {'case':problem.name,'contrast':getattr(problem,'contrast',None),'method':method,
        'training_n':n,'evaluation_n':evaluation,'budget':budget,'size':len(m.ids),
        'construction_seconds':total,'scaled_condition':float(cond),'energy':m.energy(),
        'identity_defect_max':max([h['identity_defect'] for h in m.history],default=0),
        'rejected_schur_checks':getattr(m,'rejected_blocks',0),
        'stored_geometry_scalars':int(3*(len(m.ids)-d.ng)),'trained_coefficients':len(m.ids),
        'total_stored_scalars':int(len(m.ids)+3*(len(m.ids)-d.ng)),
        'radius_diagnostic':{k:v for k,v in diagnostic.items() if k!='scales'},**score}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--n',type=int,default=32)
    parser.add_argument('--evaluation',type=int,default=64);parser.add_argument('--budget',type=int,default=129)
    parser.add_argument('--output',default='results/ca_v2_pilot.json');a=parser.parse_args()
    rows=[]
    problems=[Poisson(),Interface(),Interface()]
    problems[-1]=InterfaceContrast(5.)
    with threadpool_limits(limits=1):
        for p in problems:
            for method in ['original','uniform_1p5','local_weak','local_strong']:
                try:
                    row=experiment(p,method,a.n,a.evaluation,a.budget)
                    print(p.name,getattr(p,'contrast',''),method,row['relative_l2'],row['construction_seconds'],row['scaled_condition'],flush=True)
                except Exception as e:
                    row={'case':p.name,'contrast':getattr(p,'contrast',None),'method':method,'failure':repr(e)}
                    print('FAILED',row,flush=True)
                rows.append(row)
    out=Path(a.output);out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps({'experiment':'CA-AC-PINN v2 pilot, NOT equal-wall-clock SOTA benchmark',
        'selection_rule':'local radii from forcing gradients, no exact solution in fitting/selection',
        'rows':rows},indent=2))

class InterfaceContrast(Interface):
    def __init__(self,contrast):
        self.contrast=contrast
        super().__init__()
        self.name=f'interface_contrast_{contrast:g}'

if __name__=='__main__':main()
