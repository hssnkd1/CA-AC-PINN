"""CA-AC-PINN v3.0: guarded, PDE-only candidate radius selection.

Mathematical scope: zero-Dirichlet symmetric diffusion problems in 2-D.
The guard chooses a complete Ritz approximation from candidate dictionaries;
independent quadrature is a validation estimator, NOT a certified bound.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from acpinn import CircularDictionary,assemble,metrics,gauss_grid
from ca_acpinn import ConditionAwareRitz,pde_local_radii

@dataclass
class SelectionSettings:
    train_n:int=24
    check_n:int=32
    evaluation_n:int=48
    budget:int=65
    condition_cap:float=2e6
    energy_tolerance:float=1e-9
    scales:tuple=(1.0,1.3,1.5)
    local_strengths:tuple=(0.15,)
    min_schur_eig:float=1e-8


def physical_energy(dictionary,ids,coeff,problem,n=32,order=3):
    """Compute J(v)=.5 int a|grad v|^2 - int f v on an independent grid."""
    x,w=gauss_grid(n,order,dictionary.dim)
    out=0.
    for start in range(0,len(x),1024):
        xb=x[start:start+1024];wb=w[start:start+1024]
        z,j=problem.mapping(xb)
        v,g,_=dictionary.evaluate(z,ids)
        g*=j[:,None,:]
        u=v@coeff
        grad=np.einsum('qnd,n->qd',g,coeff)
        out+=np.dot(wb,.5*problem.coefficient(xb)*np.sum(grad*grad,axis=1)-problem.forcing(xb)*u)
    return float(out)


def fit_candidate(problem,kind,value,settings):
    t0=perf_counter();d=CircularDictionary(dim=2,depth=3)
    diagnostic={}
    if kind=='uniform': d.radii*=float(value)
    elif kind=='local': diagnostic=pde_local_radii(d,problem,base_scale=1.5,strength=float(value))
    else:raise ValueError(kind)
    sys=assemble(d,problem,n=settings.train_n,keep_samples=False)
    model=ConditionAwareRitz(d,sys.stiffness,sys.rhs,min_schur_eig=settings.min_schur_eig)
    model.grow(settings.budget)
    ids=model.ids;coeff=model.coefficients()
    a=sys.stiffness[np.ix_(ids,ids)];dd=np.sqrt(np.diag(a))
    cond=float(np.linalg.cond(a/dd[:,None]/dd[None,:]))
    train_energy=model.energy()
    check_energy=physical_energy(d,ids,coeff,problem,n=settings.check_n)
    result={'kind':kind,'parameter':float(value),'size':len(ids),'condition':cond,
        'train_energy':train_energy,'check_energy':check_energy,
        'energy_quadrature_gap':abs(check_energy-train_energy),
        'rejected_blocks':model.rejected_blocks,
        'seconds':perf_counter()-t0,
        'storage_scalars':int(len(ids)+3*(len(ids)-d.ng)),
        'radius_diagnostic':{k:v for k,v in diagnostic.items() if k!='scales'}}
    return result,(d,ids,coeff)


def guarded_selection(problem,settings=SelectionSettings()):
    """Choose under a condition bound using ONLY training and check energies.

    First candidate, scale 1.0, is always the safe reference if admissible.
    For energy scores statistically/numerically indistinguishable within tolerance,
    prefer the lower-condition solution. No exact solution enters selection.
    """
    candidates=[('uniform',s) for s in settings.scales]+[('local',s) for s in settings.local_strengths]
    records=[];states=[]
    for kind,value in candidates:
        try:rec,state=fit_candidate(problem,kind,value,settings)
        except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
            rec={'kind':kind,'parameter':value,'failure':repr(exc)};state=None
        records.append(rec);states.append(state)
    eligible=[i for i,r in enumerate(records) if 'failure' not in r and np.isfinite(r['condition'])
              and r['condition']<=settings.condition_cap and np.isfinite(r['check_energy'])]
    if not eligible:raise RuntimeError('No stable candidate: relax cap or refine dictionary')
    best=min(eligible,key=lambda i:(records[i]['check_energy'],records[i]['condition']))
    # Explicit fallback to uniform candidate if energy difference is too small to resolve.
    reference=next((i for i in eligible if records[i]['kind']=='uniform' and records[i]['parameter']==1.0),None)
    if reference is not None:
        improvement=records[reference]['check_energy']-records[best]['check_energy']
        if improvement <= settings.energy_tolerance:
            best=reference
    for i,(r,state) in enumerate(zip(records,states)):
        r['admissible']=i in eligible
        r['selected']=i==best
        if state is not None:
            d,ids,coeff=state
            r['evaluation_metrics']=metrics(d,ids,coeff,problem,n=settings.evaluation_n)
    return {'case':problem.name,'algorithm':'v3 independent-energy guarded radius selection',
            'settings':vars(settings),'selected_index':best,'selected_kind':records[best]['kind'],
            'selected_parameter':records[best]['parameter'],'candidates':records}
