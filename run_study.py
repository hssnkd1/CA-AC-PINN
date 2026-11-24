"""Reproduce all measured results used in the accompanying manuscript."""
import argparse
import json
import os
from pathlib import Path
import platform
import time
import gc
import numpy as np
from scipy.linalg import cho_factor,cho_solve,lstsq
from threadpoolctl import threadpool_limits,threadpool_info
from acpinn import CircularDictionary,assemble,HierarchicalRitz,metrics,fixed_circle_indices,prune_by_energy,gauss_grid
from problems import Poisson,Interface,Transient
from classical import fem,functional_majorant

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)


def dump(obj,name):
    (OUT/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')


def elliptic(problem,n=64):
    print('START',problem.name,flush=True)
    d=CircularDictionary(2,3);system=assemble(d,problem,n=n,keep_samples=False)
    tic=time.perf_counter();model=HierarchicalRitz(d,system.stiffness,system.rhs)
    solve_seconds=time.perf_counter()-tic
    snapshots=[]
    for target in [17,33,65,97,129,193,257]:
        tic=time.perf_counter();model.grow(target)
        c=model.coefficients();solve_seconds+=time.perf_counter()-tic
        row=metrics(d,model.ids,c,problem)
        row.update({'method':'AC-PINN','size':len(model.ids),'seconds':system.assembly_seconds+solve_seconds})
        snapshots.append(row)
        print(problem.name,len(model.ids),row['relative_l2'],row['relative_energy'],flush=True)
        np.savez(OUT/f'{problem.name}_adaptive_{len(model.ids)}.npz',ids=model.ids,c=c)
    # Independent assembly check of the selected 129-dimensional space.
    ids129=np.load(OUT/f'{problem.name}_adaptive_129.npz')['ids'].tolist()
    c129=np.load(OUT/f'{problem.name}_adaptive_129.npz')['c']
    fm=functional_majorant(d,ids129,c129,problem,n=24)
    fm['effectivity']=fm['majorant']/next(q['energy_error'] for q in snapshots if q['size']==129)
    selected=CircularDictionary(2,3)
    x,w=gauss_grid(96,3,2);kk=np.zeros((129,129));ff=np.zeros(129)
    for start in range(0,len(x),2048):
        xb=x[start:start+2048];wb=w[start:start+2048];z,jac=problem.mapping(xb)
        vv,gg,_=selected.evaluate(z,ids129);gg*=jac[:,None,:]
        kk+=sum(gg[:,:,ax].T@((wb*problem.coefficient(xb))[:,None]*gg[:,:,ax]) for ax in range(2))
        ff+=vv.T@(wb*problem.forcing(xb))
    cc=cho_solve(cho_factor(kk,lower=True),ff)
    quadrature={'training_subdivisions':n,'check_subdivisions':96,
                'coefficient_relative_change':float(np.linalg.norm(cc-c129)/np.linalg.norm(c129)),
                'refit_metrics':metrics(d,ids129,cc,problem,n=80)}
    quadrature['levels']=[{'subdivisions':64,**metrics(d,ids129,c129,problem)},
                          {'subdivisions':96,**quadrature['refit_metrics']}]
    x32,w32=gauss_grid(32,3,2);k32=np.zeros((129,129));f32=np.zeros(129)
    for start in range(0,len(x32),2048):
        xb=x32[start:start+2048];wb=w32[start:start+2048];z,jac=problem.mapping(xb)
        vv,gg,_=d.evaluate(z,ids129);gg*=jac[:,None,:]
        k32+=sum(gg[:,:,ax].T@((wb*problem.coefficient(xb))[:,None]*gg[:,:,ax]) for ax in range(2))
        f32+=vv.T@(wb*problem.forcing(xb))
    c32=cho_solve(cho_factor(k32,lower=True),f32)
    quadrature['levels'].insert(0,{'subdivisions':32,**metrics(d,ids129,c32,problem)})
    ksel=system.stiffness[np.ix_(ids129,ids129)];scale=np.sqrt(np.diag(ksel))
    quadrature['scaled_condition_number']=float(np.linalg.cond(ksel/scale[:,None]/scale[None,:]))
    # Same uniform circle geometry for weak and strong fixed-space baselines.
    fd=fixed_circle_indices(d,n=11)
    fs=assemble(fd,problem,n=n);st=time.perf_counter()
    fc=cho_solve(cho_factor(fs.stiffness,lower=True),fs.rhs)
    ft=fs.assembly_seconds+time.perf_counter()-st
    fixed=metrics(fd,list(range(fd.size)),fc,problem)
    fixed.update({'method':'Fixed circular Ritz','size':fd.size,'seconds':ft})
    np.savez(OUT/f'{problem.name}_fixed.npz',c=fc)
    st=time.perf_counter()
    z,jac=problem.mapping(fs.points);vals,grads,hess=fd.evaluate(z)
    aa=problem.coefficient(fs.points);ag=problem.coefficient_gradient(fs.points)
    op=-aa[:,None]*np.sum(hess*jac[:,None,:]**2,axis=2)-np.sum(grads*jac[:,None,:]*ag[:,None,:],axis=2)
    scales=np.sqrt(np.sum(fs.weights[:,None]*op**2,axis=0));scales=np.maximum(scales,1.e-15)
    sc=lstsq(np.sqrt(fs.weights)[:,None]*op/scales,np.sqrt(fs.weights)*problem.forcing(fs.points),lapack_driver='gelsy')[0]/scales
    stime=fs.assembly_seconds+time.perf_counter()-st
    strong=metrics(fd,list(range(fd.size)),sc,problem)
    strong.update({'method':'Fixed circular strong','size':fd.size,'seconds':stime})
    norm_energy=float(c129@system.stiffness[np.ix_(ids129,ids129)]@c129)
    pruned=[]
    for rel in [1.e-8,1.e-6,1.e-4]:
        pi,pc,meta=prune_by_energy(system.stiffness,system.rhs,ids129,.5*rel*norm_energy,protected=d.ng)
        met=metrics(d,pi,pc,problem);met.update(meta);met.update({'tolerance_squared_relative':rel,'size':len(pi)})
        pruned.append(met)
    classical=[fem(problem,nv) for nv in [16,32,64]]
    result={'case':problem.name,'assembly_seconds':system.assembly_seconds,'quadrature_points':len(system.points),'dictionary_size':d.size,
            'adaptive':snapshots,'fixed':fixed,'strong':strong,'fem':classical,'majorant':fm,'quadrature_check':quadrature,
            'pruning':pruned,'adaptation_history':model.history}
    dump(result,f'{problem.name}.json')
    print('DONE',problem.name,flush=True)
    del system,fs;gc.collect()
    return result


def baselines(iterations=1500,seeds=3):
    from pinn_baseline import train
    for problem in [Poisson(),Interface()]:
        rows=[]
        for kind in ['standard','residual_adaptive','gradient_enhanced']:
            for seed in range(seeds):
                print('PINN',problem.name,kind,seed,flush=True)
                row,theta=train(problem,seed,kind,iterations=iterations)
                rows.append(row);np.savez(OUT/f'{problem.name}_{kind}_{seed}.npz',theta=theta)
                dump(rows,f'{problem.name}_pinn.json')
                print('PINN RESULT',row['relative_l2'],row['relative_energy'],row['seconds'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['elliptic','pinn','evolution','all'],default='all')
    ap.add_argument('--iterations',type=int,default=1500);ap.add_argument('--seeds',type=int,default=3)
    args=ap.parse_args()
    with threadpool_limits(limits=1):
        dump({'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,
              'threads':1,'blas':threadpool_info(),'arguments':vars(args)},'environment.json')
        if args.part in ['elliptic','all']:
            for p in [Poisson(),Interface()]:elliptic(p)
        if args.part in ['pinn','all']:baselines(args.iterations,args.seeds)
        if args.part in ['evolution','all']:
            from evolution import run_evolution
            run_evolution(OUT)


if __name__=='__main__':main()
