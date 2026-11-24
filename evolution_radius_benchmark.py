"""Controlled sequential support radius sweeps at matched coefficient budgets.
Experiment: all original test cases x scales 1.0,1.2,1.5 with fixed and adaptive.
Same CPU thread limits, quadrature rules, time-step, and independent evaluation.
No tuning using test metrics is applied during a single run.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import json,platform,argparse,time
from pathlib import Path
from threadpoolctl import threadpool_limits
from evolution import nonlinear_network,transient_network,finite_difference_front,transient_fdm

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scales',nargs='+',type=float,default=[1.0,1.2,1.5]);ap.add_argument('--output',default='results/evolution_radius_benchmark.json')
    ap.add_argument('--dt-front',type=float,default=.002);ap.add_argument('--dt-transient',type=float,default=.005)
    args=ap.parse_args();path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    rows=[];meta={'design':'exploratory radius sweep; identical train/evaluation settings; not an AB-PINN or sparse-RBF implementation','python':platform.python_version(),'machine':platform.platform(),'front_dt':args.dt_front,'transient_dt':args.dt_transient,'results':rows}
    with threadpool_limits(limits=1):
        for case in ['allen_cahn','burgers','transient']:
            for scale in args.scales:
                for adaptive in [True,False]:
                    t0=time.perf_counter()
                    try:
                        row,_ = (transient_network(adaptive, args.dt_transient, radius_scale=scale) if case=='transient' else nonlinear_network(case,adaptive,args.dt_front,radius_scale=scale))
                        row.pop('history',None);row['radius_scale']=scale;row['success']=True
                        row['stored_scalar_estimate'] = int(row['size']+(row['size']-(8 if case=='transient' else 3))*(3 if case=='transient' else 2))
                    except Exception as exc:
                        row={'case':case,'method':'Adaptive circular' if adaptive else 'Fixed circular','radius_scale':scale,'success':False,'error':repr(exc),'seconds':time.perf_counter()-t0}
                    rows.append(row);path.write_text(json.dumps(meta,indent=2));print('RESULT',case,scale,adaptive,row,flush=True)
            ref=(transient_fdm(args.dt_transient) if case=='transient' else finite_difference_front(case,args.dt_front));row=ref[0] if isinstance(ref,tuple) else ref;row['radius_scale']=None;row['success']=True;rows.append(row);path.write_text(json.dumps(meta,indent=2));print('REFERENCE',case,row,flush=True)
if __name__=='__main__':main()
