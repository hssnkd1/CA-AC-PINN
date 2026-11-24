"""Auditable new-method experiment; preserves reference baselines."""
import argparse, json, time, platform
from pathlib import Path
from threadpoolctl import threadpool_limits
from evolution import nonlinear_network, transient_network


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/v21_evolution_benchmark.json')
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    rows=[]
    cases=[('allen_cahn',.002),('burgers',.002),('transient',.005)]
    variants=[('uniform',1.5,0.),('local_weak',1.5,.15),('local_strong',1.5,.35)]
    with threadpool_limits(limits=1):
        for name,dt in cases:
            for method,base,strength in variants:
                started=time.perf_counter()
                try:
                    if name=='transient':result,_=transient_network(True,dt,radius_scale=base,local_strength=strength,schur_min_eig=1e-8)
                    else:result,_=nonlinear_network(name,True,dt,radius_scale=base,local_strength=strength,schur_min_eig=1e-8)
                    result['variant']=method
                    result['wall_clock_seconds']=time.perf_counter()-started
                    print(name,method,result['relative_l2'],result['seconds'],result.get('newton_failures'),flush=True)
                except Exception as exc:
                    result={'case':name,'variant':method,'failure':repr(exc)}
                    print('FAILURE',result,flush=True)
                rows.append(result)
                out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
                out.write_text(json.dumps({'protocol':'Single-threaded exploratory per-case comparisons; no SOTA claim',
                    'python':platform.python_version(),'rows':rows},indent=2))
    return rows
if __name__=='__main__':main()
