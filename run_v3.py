"""Run isolated v3 elliptical benchmarks; no exact field used for radius selection."""
import argparse,json,platform,time
from pathlib import Path
from dataclasses import asdict
from threadpoolctl import threadpool_limits
from v3_select import guarded_selection,SelectionSettings
from problems import Poisson,Interface

class InterfaceContrast(Interface):
    def __init__(self,contrast):
        self.contrast=contrast
        super().__init__()
        self.name=f'interface_contrast_{contrast:g}'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-n',type=int,default=24)
    p.add_argument('--check-n',type=int,default=32)
    p.add_argument('--evaluation-n',type=int,default=48)
    p.add_argument('--budget',type=int,default=65)
    p.add_argument('--output',default='results/v3_elliptic.json')
    a=p.parse_args();s=SelectionSettings(a.train_n,a.check_n,a.evaluation_n,a.budget)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    artifact={'version':'3.0','status':'elliptic guard benchmark only, not SOTA',
              'python':platform.python_version(),'protocol':'single-thread CPU, independent quadrature; no exact solution during selection',
              'rows':[]}
    for problem in (Poisson(),Interface(),InterfaceContrast(5.)):
        with threadpool_limits(limits=1): result=guarded_selection(problem,s)
        artifact['rows'].append(result)
        out.write_text(json.dumps(artifact,indent=2))
        print(result['case'],'SELECTED',result['selected_kind'],result['selected_parameter'],
              'L2',result['candidates'][result['selected_index']]['evaluation_metrics']['relative_l2'],flush=True)
    return artifact
if __name__=='__main__':main()
