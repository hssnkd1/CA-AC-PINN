"""Reproducible radius sensitivity experiment with unchanged degree-of-freedom budget.

Run: OPENBLAS_NUM_THREADS=1 python radius_sensitivity.py
This experiment is distinct from the original reported main benchmark and does
not overwrite any pre-existing result or fitted-coefficient file.
"""
import argparse
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from acpinn import CircularDictionary, HierarchicalRitz, assemble, metrics
from problems import Poisson, Interface


def run_one(problem, scale, subdivisions=64, target=129, evaluation_subdivisions=80):
    dictionary = CircularDictionary(2, 3)
    dictionary.radii *= scale
    start = perf_counter()
    system = assemble(dictionary, problem, n=subdivisions, keep_samples=False)
    model = HierarchicalRitz(dictionary, system.stiffness, system.rhs)
    model.grow(target)
    if len(model.ids) != target:
        raise RuntimeError(f'Unable to reach target dimension: {len(model.ids)}')
    coefficients = model.coefficients()
    elapsed = perf_counter() - start
    accuracy = metrics(dictionary, model.ids, coefficients, problem, n=evaluation_subdivisions)
    active = system.stiffness[np.ix_(model.ids, model.ids)]
    d = np.sqrt(np.diag(active))
    condition = np.linalg.cond(active / d[:, None] / d[None, :])
    return dict(case=problem.name, radius_scale=scale, active_coefficients=len(model.ids),
                training_subdivisions=subdivisions, evaluation_subdivisions=evaluation_subdivisions,
                seconds=elapsed, scaled_condition_number=float(condition),
                max_refinement_identity_defect=max([h['identity_defect'] for h in model.history], default=0),
                **accuracy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scales', nargs='+', type=float, default=[1.0, 1.2, 1.5])
    parser.add_argument('--cases', nargs='+', choices=['poisson', 'interface'], default=['poisson','interface'])
    parser.add_argument('--subdivisions', type=int, default=64)
    parser.add_argument('--output', default='results/radius_sensitivity.json')
    a=parser.parse_args()
    output=Path(a.output);output.parent.mkdir(parents=True, exist_ok=True)
    results=[]
    with threadpool_limits(limits=1):
        for case in a.cases:
            for scale in a.scales:
                if scale<=0:raise ValueError('All radius scales must be positive')
                print(f'RUN {case} scale={scale}',flush=True)
                result=run_one(Poisson() if case=='poisson' else Interface(),scale,a.subdivisions)
                results.append(result)
                output.write_text(json.dumps({'description':'Independent radius sensitivity; not original benchmark',
                    'platform':platform.platform(),'python':platform.python_version(),
                    'numpy':np.__version__,'results':results},indent=2))
                print(f"  L2={result['relative_l2']:.7g}; energy={result['relative_energy']:.7g}; condition={result['scaled_condition_number']:.3g}",flush=True)

if __name__=='__main__':main()
