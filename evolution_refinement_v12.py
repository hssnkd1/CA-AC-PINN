"""Refinement grid at radius 1.5, preserving original discretization settings."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
from threadpoolctl import threadpool_limits
from evolution import nonlinear_network,transient_network,transient_fdm,finite_difference_front
from pathlib import Path
import json,platform
path=Path('results/evolution_refinement_v12.json');rows=[]
with threadpool_limits(limits=1):
 for kind in ('allen_cahn','burgers'):
  for dt in (.004,.002,.001):
   for adaptive in (True,False):
    row,_=nonlinear_network(kind,adaptive,dt,radius_scale=1.5);row.pop('history',None);row['radius_scale']=1.5;rows.append(row);print(kind,dt,adaptive,row['relative_l2'],flush=True)
   row,_=finite_difference_front(kind,dt);rows.append(row)
   path.write_text(json.dumps(rows,indent=2))
 for dt in (.01,.005,.0025):
  for adaptive in (True,False):
   row,_=transient_network(adaptive,dt,radius_scale=1.5);row.pop('history',None);row['radius_scale']=1.5;rows.append(row);print('transient',dt,adaptive,row['relative_l2'],flush=True)
  rows.append(transient_fdm(dt));path.write_text(json.dumps(rows,indent=2))
