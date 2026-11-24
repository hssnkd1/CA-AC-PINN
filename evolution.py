"""Forced Allen-Cahn, exact Burgers travelling wave, transient diffusion."""
from pathlib import Path
import json
import time
import numpy as np
from scipy import sparse
from scipy.linalg import solve,cho_factor,cho_solve
from scipy.sparse.linalg import spsolve,factorized
from acpinn import CircularDictionary,gauss_grid,fixed_circle_indices,assemble,HierarchicalRitz,metrics
from problems import Transient
from local_radius import initial_gradient_radii


def front(x,t,kind):
    if kind=='allen_cahn':
        eps=.02;speed=.2;width=np.sqrt(2)*eps
        z=(x-.45-speed*t)/width;u=np.tanh(z)
        ux=(1-u*u)/width;ut=-speed*ux
        return u,ux,ut
    nu=.005;speed=.3;amp=.5
    z=amp*(x-.35-speed*t)/(2*nu);h=np.tanh(z)
    u=speed-amp*h;ux=-amp*amp/(2*nu)*(1-h*h)
    return u,ux,np.zeros_like(x)


def nonlinear_network(kind,adaptive=True,dt=.002,target=64,radius_scale=1.0,local_strength=0.,schur_min_eig=0.):
    tic=time.perf_counter();T=.1 if kind=='allen_cahn' else .2
    nsteps=int(round(T/dt));dt=T/nsteps
    d=CircularDictionary(1,5) if adaptive else fixed_circle_indices(CircularDictionary(1,0),n=target-3)
    radius_diagnostic={}
    if local_strength:
        radius_diagnostic=initial_gradient_radii(d,lambda z: front(z[:,0],0,kind)[0],
                                                 strength=local_strength,base_scale=radius_scale)
    else:
        d.radii *= radius_scale
    x,w=gauss_grid(256,4,1);xx=x[:,0]
    v,g,_=d.evaluate(x);g=g[:,:,0]
    mass=v.T@(w[:,None]*v);stiff=g.T@(w[:,None]*g)
    nu=.02**2 if kind=='allen_cahn' else .005
    metric=mass/dt+nu*stiff
    ids=d.initial.copy() if adaptive else list(range(d.size));split=set()
    old=front(xx,0,kind)[0];c=np.zeros(len(ids));trace=[];failures=0

    def current(t,ii,cc):
        bc=front(np.array([0.,1.]),t,kind)[0]
        lift=bc[0]*(1-xx)+bc[1]*xx;liftx=np.full(len(xx),bc[1]-bc[0])
        return lift+v[:,ii]@cc,liftx+g[:,ii]@cc

    for step in range(1,nsteps+1):
        old_at_step_start=old.copy()
        t=step*dt;source=front(xx,t,kind)[2]
        if kind=='burgers':source=np.zeros(len(xx))
        maxsplit=8 if step==1 else 2
        for refinement in range(maxsplit+1):
            vv=v[:,ids];gg=g[:,ids]
            for iteration in range(25):
                u,ux=current(t,ids,c)
                if kind=='allen_cahn':
                    nonlinear=u**3-u
                    H=metric[np.ix_(ids,ids)]+vv.T@((w*(3*u*u-1))[:,None]*vv)
                else:
                    nonlinear=u*ux
                    H=metric[np.ix_(ids,ids)]+vv.T@(w[:,None]*(ux[:,None]*vv+u[:,None]*gg))
                r=vv.T@(w*((u-old)/dt+nonlinear-source))+nu*gg.T@(w*ux)
                if np.linalg.norm(r)<1.e-10:break
                delta=solve(H,-r,assume_a='gen',check_finite=False)
                factor=1.
                for ls in range(12):
                    cu,cux=current(t,ids,c+factor*delta)
                    nl=cu**3-cu if kind=='allen_cahn' else cu*cux
                    rr=vv.T@(w*((cu-old)/dt+nl-source))+nu*gg.T@(w*cux)
                    if np.linalg.norm(rr)<np.linalg.norm(r) or factor<.001:break
                    factor*=.5
                c+=factor*delta
            if iteration==24:failures+=1
            if not adaptive or len(ids)+2>target or refinement==maxsplit:break
            u,ux=current(t,ids,c)
            nl=u**3-u if kind=='allen_cahn' else u*ux
            residual=v.T@(w*((u-old)/dt+nl-source))+nu*g.T@(w*ux)
            amat=metric[np.ix_(ids,ids)];fac=cho_factor(amat,lower=True,check_finite=False)
            parents=[p for p in ids if p in d.children and p not in split]
            if not parents:break
            scores=[]
            for parent in parents:
                child=d.children[parent]
                cross=metric[np.ix_(ids,child)]
                s=metric[np.ix_(child,child)]-cross.T@cho_solve(fac,cross,check_finite=False)
                try:
                    diag=np.diag(s)
                    if np.any(diag<=0): raise np.linalg.LinAlgError('nonpositive Schur')
                    if schur_min_eig:
                        e=np.linalg.eigvalsh((s/np.sqrt(diag[:,None]*diag[None,:]) +
                                             (s/np.sqrt(diag[:,None]*diag[None,:])).T)/2)
                        if e[0]<schur_min_eig: raise np.linalg.LinAlgError('unstable Schur block')
                    score=residual[child]@np.linalg.solve(s,residual[child])
                except np.linalg.LinAlgError:score=-np.inf
                scores.append(score)
            if not np.any(np.isfinite(scores)): break
            parent=parents[int(np.argmax(scores))];child=d.children[parent]
            ids.extend(child);c=np.r_[c,np.zeros(len(child))];split.add(parent)
        old,oldx=current(t,ids,c)
        ue,ge,_=front(xx,t,kind)
        trace.append({'time':t,'size':len(ids),'relative_l2':float(np.sqrt(np.sum(w*(old-ue)**2)/np.sum(w*ue**2)))})
    # Held-out weak residual on a common sine test space. This diagnostic does
    # not evaluate the exact solution; the manufactured source defines the PDE.
    kk=np.arange(1,17,dtype=float)
    tests=np.sin(np.pi*xx[:,None]*kk[None,:])
    testgrad=np.pi*kk[None,:]*np.cos(np.pi*xx[:,None]*kk[None,:])
    nonlinear_final=old**3-old if kind=='allen_cahn' else old*oldx
    forcing_final=front(xx,T,kind)[2] if kind=='allen_cahn' else np.zeros_like(xx)
    weak=tests.T@(w*((old-old_at_step_start)/dt+nonlinear_final-forcing_final)) + nu*testgrad.T@(w*oldx)
    independent_weak_residual=float(np.linalg.norm(weak))
    seconds=time.perf_counter()-tic
    ue,ge,_=front(xx,T,kind)
    row={'case':kind,'method':'Adaptive circular' if adaptive else 'Fixed circular','size':len(ids),'dt':dt,'seconds':seconds,
         'relative_l2':float(np.sqrt(np.sum(w*(old-ue)**2)/np.sum(w*ue**2))),
         'relative_gradient':float(np.sqrt(np.sum(w*(oldx-ge)**2)/np.sum(w*ge**2))),
         'newton_failures':failures,'heldout_weak_residual':independent_weak_residual,'history':trace,'radius_diagnostic':radius_diagnostic, 'local_strength':local_strength, 'radius_scale':radius_scale}
    return row,{'x':xx,'u':old,'exact':ue,'ids':ids,'c':c}


def finite_difference_front(kind,dt=.002,n=512):
    tic=time.perf_counter();T=.1 if kind=='allen_cahn' else .2
    steps=int(round(T/dt));dt=T/steps;h=1/n;x=np.linspace(0,1,n+1)
    u=front(x,0,kind)[0];nu=.02**2 if kind=='allen_cahn' else .005
    for step in range(1,steps+1):
        t=step*dt;bc=front(np.array([0.,1.]),t,kind)[0];old=u.copy();u[0]=bc[0];u[-1]=bc[1]
        source=front(x[1:-1],t,kind)[2]
        for _ in range(15):
            ux=(u[2:]-u[:-2])/(2*h);lap=(u[2:]-2*u[1:-1]+u[:-2])/h**2
            if kind=='allen_cahn':
                r=(u[1:-1]-old[1:-1])/dt-nu*lap+u[1:-1]**3-u[1:-1]-source
                diagonal=1/dt+2*nu/h**2+3*u[1:-1]**2-1
                lower=np.full(n-2,-nu/h**2);upper=lower.copy()
            else:
                r=(u[1:-1]-old[1:-1])/dt-nu*lap+u[1:-1]*ux
                diagonal=1/dt+2*nu/h**2+ux
                lower=-nu/h**2-u[2:-1]/(2*h);upper=-nu/h**2+u[1:-2]/(2*h)
            if np.linalg.norm(r)<1.e-9:break
            j=sparse.diags([lower,diagonal,upper],[-1,0,1],format='csc')
            u[1:-1]-=spsolve(j,r)
    ue,ge,_=front(x,T,kind);w=np.full(n+1,h);w[[0,-1]]=h/2
    return {'case':kind,'method':'Central FDM','size':n-1,'dt':dt,'seconds':time.perf_counter()-tic,
            'relative_l2':float(np.sqrt(np.sum(w*(u-ue)**2)/np.sum(w*ue**2))),
            'relative_gradient':float(np.sqrt(np.sum(w*(np.gradient(u,h,edge_order=2)-ge)**2)/np.sum(w*ge**2)))}, {'x':x,'u':u,'exact':ue}


def transient_network(adaptive=True,dt=.005,radius_scale=1.0,local_strength=0.,schur_min_eig=0.,target=129,quadrature_n=48,evaluation_n=80):
    tic=time.perf_counter();p=Transient(0)
    d=CircularDictionary(2,3) if adaptive else fixed_circle_indices(CircularDictionary(2,0),11)
    radius_diagnostic={}
    if local_strength:
        radius_diagnostic=initial_gradient_radii(d,lambda z: p.field0(z)[0],
                                                strength=local_strength,base_scale=radius_scale)
    else:
        d.radii *= radius_scale
    s=assemble(d,p,n=quadrature_n,order=3,keep_samples=False)
    # Initial data are prescribed analytically and integrated independently.
    x,w=gauss_grid(quadrature_n,3,2);m0=np.zeros(d.size)
    for start in range(0,len(x),2048):
        xb=x[start:start+2048];wb=w[start:start+2048]
        v,_,_=d.evaluate(xb);m0+=v.T@(wb*p.field0(xb)[0])
    a=s.mass/dt+s.stiffness;f0=s.rhs.copy()
    ids=d.initial.copy() if adaptive else list(range(d.size));previous=None;trace=[]
    steps=int(round(.1/dt));dt=.1/steps
    for step in range(1,steps+1):
        last_prev=None if previous is None else previous.copy()
        t=step*dt
        rhs=np.exp(-t)*f0+(m0 if previous is None else s.mass[:,ids]@previous)/dt
        if adaptive and step==1:
            if schur_min_eig:
                from ca_acpinn import ConditionAwareRitz
                model=ConditionAwareRitz(d,a,rhs,min_schur_eig=schur_min_eig)
            else:
                model=HierarchicalRitz(d,a,rhs)
            model.grow(target);ids=model.ids.copy();c=model.coefficients()
        else:c=cho_solve(cho_factor(a[np.ix_(ids,ids)],lower=True),rhs[ids])
        previous=c.copy();trace.append({'time':t,'size':len(ids)})
    # Common held-out 2D sine-test weak defect (final backward Euler step).
    # For this benchmark, there are multiple time steps; last_prev is available.
    check_x,check_w=gauss_grid(18,3,2)
    last_p=Transient(.1)
    weak=np.zeros(9)
    modes=[(i,j) for i in range(1,4) for j in range(1,4)]
    for start in range(0,len(check_x),512):
        xb=check_x[start:start+512];wb=check_w[start:start+512]
        vv,gg,_=d.evaluate(xb,ids)
        current=vv@c;old_u=vv@last_prev if last_prev is not None else last_p.field0(xb)[0]
        grad=np.einsum('qnd,n->qd',gg,c)
        xs=xb[:,0,None];ys=xb[:,1,None]
        ki=np.array([m[0] for m in modes])[None,:];kj=np.array([m[1] for m in modes])[None,:]
        psi=np.sin(np.pi*ki*xs)*np.sin(np.pi*kj*ys)
        gx=np.pi*ki*np.cos(np.pi*ki*xs)*np.sin(np.pi*kj*ys)
        gy=np.pi*kj*np.sin(np.pi*ki*xs)*np.cos(np.pi*kj*ys)
        coeff=last_p.coefficient(xb)
        weak+=psi.T@(wb*((current-old_u)/dt-last_p.forcing(xb))) + gx.T@(wb*coeff*grad[:,0]) + gy.T@(wb*coeff*grad[:,1])
    independent_weak_residual=float(np.linalg.norm(weak))
    seconds=time.perf_counter()-tic;p.t=.1
    met=metrics(d,ids,c,p,n=evaluation_n);met.update({'case':'transient','method':'Adaptive circular' if adaptive else 'Fixed circular','size':len(ids),'dt':dt,'seconds':seconds,'history':trace,'radius_diagnostic':radius_diagnostic,'local_strength':local_strength,'radius_scale':radius_scale,'heldout_weak_residual':independent_weak_residual,'quadrature_n':quadrature_n})
    return met,{'ids':ids,'c':c}


def transient_fdm(dt=.005,n=80):
    tic=time.perf_counter();h=1/n;p=Transient(0)
    v=np.linspace(0,1,n+1)[1:-1];xy=np.stack(np.meshgrid(v,v,indexing='ij'),axis=-1).reshape(-1,2)
    N=len(xy);rows=[];cols=[];data=[];diag=np.zeros(N)
    for axis in range(2):
        for sign in [-1,1]:
            face=xy.copy();face[:,axis]+=sign*h/2;aa=p.coefficient(face)/h**2
            diag+=aa
            keep=(xy[:,axis]+sign*h>1.e-10)&(xy[:,axis]+sign*h<1-1.e-10)
            offset=sign*((n-1) if axis==0 else 1)
            row=np.flatnonzero(keep);rows.extend(row);cols.extend(row+offset);data.extend(-aa[keep])
    rows.extend(range(N));cols.extend(range(N));data.extend(diag)
    k=sparse.coo_matrix((data,(rows,cols)),shape=(N,N)).tocsc()
    a=sparse.eye(N,format='csc')/dt+k;solver=factorized(a)
    u=p.field0(xy)[0];f0=p.forcing(xy)
    steps=int(round(.1/dt))
    for step in range(1,steps+1):u=solver(u/dt+np.exp(-step*dt)*f0)
    p.t=.1;ue=p.exact(xy)[0]
    return {'case':'transient','method':'Conservative FDM','size':N,'dt':dt,'seconds':time.perf_counter()-tic,
            'relative_l2':float(np.linalg.norm(u-ue)/np.linalg.norm(ue))}


def run_evolution(out):
    out=Path(out);rows=[]
    for kind in ['allen_cahn','burgers']:
        for dt in [.004,.002,.001]:
            for adaptive in [True,False]:
                row,state=nonlinear_network(kind,adaptive,dt)
                rows.append(row);np.savez(out/f'{kind}_{"adaptive" if adaptive else "fixed"}_{dt}.npz',**state)
                (out/'evolution.json').write_text(json.dumps(rows,indent=2))
                print('EVOLUTION',kind,adaptive,dt,row['relative_l2'],row['newton_failures'],flush=True)
            row,state=finite_difference_front(kind,dt);rows.append(row)
            np.savez(out/f'{kind}_fdm_{dt}.npz',**state)
    for dt in [.01,.005,.0025]:
        for adaptive in [True,False]:
            row,state=transient_network(adaptive,dt);rows.append(row)
            np.savez(out/f'transient_{"adaptive" if adaptive else "fixed"}_{dt}.npz',**state)
            (out/'evolution.json').write_text(json.dumps(rows,indent=2))
            print('EVOLUTION transient',adaptive,dt,row['relative_l2'],flush=True)
        rows.append(transient_fdm(dt))
    (out/'evolution.json').write_text(json.dumps(rows,indent=2))
