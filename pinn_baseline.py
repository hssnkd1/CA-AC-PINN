"""One-hidden-layer tanh PINNs with exact parameter derivatives of residuals.

Gradient-enhanced PINN uses symmetric finite differences of the strong residual
in physical coordinates. It is explicitly a gPINN implementation variant.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from acpinn import boundary_factor, gauss_grid


def residual_and_jac(theta,x,problem,hidden=32,need_jac=True):
    z,jmap=problem.mapping(x)
    w=theta[:hidden*2].reshape(hidden,2)
    bias=theta[hidden*2:hidden*3]
    out=theta[hidden*3:hidden*4];off=theta[-1]
    hh=np.tanh(z@w.T+bias)
    h1=1-hh*hh;h2=-2*hh*h1;h3=-2*h1*h1+4*hh*hh*h1
    b,db,ddb=boundary_factor(z)
    a=problem.coefficient(x);ag=problem.coefficient_gradient(x)
    D=-a*np.sum(ddb*jmap*jmap,axis=1)-np.sum(ag*db*jmap,axis=1)
    E=-2*a[:,None]*db*jmap*jmap-b[:,None]*ag*jmap
    H=-a[:,None]*b[:,None]*jmap*jmap
    ew=E@w.T;hw=H@(w*w).T
    feature=D[:,None]*hh+h1*ew+h2*hw
    residual=feature@out+D*off-problem.forcing(x)
    if not need_jac:return residual
    j=np.empty((len(x),4*hidden+1))
    common=D[:,None]*h1+h2*ew+h3*hw
    for axis in range(2):
        jw=common*z[:,axis,None]+h1*E[:,axis,None]+2*h2*H[:,axis,None]*w[:,axis]
        j[:,axis:hidden*2:2]=jw*out
    j[:,hidden*2:hidden*3]=common*out
    j[:,hidden*3:hidden*4]=feature;j[:,-1]=D
    return residual,j


def predict(theta,x,problem,hidden=32):
    z,jmap=problem.mapping(x)
    w=theta[:hidden*2].reshape(hidden,2);bias=theta[hidden*2:hidden*3]
    out=theta[hidden*3:hidden*4];off=theta[-1]
    h=np.tanh(z@w.T+bias);h1=1-h*h
    b,db,_=boundary_factor(z)
    f=h@out+off;df=(h1*out)@w
    return b*f,(db*f[:,None]+b[:,None]*df)*jmap


def train(problem,seed=0,kind="standard",hidden=32,iterations=1500):
    rng=np.random.default_rng(seed)
    theta=np.r_[rng.normal(0,2.0,(hidden,2)).ravel(),rng.normal(0,.6,hidden),rng.normal(0,.15,hidden),0.]
    size=10 if kind in ("residual_adaptive","gradient_enhanced") else 11
    x=qmc.Sobol(2,scramble=True,seed=seed).random_base2(size)
    norm=max(float(np.mean(problem.forcing(x)**2)),1.)
    tic=time.perf_counter();history=[];nfev=0
    stages=3 if kind=="residual_adaptive" else 1
    for stage in range(stages):
        def objective(t):
            r,j=residual_and_jac(t,x,problem,hidden)
            loss=np.mean(r*r);grad=2*(j.T@r)/len(r)
            if kind=="gradient_enhanced":
                delta=1.e-4;weight=1.e-3
                for axis in range(2):
                    xp=x.copy();xm=x.copy();xp[:,axis]+=delta;xm[:,axis]-=delta
                    # Do not difference across a material interface.
                    keep=(xm[:,axis]>0)&(xp[:,axis]<1)
                    if problem.name=="interface" and axis==0:keep&=(np.abs(x[:,0]-.5)>2*delta)
                    rp,jp=residual_and_jac(t,xp[keep],problem,hidden)
                    rm,jm=residual_and_jac(t,xm[keep],problem,hidden)
                    dr=(rp-rm)/(2*delta);dj=(jp-jm)/(2*delta)
                    loss+=weight*np.mean(dr*dr);grad+=weight*2*(dj.T@dr)/len(dr)
            return loss/norm,grad/norm
        opt=minimize(objective,theta,method="L-BFGS-B",jac=True,options={"maxiter":iterations//stages,"maxls":30,"ftol":1.e-14,"gtol":1.e-10,"maxcor":30})
        theta=opt.x;nfev+=opt.nfev
        history.append({"stage":stage,"iterations":int(opt.nit),"loss":float(opt.fun),"message":str(opt.message),"points":len(x)})
        if kind=="residual_adaptive" and stage<stages-1:
            pool=qmc.Sobol(2,scramble=True,seed=100+seed+stage).random_base2(12)
            rr=np.abs(residual_and_jac(theta,pool,problem,hidden,False))
            x=np.vstack([x,pool[np.argsort(rr)[-256:]]])
    seconds=time.perf_counter()-tic
    xx,ww=gauss_grid(80,3,2);u,g=predict(theta,xx,problem,hidden)
    ue,ge=problem.exact(xx);a=problem.coefficient(xx)
    row={"method":{"standard":"tanh PINN","residual_adaptive":"RAR tanh PINN","gradient_enhanced":"FD gPINN"}[kind],
         "seed":seed,"size":len(theta),"seconds":seconds,"nfev":nfev,"training":history,
         "relative_l2":float(np.sqrt(np.sum(ww*(u-ue)**2)/np.sum(ww*ue**2))),
         "relative_energy":float(np.sqrt(np.sum(ww*a*np.sum((g-ge)**2,axis=1))/np.sum(ww*a*np.sum(ge**2,axis=1))))}
    return row,theta
