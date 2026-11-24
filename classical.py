"""Independent P1 finite elements and RT0 functional majorant reconstruction."""
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from acpinn import gauss_grid


def fem(problem,n=32):
    tic=time.perf_counter()
    grid=np.linspace(0,1,n+1)
    pts=np.stack(np.meshgrid(grid,grid,indexing="ij"),axis=-1).reshape(-1,2)
    triangles=[]
    for i in range(n):
        for j in range(n):
            p=i*(n+1)+j
            triangles.extend([[p,p+n+1,p+1],[p+n+1,p+n+2,p+1]])
    tri=np.asarray(triangles); coords=pts[tri]
    # Seven-point Dunavant rule, weights normalized to one.
    a=.059715871789770;b=.470142064105115;c=.797426985353087;d=.101286507323456
    bary=np.array([[1/3]*3,[a,b,b],[b,a,b],[b,b,a],[c,d,d],[d,c,d],[d,d,c]])
    qw=np.array([.225]+[.132394152788506]*3+[.125939180544827]*3)
    qx=np.einsum('qa,tad->tqd',bary,coords)
    bmat=np.stack([coords[:,1]-coords[:,0],coords[:,2]-coords[:,0]],axis=-1)
    inv=np.linalg.inv(bmat)
    ref=np.array([[-1,-1],[1,0],[0,1]])
    grad=np.einsum('ai,tij->taj',ref,inv)
    area=np.abs(np.linalg.det(bmat))/2
    av=problem.coefficient(qx.reshape(-1,2)).reshape(len(tri),7)@qw
    loc=area[:,None,None]*av[:,None,None]*np.einsum('tad,tbd->tab',grad,grad)
    row=np.repeat(tri,3,axis=1).ravel();col=np.tile(tri,(1,3)).ravel()
    k=sparse.coo_matrix((loc.ravel(),(row,col)),shape=(len(pts),len(pts))).tocsr()
    ff=problem.forcing(qx.reshape(-1,2)).reshape(len(tri),7)
    floc=area[:,None]*np.einsum('q,tq,qa->ta',qw,ff,bary)
    rhs=np.bincount(tri.ravel(),weights=floc.ravel(),minlength=len(pts))
    boundary=np.any((pts==0)|(pts==1),axis=1)
    interior=np.flatnonzero(~boundary);u=np.zeros(len(pts))
    u[interior]=spsolve(k[interior][:,interior],rhs[interior])
    elapsed=time.perf_counter()-tic
    uv=np.einsum('ta,qa->tq',u[tri],bary)
    ug=np.einsum('ta,tad->td',u[tri],grad)
    ue,ge=problem.exact(qx.reshape(-1,2));ue=ue.reshape(len(tri),7);ge=ge.reshape(len(tri),7,2)
    w=area[:,None]*qw
    aa=problem.coefficient(qx.reshape(-1,2)).reshape(len(tri),7)
    return {"method":"P1 FEM","size":len(interior),"seconds":elapsed,
            "relative_l2":float(np.sqrt(np.sum(w*(uv-ue)**2)/np.sum(w*ue**2))),
            "relative_energy":float(np.sqrt(np.sum(w*aa*np.sum((ug[:,None,:]-ge)**2,axis=2))/np.sum(w*aa*np.sum(ge**2,axis=2))))}


def functional_majorant(dictionary,ids,coefficients,problem,n=24,order=4):
    """RT0 flux; the returned value is quadrature-evaluated, not a certificate."""
    tic=time.perf_counter();h=1/n
    q,w=np.polynomial.legendre.leggauss(order);t=(q+1)/2;w=w/2
    tx,ty=np.meshgrid(t,t,indexing="ij");wx,wy=np.meshgrid(w,w,indexing="ij")
    local=np.column_stack([tx.ravel(),ty.ravel()]);qw=(wx*wy).ravel()
    ij=np.stack(np.meshgrid(np.arange(n),np.arange(n),indexing="ij"),axis=-1).reshape(-1,2)
    x=(ij[:,None,:]+local[None,:,:])*h
    z,jac=problem.mapping(x.reshape(-1,2))
    v,g,_=dictionary.evaluate(z,ids);g*=jac[:,None,:]
    gv=np.einsum('qnd,n->qd',g,coefficients).reshape(n*n,-1,2)
    aa=problem.coefficient(x.reshape(-1,2)).reshape(n*n,-1)
    ff=problem.forcing(x.reshape(-1,2)).reshape(n*n,-1)
    nf=(n+1)*n;total=2*nf
    i,j=ij[:,0],ij[:,1]
    faces=np.column_stack([i*n+j,(i+1)*n+j,nf+i*(n+1)+j,nf+i*(n+1)+j+1])
    basis=np.zeros((len(qw),4,2))
    basis[:,0,0]=1-local[:,0];basis[:,1,0]=local[:,0]
    basis[:,2,1]=1-local[:,1];basis[:,3,1]=local[:,1]
    ml=h*h*np.einsum('q,tq,qad,qbd->tab',qw,1/aa,basis,basis)
    b=h*h*np.einsum('q,tqd,qad->ta',qw,gv,basis)
    row=np.repeat(faces,4,axis=1).ravel();col=np.tile(faces,(1,4)).ravel()
    m=sparse.coo_matrix((ml.ravel(),(row,col)),shape=(total,total)).tocsr()
    bv=np.bincount(faces.ravel(),weights=b.ravel(),minlength=total)
    signs=np.array([-1,1,-1,1])/h
    div=sparse.coo_matrix((np.tile(signs,n*n),(np.repeat(np.arange(n*n),4),faces.ravel())),shape=(n*n,total)).tocsr()
    fmean=ff@qw
    cp=1/(np.pi*np.sqrt(2));amin=1 if problem.name=="poisson" else min(1.,problem.contrast)
    beta=1.;yp=None
    for _ in range(4):
        al=1+beta;gam=(1+1/beta)*cp*cp/amin
        y=spsolve(al*m+gam*h*h*(div.T@div),al*bv-gam*h*h*(div.T@fmean))
        yg=np.einsum('ta,qad->tqd',y[faces],basis)
        defect=float(np.sqrt(h*h*np.sum(qw[None,:]/aa*np.sum((yg-aa[:,:,None]*gv)**2,axis=2))))
        equilibrium=float(cp/np.sqrt(amin)*np.sqrt(h*h*np.sum(qw[None,:]*(ff+(div@y)[:,None])**2)))
        beta=max(1.e-6,min(1.e6,equilibrium/max(defect,1.e-30)))
    return {"majorant":defect+equilibrium,"flux_defect":defect,"equilibrium":equilibrium,
            "flux_dofs":total,"seconds":time.perf_counter()-tic}
