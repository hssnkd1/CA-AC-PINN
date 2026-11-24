"""CA-AC-PINN v2 experimental core.

PDE-only local radius selection, Schur-condition guard and cost-aware block
ranking. Elliptic symmetric problems only; no unproved nonlinear certificate.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from acpinn import CircularDictionary, HierarchicalRitz


def inverse_coordinates(problem, z):
    """Physical x corresponding to dictionary z for built-in fitted interface."""
    x=np.array(z,copy=True)
    if hasattr(problem,'canonical') and getattr(problem,'warped',False):
        s=problem.s
        x[:,0]=np.where(z[:,0]*s<=.5,z[:,0]*s,.5+problem.contrast*(z[:,0]*s-.5))
    return x


def pde_local_radii(dictionary, problem, base_scale=1.5, strength=0.25,
                    min_scale=0.85,max_scale=2.0,probe=0.008):
    """Construct local radius using a normalized *PDE forcing gradient*.

    All inputs use problem.forcing; exact solution and error are never used.
    Forcing may be discontinuous: restrict probes to same diffusion subdomain.
    The returned diagnostic is sufficient to audit the scale choice.
    """
    if not (base_scale>0 and 0<=strength<=1 and min_scale>0 and max_scale>=min_scale):
        raise ValueError('Invalid scale settings')
    z=dictionary.centers
    x=inverse_coordinates(problem,z)
    f0=problem.forcing(x)
    g2=np.zeros(len(x))
    for axis in range(dictionary.dim):
        left=x.copy();right=x.copy()
        left[:,axis]=np.maximum(probe,x[:,axis]-probe)
        right[:,axis]=np.minimum(1-probe,x[:,axis]+probe)
        # When a material jump is present, one-sided probes stay on same side.
        if axis==0 and hasattr(problem,'contrast'):
            left[:,axis]=np.where(x[:,axis]>.5,np.maximum(left[:,axis],.5+1e-9),np.minimum(left[:,axis],.5-1e-9))
            right[:,axis]=np.where(x[:,axis]>.5,np.maximum(right[:,axis],.5+1e-9),np.minimum(right[:,axis],.5-1e-9))
        g=(problem.forcing(right)-problem.forcing(left))/np.maximum(right[:,axis]-left[:,axis],1e-12)
        g2+=g*g
    magnitude=np.sqrt(g2)
    # Robust normalization prevents an extreme forcing point setting all radii.
    ref=max(float(np.quantile(magnitude,.9)),1e-12)
    indicator=np.clip(magnitude/ref,0.,1.)
    # Sharper forcing variation -> slightly smaller circle, smooth region -> larger circle.
    scales=np.clip(base_scale*(1+strength*(.5-indicator)),min_scale,max_scale)
    dictionary.radii*=scales
    return {'radius_min':float(np.min(dictionary.radii)), 'radius_max':float(np.max(dictionary.radii)),
            'scale_min':float(scales.min()),'scale_max':float(scales.max()),
            'scale_mean':float(scales.mean()),'indicator_p90':float(np.quantile(indicator,.9)),
            'scales':scales.tolist()}


class ConditionAwareRitz(HierarchicalRitz):
    """Quadrature-energy Schur selection with optional relative pivot guard.

    For all children in a block, the smallest eigenvalue of the diagonal-
    normalized Schur complement must exceed min_schur_eig. This is a local
    redundancy diagnostic; it does NOT certify a global condition number.
    """
    def __init__(self,dictionary,matrix,rhs,min_schur_eig=1e-7,cost_power=0.):
        super().__init__(dictionary,matrix,rhs)
        self.min_schur_eig=float(min_schur_eig)
        self.cost_power=float(cost_power)
        self.rejected_blocks=0
        self.checked_blocks=0

    def score(self,children):
        s=self.k[np.ix_(children,children)]-self.l[children]@self.l[children].T
        s=(s+s.T)*.5
        diag=np.diag(s)
        self.checked_blocks+=1
        if np.any(diag<=0) or np.any(~np.isfinite(s)):
            self.rejected_blocks+=1
            return -np.inf
        d=np.sqrt(diag)
        eig=np.linalg.eigvalsh(s/d[:,None]/d[None,:])
        if eig[0]<self.min_schur_eig:
            self.rejected_blocks+=1
            return -np.inf
        try:
            c=cho_factor(s,lower=True,check_finite=False)
            gain=self.res[children]@cho_solve(c,self.res[children],check_finite=False)
        except np.linalg.LinAlgError:
            self.rejected_blocks+=1
            return -np.inf
        # Approximate integration footprint of the proposed feature supports.
        footprint=np.sum(self.d.radii[np.array(children)-self.d.ng]**self.d.dim)
        return float(gain/(max(footprint,1e-12)**self.cost_power))


def assemble_sparse(dictionary,problem,n=24,order=3):
    """Assemble sparse CSR trial Gram matrices without keeping sample matrices.

    This is an exact sparsity-preserving algebraic assembly on the supplied
    quadrature rule. The present implementation still evaluates every trial
    feature at all points and must convert to dense for the legacy Ritz solver.
    """
    from scipy import sparse
    from acpinn import gauss_grid
    import time
    start=time.perf_counter();x,w=gauss_grid(n,order,dictionary.dim)
    k=sparse.csr_matrix((dictionary.size,dictionary.size));m=k.copy()
    f=np.zeros(dictionary.size)
    for start_index in range(0,len(x),1024):
        xb=x[start_index:start_index+1024];wb=w[start_index:start_index+1024]
        z,jac=problem.mapping(xb);v,g,_=dictionary.evaluate(z)
        g*=jac[:,None,:];a=problem.coefficient(xb)
        vv=sparse.csr_matrix(v)
        m=m+vv.T@vv.multiply(wb[:,None])
        for axis in range(dictionary.dim):
            gg=sparse.csr_matrix(g[:,:,axis]);k=k+gg.T@gg.multiply((wb*a)[:,None])
        f+=v.T@(wb*problem.forcing(xb))
    return {'stiffness':k.tocsr(),'mass':m.tocsr(),'rhs':f,'seconds':time.perf_counter()-start,
            'stiffness_nnz':int(k.nnz),'mass_nnz':int(m.nnz)}


def choose_by_energy(rows,condition_cap=1.e6):
    """Select using PDE Ritz energy and stability only, NOT true errors."""
    eligible=[r for r in rows if 'failure' not in r and r['scaled_condition']<=condition_cap]
    if not eligible:return None
    return min(eligible,key=lambda r:r['energy'])
