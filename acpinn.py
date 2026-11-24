"""Compact circular Ritz networks and residual-guided hierarchical enrichment.

The numerical implementation uses quadrature, not interval-certified integration.
All scores and monotonicity checks therefore refer to the quadrature problem.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.special import eval_legendre


def gauss_grid(n=32, order=3, dim=2):
    z, w = np.polynomial.legendre.leggauss(order)
    xx = ((np.arange(n)[:, None] + (z + 1) / 2) / n).ravel()
    ww = np.tile(w / (2 * n), n)
    if dim == 1:
        return xx[:, None], ww
    x, y = np.meshgrid(xx, xx, indexing="ij")
    wx, wy = np.meshgrid(ww, ww, indexing="ij")
    return np.column_stack([x.ravel(), y.ravel()]), (wx * wy).ravel()


def wendland(points, centers, radii, dim=2):
    """Value, gradient and diagonal Hessian of (1-s)^4_+ (4s+1)."""
    d = points[:, None, :] - centers[None, :, :]
    s = np.linalg.norm(d, axis=2) / radii
    t = np.maximum(1-s, 0)
    val = t**4 * (4*s+1)
    grad = -20*t[:, :, None]**3*d / radii[None, :, None]**2
    # The apparent 1/s factor is removable because d_l^2/s -> 0 at the centre.
    diag = -20*t[:, :, None]**3 / radii[None, :, None]**2
    diag = np.broadcast_to(diag, d.shape).copy()
    diag += 60*t[:, :, None]**2*d*d / (
        radii[None, :, None]**4*np.maximum(s[:, :, None], 1.e-30))
    return val, grad, diag


def boundary_factor(x):
    q = 4*x*(1-x)
    qp = 4-8*x
    qpp = np.full_like(x, -8.)
    if x.shape[1] == 1:
        return q[:, 0], qp, qpp
    b = q[:, 0]*q[:, 1]
    db = np.column_stack([qp[:, 0]*q[:, 1], q[:, 0]*qp[:, 1]])
    ddb = np.column_stack([qpp[:, 0]*q[:, 1], q[:, 0]*qpp[:, 1]])
    return b, db, ddb


class CircularDictionary:
    """Dyadic balls, retaining parents, plus a small global polynomial span."""
    def __init__(self, dim=2, depth=3):
        self.dim, self.depth = dim, depth
        self.powers = [(i, j) for i in range(3) for j in range(3) if (i,j)!=(2,2)] if dim==2 else [(i,) for i in range(3)]
        self.ng = len(self.powers)
        centers, radii, levels = [], [], []
        self.offsets, self.ns = [], []
        for level in range(depth+1):
            n = 3*2**level
            self.offsets.append(self.ng+len(centers)); self.ns.append(n)
            grid = (np.arange(n)+.5)/n
            pts = np.stack(np.meshgrid(*([grid]*dim), indexing="ij"), axis=-1).reshape(-1,dim)
            centers.extend(pts); radii.extend([2/n]*len(pts)); levels.extend([level]*len(pts))
        self.centers = np.asarray(centers); self.radii = np.asarray(radii)
        self.levels = np.asarray(levels)
        self.size = self.ng+len(centers)
        self.children = {}
        for level in range(depth):
            n = self.ns[level]
            for ij in np.ndindex(*([n]*dim)):
                parent = self.offsets[level]+np.ravel_multi_index(ij, (n,)*dim)
                cc = []
                for off in np.ndindex(*([2]*dim)):
                    kk = tuple(2*ij[k]+off[k] for k in range(dim))
                    cc.append(self.offsets[level+1]+np.ravel_multi_index(kk, (2*n,)*dim))
                self.children[parent] = cc
        self.initial = list(range(self.ng+3**dim))

    def evaluate(self, x, indices=None):
        ids = list(range(self.size)) if indices is None else list(indices)
        b, db, ddb = boundary_factor(x)
        v = np.zeros((len(x),len(ids)))
        g = np.zeros((len(x),len(ids),self.dim)); h = np.zeros_like(g)
        ci = [k for k,i in enumerate(ids) if i>=self.ng]
        if ci:
            ii = np.array([ids[k]-self.ng for k in ci])
            vv,gg,hh = wendland(x,self.centers[ii],self.radii[ii],self.dim)
            v[:,ci] = b[:,None]*vv
            g[:,ci,:] = db[:,None,:]*vv[:,:,None]+b[:,None,None]*gg
            h[:,ci,:] = ddb[:,None,:]*vv[:,:,None]+2*db[:,None,:]*gg+b[:,None,None]*hh
        for k,i in enumerate(ids):
            if i>=self.ng: continue
            p = self.powers[i]
            factors=[]; dfs=[]; ddfs=[]
            for axis,degree in enumerate(p):
                coeff=np.zeros(degree+1); coeff[-1]=1
                leg=np.polynomial.legendre.Legendre(coeff)
                factors.append(leg(2*x[:,axis]-1))
                dfs.append(2*leg.deriv(1)(2*x[:,axis]-1))
                ddfs.append(4*leg.deriv(2)(2*x[:,axis]-1))
            pv=np.prod(factors,axis=0)
            pg=np.column_stack([dfs[a]*np.prod([factors[c] for c in range(self.dim) if c!=a],axis=0) if self.dim>1 else dfs[a] for a in range(self.dim)])
            ph=np.column_stack([ddfs[a]*np.prod([factors[c] for c in range(self.dim) if c!=a],axis=0) if self.dim>1 else ddfs[a] for a in range(self.dim)])
            v[:,k]=b*pv; g[:,k,:]=db*pv[:,None]+b[:,None]*pg
            h[:,k,:]=ddb*pv[:,None]+2*db*pg+b[:,None]*ph
        return v,g,h

    def predict(self,x,indices,coefficients):
        """Value-only inference, without constructing derivative arrays."""
        ids=list(indices);out=np.zeros(len(x));b,_,_=boundary_factor(x)
        for k,i in enumerate(ids):
            if i<self.ng:
                factors=[eval_legendre(degree,2*x[:,a]-1) for a,degree in enumerate(self.powers[i])]
                value=np.prod(factors,axis=0)
            else:
                j=i-self.ng;s=np.linalg.norm(x-self.centers[j],axis=1)/self.radii[j]
                value=np.maximum(1-s,0)**4*(4*s+1)
            out+=coefficients[k]*b*value
        return out


def weighted_gram(v, w):
    z=sparse.csr_matrix(v)
    return (z.T @ z.multiply(w[:,None])).toarray()


@dataclass
class LinearSystem:
    dictionary: CircularDictionary
    points: np.ndarray
    weights: np.ndarray
    values: np.ndarray
    gradients: np.ndarray
    stiffness: np.ndarray
    mass: np.ndarray
    rhs: np.ndarray
    assembly_seconds: float


def assemble(dictionary, problem, n=32, order=3, keep_samples=True):
    tic=time.perf_counter()
    x,w=gauss_grid(n,order,dictionary.dim)
    k=np.zeros((dictionary.size,dictionary.size))
    m=np.zeros_like(k);f=np.zeros(dictionary.size);vs=[];gs=[]
    for start in range(0,len(x),2048):
        xb=x[start:start+2048];wb=w[start:start+2048]
        z,jac=problem.mapping(xb);v,g,_=dictionary.evaluate(z)
        g*=jac[:,None,:];a=problem.coefficient(xb)
        for axis in range(dictionary.dim):k+=weighted_gram(g[:,:,axis],wb*a)
        m+=weighted_gram(v,wb);f+=v.T@(wb*problem.forcing(xb))
        if keep_samples:vs.append(v);gs.append(g)
    v=np.concatenate(vs) if keep_samples else None
    g=np.concatenate(gs) if keep_samples else None
    return LinearSystem(dictionary,x,w,v,g,k,m,f,time.perf_counter()-tic)


class HierarchicalRitz:
    """Exact block residual scores for a fixed, quadrature-defined SPD form."""
    def __init__(self, dictionary, matrix, rhs, initial=None):
        self.d=dictionary; self.k=matrix; self.f=rhs
        self.ids=[]; self.split=set()
        self.l=np.zeros((len(rhs),0)); self.res=rhs.copy()
        self.history=[]
        for p in (dictionary.initial if initial is None else initial): self._append(p)

    def _append(self,p):
        cross=self.k[:,p]-self.l@self.l[p,:]
        s=cross[p]
        if s<=1.e-12*max(self.k[p,p],1.e-30):
            raise np.linalg.LinAlgError(f"Dependent feature {p}: pivot {s}")
        col=cross/np.sqrt(s)
        self.res-=col*(self.res[p]/np.sqrt(s))
        self.l=np.column_stack([self.l,col]); self.ids.append(int(p))

    def score(self,children):
        s=self.k[np.ix_(children,children)]-self.l[children]@self.l[children].T
        s=(s+s.T)/2
        try:
            c=cho_factor(s,lower=True,check_finite=False)
            score=self.res[children]@cho_solve(c,self.res[children],check_finite=False)
        except np.linalg.LinAlgError:
            score=-np.inf
        return score

    def grow(self,target,callback=None,max_splits=None):
        count=0
        while len(self.ids)+2**self.d.dim<=target:
            cand=[p for p in self.ids if p in self.d.children and p not in self.split]
            if not cand:break
            scores=[self.score(self.d.children[p]) for p in cand]
            j=int(np.argmax(scores)); p=cand[j]
            if not np.isfinite(scores[j]):break
            before=self.energy()
            for q in self.d.children[p]: self._append(q)
            self.split.add(p)
            after=self.energy()
            self.history.append({"size":len(self.ids),"parent":int(p),"gain":float(scores[j]),"energy":after,"identity_defect":float(abs((before-after)*2-scores[j]))})
            if callback is not None: callback(self)
            count+=1
            if max_splits is not None and count>=max_splits:break
        return self

    def coefficients(self):
        a=self.k[np.ix_(self.ids,self.ids)]
        return cho_solve(cho_factor(a,lower=True,check_finite=False),self.f[self.ids],check_finite=False)

    def energy(self):
        c=self.coefficients()
        return float(-.5*c@self.f[self.ids])


def metrics(dictionary, ids, c, problem, n=80, order=3):
    x,w=gauss_grid(n,order,dictionary.dim)
    l2e=l2u=ene=enu=maxe=0.
    for start in range(0,len(x),2048):
        xb=x[start:start+2048];wb=w[start:start+2048]
        z,jac=problem.mapping(xb);v,g,_=dictionary.evaluate(z,ids);g*=jac[:,None,:]
        u=v@c;grad=np.einsum('qnd,n->qd',g,c);ue,ge=problem.exact(xb)
        err=u-ue;eg=grad-ge;a=problem.coefficient(xb)
        l2e+=np.sum(wb*err**2);l2u+=np.sum(wb*ue**2)
        ene+=np.sum(wb*a*np.sum(eg**2,axis=1));enu+=np.sum(wb*a*np.sum(ge**2,axis=1))
        maxe=max(maxe,np.max(np.abs(err)))
    return {"relative_l2":float(np.sqrt(l2e/l2u)),"relative_energy":float(np.sqrt(ene/enu)),
            "energy_error":float(np.sqrt(ene)),"sample_max":float(maxe)}


def fixed_circle_indices(dictionary,n=11,radius=None):
    # A separate fixed uniform network, same global span and 11^2=121 circles.
    d=CircularDictionary(dictionary.dim,depth=0)
    grid=(np.arange(n)+.5)/n
    d.centers=np.stack(np.meshgrid(*([grid]*d.dim),indexing="ij"),axis=-1).reshape(-1,d.dim)
    d.radii=np.full(len(d.centers),2/n if radius is None else radius)
    d.levels=np.zeros(len(d.centers),int)
    d.size=d.ng+len(d.centers);d.children={};d.initial=list(range(d.size))
    return d


def prune_by_energy(matrix,rhs,ids,tolerance,protected=8):
    """Accept removals only if cumulative quadrature energy growth stays bounded."""
    ids=list(ids)
    def solve(ii):
        a=matrix[np.ix_(ii,ii)]
        cc=cho_solve(cho_factor(a,lower=True,check_finite=False),rhs[ii],check_finite=False)
        return cc,float(-.5*cc@rhs[ii])
    c,j0=solve(ids); initial_j=j0; removed=[]
    while len(ids)>protected:
        k=matrix[np.ix_(ids,ids)]
        inv=cho_solve(cho_factor(k,lower=True,check_finite=False),np.eye(len(ids)),check_finite=False)
        costs=.5*c*c/np.diag(inv)
        costs[:protected]=np.inf
        best=int(np.argmin(costs))
        if not np.isfinite(costs[best]):break
        trial=ids[:best]+ids[best+1:]
        tc,tj=solve(trial)
        if tj-initial_j>tolerance:break
        removed.append(ids[best]);ids,c=trial,tc
    return ids,c,{"removed":len(removed),"accepted_growth":float(-.5*c@rhs[ids]-initial_j)}
