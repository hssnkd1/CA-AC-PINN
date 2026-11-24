"""Manufactured fields and derivatives, independent of network implementation."""
import numpy as np


def bump(x,center=(.62,.38),sigma=.07,amplitude=.2):
    xx,yy=x[:,0],x[:,1]
    qx=4*xx*(1-xx); qy=4*yy*(1-yy)
    bx=4-8*xx; by=4-8*yy
    b=qx*qy
    db=np.column_stack([bx*qy,qx*by])
    ddb=np.column_stack([-8*qy,-8*qx])
    dx=x-np.asarray(center)
    h=amplitude*np.exp(-np.sum(dx*dx,axis=1)/sigma**2)
    dh=-2*dx/sigma**2*h[:,None]
    ddh=(4*dx*dx/sigma**4-2/sigma**2)*h[:,None]
    return b*h, db*h[:,None]+b[:,None]*dh, ddb*h[:,None]+2*db*dh+b[:,None]*ddh


class Poisson:
    name="poisson"
    def mapping(self,x): return x,np.ones_like(x)
    def coefficient(self,x):return np.ones(len(x))
    def coefficient_gradient(self,x):return np.zeros_like(x)
    def exact(self,x):
        xx,yy=x[:,0],x[:,1]
        u=np.sin(np.pi*xx)*np.sin(np.pi*yy)
        g=np.pi*np.column_stack([np.cos(np.pi*xx)*np.sin(np.pi*yy),np.sin(np.pi*xx)*np.cos(np.pi*yy)])
        return u,g
    def forcing(self,x):return 2*np.pi**2*self.exact(x)[0]


class Interface:
    name="interface"
    contrast=10.
    def __init__(self,warped=True):
        self.warped=warped
        self.s=.5+.5/self.contrast
        self.c=(.125+.375/self.contrast)/self.s
    def coefficient(self,x):return np.where(x[:,0]<=.5,1.,self.contrast)
    def coefficient_gradient(self,x):return np.zeros_like(x)
    def canonical(self,x):
        z=x.copy()
        z[:,0]=np.where(x[:,0]<=.5,x[:,0],.5+(x[:,0]-.5)/self.contrast)/self.s
        jac=np.ones_like(x);jac[:,0]=1/(self.s*self.coefficient(x))
        return z,jac
    def mapping(self,x):return self.canonical(x) if self.warped else (x,np.ones_like(x))
    def field(self,x):
        xx,yy=x[:,0],x[:,1];a=self.coefficient(x)
        u1=np.where(xx<=.5,self.c*xx-xx**2/2,
                    self.c*.5-.125+(self.c*(xx-.5)-(xx**2-.25)/2)/self.contrast)
        du=(self.c-xx)/a
        s=np.sin(np.pi*yy);cy=np.cos(np.pi*yy)
        u=u1*s
        g=np.column_stack([du*s,np.pi*u1*cy])
        d2=np.column_stack([-s/a,-np.pi**2*u1*s])
        z,jac=self.canonical(x)
        v,gv,hv=bump(z,center=(.46,.43),sigma=.07,amplitude=.22)
        return u+v,g+gv*jac,d2+hv*jac**2
    def exact(self,x):u,g,_=self.field(x);return u,g
    def forcing(self,x):return -self.coefficient(x)*np.sum(self.field(x)[2],axis=1)


class Transient:
    name="transient"
    def __init__(self,t=0.):self.t=t
    def mapping(self,x):return x,np.ones_like(x)
    def coefficient(self,x):return 1+.5*np.sin(2*np.pi*x[:,0])*np.sin(2*np.pi*x[:,1])
    def coefficient_gradient(self,x):
        return np.pi*np.column_stack([np.cos(2*np.pi*x[:,0])*np.sin(2*np.pi*x[:,1]),np.sin(2*np.pi*x[:,0])*np.cos(2*np.pi*x[:,1])])
    def field0(self,x):
        xx,yy=x[:,0],x[:,1]
        u=np.sin(np.pi*xx)*np.sin(np.pi*yy)
        g=np.pi*np.column_stack([np.cos(np.pi*xx)*np.sin(np.pi*yy),np.sin(np.pi*xx)*np.cos(np.pi*yy)])
        h=-np.pi**2*np.column_stack([u,u])
        b,bg,bh=bump(x,center=(.65,.35),sigma=.07,amplitude=.3)
        return u+b,g+bg,h+bh
    def exact(self,x):u,g,_=self.field0(x);return np.exp(-self.t)*u,np.exp(-self.t)*g
    def forcing(self,x):
        u,g,h=self.field0(x)
        return np.exp(-self.t)*(-u-self.coefficient(x)*np.sum(h,axis=1)-np.sum(self.coefficient_gradient(x)*g,axis=1))
