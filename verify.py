"""Focused mathematical and numerical checks; run with python verify.py."""
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from acpinn import CircularDictionary,assemble,HierarchicalRitz
from problems import Poisson,Interface
from pinn_baseline import residual_and_jac


class Verification(unittest.TestCase):
    def test_basis_derivatives(self):
        d=CircularDictionary(2,1);x=np.array([[.31,.44],[.65,.27],[.51,.62]])
        ids=[0,3,8,15,25];v,g,h=d.evaluate(x,ids);eps=1.e-5
        for a in range(2):
            xp=x.copy();xm=x.copy();xp[:,a]+=eps;xm[:,a]-=eps
            vp,gp,_=d.evaluate(xp,ids);vm,gm,_=d.evaluate(xm,ids)
            np.testing.assert_allclose((vp-vm)/(2*eps),g[:,:,a],rtol=2.e-6,atol=1.e-7)
            np.testing.assert_allclose((gp[:,:,a]-gm[:,:,a])/(2*eps),h[:,:,a],rtol=2.e-6,atol=1.e-6)

    def test_exact_trace(self):
        d=CircularDictionary(2,1);x=np.array([[0,.4],[1,.2],[.3,0],[.8,1]])
        v,_,_=d.evaluate(x);self.assertEqual(np.max(np.abs(v)),0.)

    def test_pinn_jacobian(self):
        rng=np.random.default_rng(9);th=rng.normal(0,.5,17);x=rng.uniform(.1,.9,(7,2))
        r,j=residual_and_jac(th,x,Poisson(),hidden=4);eps=1.e-6
        for k in range(len(th)):
            tp=th.copy();tm=th.copy();tp[k]+=eps;tm[k]-=eps
            rp=residual_and_jac(tp,x,Poisson(),hidden=4,need_jac=False)
            rm=residual_and_jac(tm,x,Poisson(),hidden=4,need_jac=False)
            np.testing.assert_allclose((rp-rm)/(2*eps),j[:,k],rtol=1.e-6,atol=1.e-7)

    def test_interface_flux(self):
        p=Interface();y=np.linspace(.05,.95,30);eps=1.e-9
        l=np.column_stack([np.full(len(y),.5-eps),y]);r=l.copy();r[:,0]=.5+eps
        ul,gl=p.exact(l);ur,gr=p.exact(r)
        np.testing.assert_allclose(ul,ur,atol=5.e-9)
        np.testing.assert_allclose(p.coefficient(l)*gl[:,0],p.coefficient(r)*gr[:,0],atol=1.e-8)

    def test_ritz_gain(self):
        d=CircularDictionary(2,2);s=assemble(d,Poisson(),n=8,order=4)
        m=HierarchicalRitz(d,s.stiffness,s.rhs);m.grow(33)
        self.assertLess(max(q['identity_defect'] for q in m.history),1.e-9)
        self.assertTrue(all(q['gain']>=0 for q in m.history))
        c=m.coefficients();r=s.stiffness[np.ix_(m.ids,m.ids)]@c-s.rhs[m.ids]
        self.assertLess(np.linalg.norm(r),1.e-10)

    def test_manufactured_forcing(self):
        from problems import Transient
        x=np.array([[.24,.43],[.67,.35],[.81,.71]])
        eps=1.e-5
        for p in [Poisson(),Interface(),Transient()]:
            div=np.zeros(len(x))
            for axis in range(2):
                xp=x.copy();xm=x.copy();xp[:,axis]+=eps;xm[:,axis]-=eps
                gp=p.exact(xp)[1][:,axis]*p.coefficient(xp)
                gm=p.exact(xm)[1][:,axis]*p.coefficient(xm)
                div+=(gp-gm)/(2*eps)
            expected=-div-(p.exact(x)[0] if p.name=='transient' else 0)
            np.testing.assert_allclose(p.forcing(x),expected,rtol=3.e-6,atol=2.e-6)

    def test_burgers_travelling_wave(self):
        from evolution import front
        x=np.linspace(.3,.5,41);t=.1;eps=1.e-6
        u,ux,_=front(x,t,'burgers')
        ut=(front(x,t+eps,'burgers')[0]-front(x,t-eps,'burgers')[0])/(2*eps)
        uxx=(front(x+eps,t,'burgers')[1]-front(x-eps,t,'burgers')[1])/(2*eps)
        np.testing.assert_allclose(ut+u*ux,.005*uxx,rtol=1.e-5,atol=2.e-7)


if __name__=='__main__':
    with threadpool_limits(limits=1):unittest.main()
