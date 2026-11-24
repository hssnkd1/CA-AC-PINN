import unittest
import numpy as np
from scipy import sparse
from acpinn import CircularDictionary,assemble,HierarchicalRitz
from ca_acpinn import pde_local_radii,ConditionAwareRitz,assemble_sparse,choose_by_energy
from problems import Poisson,Interface

class CAACPINNTests(unittest.TestCase):
    def test_local_radii_are_nonuniform_positive(self):
        d=CircularDictionary(2,1);p=Interface()
        info=pde_local_radii(d,p,1.5,.3)
        self.assertTrue(np.all(d.radii>0));self.assertGreater(info['scale_max']-info['scale_min'],.05)
    def test_boundary_still_exact(self):
        d=CircularDictionary(2,1);pde_local_radii(d,Poisson(),1.5,.2)
        x=np.array([[0,.3],[1,.7],[.5,0],[.4,1]])
        v,_,_=d.evaluate(x)
        self.assertLess(np.max(np.abs(v)),1e-12)
    def test_condition_score_guard(self):
        d=CircularDictionary(1,1)
        rng=np.random.default_rng(2);A=rng.normal(size=(d.size,d.size));K=A.T@A+np.eye(d.size)
        f=rng.normal(size=d.size)
        model=ConditionAwareRitz(d,K,f,min_schur_eig=1e-7)
        model.grow(len(d.initial)+2)
        self.assertEqual(len(model.history),1)
        self.assertLess(model.history[0]['identity_defect'],1e-8)
    def test_sparse_dense_assembly_agree(self):
        d=CircularDictionary(2,1);p=Poisson()
        a=assemble(d,p,n=5,order=2,keep_samples=False)
        b=assemble_sparse(d,p,n=5,order=2)
        self.assertTrue(sparse.issparse(b['stiffness']))
        self.assertTrue(np.allclose(a.stiffness,b['stiffness'].toarray(),atol=1e-10))
        self.assertTrue(np.allclose(a.mass,b['mass'].toarray(),atol=1e-10))
        self.assertTrue(np.allclose(a.rhs,b['rhs'],atol=1e-10))
    def test_selector_does_not_use_true_error(self):
        rows=[{'method':'a','energy':-10.,'scaled_condition':2e6,'relative_l2':1e-6},
              {'method':'b','energy':-9.,'scaled_condition':1e4,'relative_l2':1.}]
        self.assertEqual(choose_by_energy(rows,1e6)['method'],'b')

if __name__=='__main__':unittest.main()
