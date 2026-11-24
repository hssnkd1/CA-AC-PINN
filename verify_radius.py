"""Tests for the radius-sensitivity extension; does not modify original results."""
import unittest
import numpy as np
from acpinn import CircularDictionary, gauss_grid, HierarchicalRitz

class RadiusExtensionTests(unittest.TestCase):
    def test_radius_scaling_keeps_centers_and_topology(self):
        base=CircularDictionary(2,3); enlarged=CircularDictionary(2,3)
        enlarged.radii*=1.2
        self.assertTrue(np.array_equal(base.centers,enlarged.centers))
        self.assertEqual(base.children,enlarged.children)
        self.assertTrue(np.allclose(enlarged.radii,1.2*base.radii))
    def test_boundary_trace_with_wide_support(self):
        d=CircularDictionary(2,1);d.radii*=1.5
        t=np.linspace(0,1,23)
        x=np.vstack([np.column_stack([t,t*0]),np.column_stack([t,1+0*t]),
                     np.column_stack([0*t,t]),np.column_stack([1+0*t,t])])
        v,_,_=d.evaluate(x,list(range(d.size)))
        self.assertLess(np.max(np.abs(v)),1e-13)
    def test_quadratic_block_gain_identity(self):
        d=CircularDictionary(1,1)
        # Independent synthetic SPD coefficient system. Verify the algebraic invariant.
        rng=np.random.default_rng(12);Q=rng.normal(size=(d.size,d.size));K=Q.T@Q+np.eye(d.size)
        f=rng.normal(size=d.size);model=HierarchicalRitz(d,K,f)
        model.grow(len(model.ids)+2)
        self.assertEqual(len(model.history),1)
        self.assertLess(model.history[0]['identity_defect'],1e-9)

if __name__=='__main__':unittest.main()
