"""Regression checks for initial-condition driven adaptive support."""
import unittest
import numpy as np
from acpinn import CircularDictionary
from local_radius import initial_gradient_radii

class TestInitialRadii(unittest.TestCase):
    def test_constant_initial_condition_matches_base_scale(self):
        d=CircularDictionary(1,2);before=d.radii.copy()
        initial_gradient_radii(d,lambda z: np.ones(len(z)),strength=.3,base_scale=1.5)
        np.testing.assert_allclose(d.radii/before,1.5*(1+.3*.5))
    def test_nonconstant_localization_and_positive_radii(self):
        d=CircularDictionary(1,2);before=d.radii.copy()
        diagnostic=initial_gradient_radii(d,lambda z: np.tanh((z[:,0]-.5)/.025),strength=.2)
        self.assertGreater(np.ptp(d.radii/before),.05)
        self.assertTrue(np.all(d.radii>0))
        self.assertGreater(diagnostic['max_scale'],diagnostic['min_scale'])
    def test_invalid_strength_rejected(self):
        d=CircularDictionary(2,1)
        with self.assertRaises(ValueError):
            initial_gradient_radii(d,lambda z:np.zeros(len(z)),strength=-.5)

if __name__=='__main__': unittest.main()
