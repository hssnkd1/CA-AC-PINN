import unittest
import numpy as np
from acpinn import CircularDictionary
from problems import Poisson
from v3_select import physical_energy,SelectionSettings,guarded_selection

class TestV3(unittest.TestCase):
    def test_energy_zero(self):
        d=CircularDictionary(2,0)
        self.assertEqual(physical_energy(d,[0],np.zeros(1),Poisson(),n=4),0.)
    def test_energy_sign(self):
        d=CircularDictionary(2,0)
        j=physical_energy(d,[0],np.array([.01]),Poisson(),n=8)
        self.assertLess(j,0.)
    def test_selection_not_using_exact(self):
        p=Poisson()
        def forbidden(_):raise RuntimeError('exact solution cannot be used to select')
        p.exact=forbidden
        p.forcing=lambda x: 2*np.pi**2*np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1])
        settings=SelectionSettings(train_n=8,check_n=10,evaluation_n=8,budget=17,scales=(1.,1.3),local_strengths=())
        # Metrics require exact only AFTER selection; mock to distinguish this phase.
        from unittest.mock import patch
        with patch('v3_select.metrics',lambda *args,**kw: {'relative_l2':None}):
            x=guarded_selection(p,settings)
        self.assertIn(x['selected_parameter'],(1.,1.3))
    def test_guard_condition(self):
        from unittest.mock import patch
        p=Poisson();settings=SelectionSettings(train_n=8,check_n=10,evaluation_n=8,budget=17,
                                  scales=(1.,1.3),local_strengths=(),condition_cap=1.)
        with self.assertRaises(RuntimeError):
            with patch('v3_select.metrics',lambda *args,**kw:{}):guarded_selection(p,settings)

if __name__=='__main__':unittest.main()
