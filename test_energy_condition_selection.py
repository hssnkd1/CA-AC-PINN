import unittest
from energy_condition_selection import select
class TestEnergyConditionSelection(unittest.TestCase):
    def test_guard_excludes_unstable_lower_energy(self):
        rows=[{'scale':1,'ritz_energy':-2,'scaled_condition':10},{'scale':2,'ritz_energy':-3,'scaled_condition':1e10}]
        self.assertEqual(select(rows,1e6)['scale'],1)
    def test_lower_ritz_energy_within_guard(self):
        rows=[{'scale':1,'ritz_energy':-2,'scaled_condition':10},{'scale':2,'ritz_energy':-3,'scaled_condition':100}]
        self.assertEqual(select(rows,1e6)['scale'],2)
    def test_no_eligible(self):
        self.assertIsNone(select([{'ritz_energy':0,'scaled_condition':float('inf')}],100))
if __name__=='__main__':unittest.main()
