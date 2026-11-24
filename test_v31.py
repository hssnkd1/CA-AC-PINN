import unittest
from guarded_evolution_v31 import select

class GuardedEvolutionTests(unittest.TestCase):
    def test_uses_residual_not_exact_error(self):
        a={'variant':'a','newton_failures':0,'heldout_weak_residual':.1,'relative_l2':.9}
        b={'variant':'b','newton_failures':0,'heldout_weak_residual':.2,'relative_l2':.01}
        self.assertEqual(select([a,b])['variant'],'a')
    def test_rejects_newton_failures(self):
        a={'variant':'a','newton_failures':1,'heldout_weak_residual':.01}
        b={'variant':'b','newton_failures':0,'heldout_weak_residual':.2}
        self.assertEqual(select([a,b])['variant'],'b')
    def test_all_failed(self):
        self.assertIsNone(select([{'variant':'a','failure':'error'}]))

if __name__=='__main__':unittest.main()
