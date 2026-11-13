import unittest
import numpy as np
from fixed_arm_env import FixedArmReach

class FixedArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.env=FixedArmReach(seed=1)
    def test_only_right_arm_is_dynamic_and_reset_repeats(self):
        e=self.env
        first=e.reset([.02,-.01,.01]);p0=e.data.xpos[e.pelvis_id].copy()
        for _ in range(30):e.step([.4,-.2,.2])
        self.assertEqual((e.model.nq,e.model.nv,e.model.nu),(7,7,7))
        np.testing.assert_allclose(e.data.xpos[e.pelvis_id],p0,atol=1e-10)
        np.testing.assert_allclose(e.reset([.02,-.01,.01]),first,atol=1e-8)
    def test_rule_target_reaches_held_out_goals(self):
        e=self.env
        for goal in ([.02,.01,-.015],[-.02,-.01,.02],[.01,-.02,-.02]):
            e.reset(goal)
            for _ in range(e.steps):_,_,_,info=e.step(np.array(goal)/.05)
            self.assertLess(info['error_m'],.01)
    def test_invalid_action_rejected(self):
        with self.assertRaises(ValueError):self.env.step([np.nan,0,0])
if __name__=='__main__':unittest.main()
