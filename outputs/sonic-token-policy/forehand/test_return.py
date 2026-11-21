"""Physical event-gate tests for the actual ball-return reward."""
import unittest
from return_env import *
class ReturnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.env=ReturnEnv()
    def setUp(self):self.e=self.env;self.e.reset(np.zeros(9),NOMINAL)
    def drop(self,x,y=0,hit=True,crossed=True):
        e=self.e;e.hit=hit;e.hit_time=0 if hit else None;e.crossed=crossed;e.bounces=1
        e.data.qpos[e.qa:e.qa+3]=[x,y,.86];e.data.qvel[e.va:e.va+6]=[0,0,-.5,0,0,0]
        mujoco.mj_forward(e.model,e.data);e.prevpos=e.bp();e.prevvel=e.bv()
        for _ in range(15):
            if e.step():break
        return e.report()
    def test_launch_is_from_opposite_side(self):
        self.assertGreater(self.e.bp()[0],NET);self.assertLess(self.e.bv()[0],0)
        self.assertEqual(self.e.observation().shape,(6,))
    def test_actual_opponent_landing_scores(self):
        r=self.drop(2.3);self.assertTrue(r['legal_return']);self.assertTrue(any(e['type']=='table' for e in self.e.events));self.assertGreater(r['reward'],190)
    def test_no_net_crossing_no_return(self):
        r=self.drop(2.3,crossed=False);self.assertFalse(r['legal_return']);self.assertEqual(r['failure'],'wrong_landing')
    def test_wrong_half_not_success(self):
        self.assertFalse(self.drop(1.)['legal_return'])
    def test_wide_landing_not_success(self):
        self.assertFalse(self.drop(2.3,y=.79)['legal_return'])
    def test_miss_cannot_get_return_reward(self):
        r=self.drop(2.3,hit=False,crossed=False);self.assertFalse(r['legal_return']);self.assertLess(r['reward'],20)
    def test_reference_parameters_are_bounded(self):
        with self.assertRaises(ValueError):self.e.reset(np.ones(9)*2,NOMINAL)
        q,dq=self.e.reference(np.arange(10)*.1+.5,np.ones(9))
        limits=self.e.model.jnt_range[self.e.model.actuator_trnid[self.e.base.body_act,0]]
        self.assertTrue(np.all(q>=limits[:,0]));self.assertTrue(np.all(q<=limits[:,1]));self.assertTrue(np.isfinite(dq).all())
if __name__=='__main__':unittest.main()
