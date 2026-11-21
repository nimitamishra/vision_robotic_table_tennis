import unittest
import mujoco
import numpy as np
from front_env import FrontEnv
from front_model import TABLE

class FrontPhysicsTests(unittest.TestCase):
    def setUp(self):self.e=FrontEnv()
    def test_front_launch_and_observations(self):
        e=self.e
        self.assertGreater(e.ball_position()[0],TABLE['net_x'])
        self.assertLess(e.ball_velocity()[0],0)
        self.assertGreater(e.face()[2],TABLE['height'])
        self.assertEqual(e.observation().shape,(141,))
        self.assertTrue(np.isfinite(e.observation()).all())
        for _ in range(25):
            _,_,done,_=e.step(np.zeros(64))
            if done:break
        self.assertEqual(e.bounces,1)
        self.assertIsNone(e.failure)
        self.assertEqual(e.contact_events[0]['type'],'table')
    def drop(self,x,crossed):
        e=self.e;e.hit=True;e.crossed_net=crossed;e.bounces=1
        e.data.qpos[e.ball_qa:e.ball_qa+3]=[x,0,.84]
        e.data.qvel[e.ball_va:e.ball_va+6]=[0,0,-.5,0,0,0]
        mujoco.mj_forward(e.model,e.data)
        for _ in range(15):
            _,_,done,i=e.step(np.zeros(64))
            if done:break
        return i
    def test_far_landing_requires_net_crossing(self):
        i=self.drop(TABLE['net_x']+.4,False)
        self.assertFalse(i['legal_return']);self.assertEqual(i['failure'],'return_wrong_side')
    def test_legal_landing_has_physical_table_contact(self):
        i=self.drop(TABLE['net_x']+.4,True)
        self.assertTrue(i['legal_return']);self.assertEqual(self.e.contact_events[-1]['type'],'table')
    def test_near_landing_is_not_return(self):
        i=self.drop(TABLE['near_x']+.3,True)
        self.assertFalse(i['legal_return'])
    def test_forecast_freezes_after_hit(self):
        e=self.e;before=e.target.copy();e.hit=True;e.data.qvel[e.ball_va]=3;e._forecast()
        np.testing.assert_array_equal(before,e.target)

if __name__=='__main__':unittest.main()
