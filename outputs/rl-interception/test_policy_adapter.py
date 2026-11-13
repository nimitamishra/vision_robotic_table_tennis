import unittest
import numpy as np
from policy_adapter import right_hand_target,vr_target_vectors,encode_sonic_planner,load_policy
class FakePolicy:
    def __call__(self,x):
        import torch
        return torch.tensor([.8,.8,.8]),torch.tensor(0.)
class AdapterTests(unittest.TestCase):
    def test_bounded_action_and_rate(self):
        args=(FakePolicy(),np.zeros(7),np.zeros(7),np.zeros(3),np.zeros(3),
              np.array([.5,0,.5]),np.array([-1.,0,0]),.5)
        t=right_hand_target(*args)
        np.testing.assert_allclose(t,[.04,.04,.04])
        t2=right_hand_target(*args,previous_target=np.zeros(3),dt=.05)
        np.testing.assert_allclose(t2,[.0015,.0015,.0015])
    def test_reject_stale_or_invalid(self):
        args=(FakePolicy(),np.zeros(7),np.zeros(7),np.zeros(3),np.zeros(3),
              np.zeros(3),np.zeros(3))
        with self.assertRaises(ValueError):right_hand_target(*args,.7)
        with self.assertRaises(ValueError):right_hand_target(*args,0.)
    def test_sonic_vector_order(self):
        p,q=vr_target_vectors([1,2,3],[4,5,6],[7,8,9],[1,0,0,0],[1,0,0,0],[1,0,0,0])
        self.assertEqual(p,[1.,2.,3.,4.,5.,6.,7.,8.,9.]);self.assertEqual(len(q),12)
        self.assertTrue(encode_sonic_planner([1,2,3],[4,5,6],[7,8,9],[1,0,0,0],[1,0,0,0],[1,0,0,0]).startswith(b'planner'))
if __name__=='__main__':unittest.main()
