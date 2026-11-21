import unittest
import numpy as np
import torch
from decoder_env import *
from runtime import predict
from train import TokenPolicy

class DecoderTests(unittest.TestCase):
    def test_joint_order_round_trip(self):
        x=np.arange(29)
        np.testing.assert_array_equal(x[TO_MJ][TO_ISAAC],x)
        self.assertEqual(len(DEFAULT),29)
    def test_history_round_trip(self):
        h=np.arange(930,dtype=np.float32).reshape(10,93)
        obs=pack_observation(np.zeros(64),h)
        self.assertEqual(obs.shape,(994,))
        np.testing.assert_array_equal(split_history(obs),h)
    def test_reject_wrong_old_policy_input(self):
        p=TokenPolicy()
        for obs in (np.zeros(27),np.full(131,np.nan)):
            with self.assertRaises(ValueError):predict(p,obs)
    def test_nominal_stability_and_slew_limit(self):
        e=TokenReachEnv(steps=75)
        for invalid in (np.zeros(3),np.full(64,np.nan)):
            with self.assertRaises(ValueError):e.step(invalid)
        for _ in range(75):_,_,_,info=e.step(np.zeros(64))
        self.assertFalse(info['unsafe']);self.assertLess(info['error_cm'],.5)
        e.reset([0,-.02,0]);_,_,_,info=e.step(np.ones(64))
        self.assertLessEqual(np.max(abs(np.array(info['token'])-e.base_token)),.015001)
        self.assertEqual(e.observation().shape,(131,))

if __name__=='__main__':unittest.main()
