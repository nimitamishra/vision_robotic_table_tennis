import unittest
import numpy as np
import torch
from ball_proxy_env import BallProxyReach
from train_ball_proxy import features,expert_action
class BallProxyTests(unittest.TestCase):
    def test_ball_crossing_and_model_input(self):
        env=BallProxyReach(seed=7)
        obs=env.reset([.02,-.01,.015])
        self.assertEqual(obs.shape,(27,))
        np.testing.assert_allclose(expert_action(features(obs)[None,:]).detach().numpy()[0]*.05,
                                   env.goal-env.ready,atol=2e-7)
        for _ in range(env.steps):obs,_,done,info=env.step((env.goal-env.ready)/.05)
        self.assertTrue(done)
        np.testing.assert_allclose(env.ball()[0],env.goal,atol=1e-8)
        self.assertAlmostEqual(info['proxy_contact_error_m'],np.linalg.norm(env.point()-env.goal))
if __name__=='__main__':unittest.main()
