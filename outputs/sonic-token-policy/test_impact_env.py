import unittest
import numpy as np
from impact_env import ImpactEnv

class ImpactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.env = ImpactEnv()

    def reset(self, uv=(0, 0)):
        return self.env.reset({'offset_uv_m': list(uv), 'flight_time_s': .8, 'speed_m_s': 1.1})

    def test_free_flight_and_forecast(self):
        self.reset(); e = self.env
        p, v = e.ball_position(), e.ball_velocity()
        self.assertAlmostEqual(e.time_to_plane, .8, places=6)
        for _ in range(5):e.step(np.zeros(64))
        exact = p+v*.1+.5*e.model.opt.gravity*.1**2
        self.assertLess(np.linalg.norm(e.ball_position()-exact), .0011)
        self.assertFalse(e.hit)
        self.assertEqual(e.observation().shape, (141,))

    def test_contact_requires_force_and_velocity_reversal(self):
        self.reset(); e = self.env
        for _ in range(e.steps):
            _, _, done, info = e.step(np.zeros(64))
            if done:break
        self.assertTrue(info['hit'])
        self.assertFalse(info['unsafe'])
        self.assertGreater(e.contact_events[0]['normal_force_n'], 0)
        self.assertLess(np.array(e.contact_events[0]['velocity_before_m_s']) @ e.normal, -.1)
        self.assertGreater(info['outgoing_normal_speed_m_s'], .2)

    def test_nonface_contact_cannot_change_target_or_score_hit(self):
        self.reset(); e = self.env
        target = e.target.copy()
        e.other_contact = True
        e.data.qvel[e.ball_va:e.ball_va+3] = [4, -3, 6]
        e._update_goal()
        np.testing.assert_array_equal(e.target, target)
        _, _, done, info = e.step(np.zeros(64))
        self.assertTrue(done); self.assertFalse(info['hit'])

    def test_invalid_launch(self):
        for uv in ([float('nan'), 0], [.5, 0]):
            with self.assertRaises(ValueError):self.reset(uv)

if __name__ == '__main__':unittest.main()
