import unittest
from pathlib import Path

import numpy as np

from live_shadow import joined_observation
from policy_adapter import load_policy


class LiveShadowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = Path(__file__).parent / 'runs/ball-proxy-02/policy.pt'
        if not checkpoint.is_file():
            raise unittest.SkipTest('Local policy checkpoint is not included in the code export')
        cls.policy = load_policy(checkpoint)

    def packets(self):
        # Ninety-degree yaw tests table-to-pelvis conversion, not only identity.
        T = np.eye(4)
        T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        T[:3, 3] = [.1, -.4, 0]
        local_ball = np.array([.68, -.08, .99])
        local_velocity = np.array([-1., 0., 1.])
        pose = {'capture_monotonic_ns': 10_000_000_000, 'frame_id': 3,
                'pose_fresh': True, 'reference_seen_both_cameras': True,
                'right_arm_fresh': True, 'T_world_pelvis': T.tolist(),
                'right_arm_q_rad': [-.2, 0, 0, .45, 0, 0, 0],
                'right_arm_dq_rad_s': [0] * 7,
                'right_hand_pelvis_m': [.38, -.08, .83]}
        trajectory = {'timestamp_s': 10., 'frame_id': 3, 'status': 'tracking',
                      'measurement_accepted': True, 'measurement_age_s': 0.,
                      'position_m': (T[:3, :3] @ local_ball + T[:3, 3]).tolist(),
                      'velocity_m_s': (T[:3, :3] @ local_velocity).tolist()}
        return trajectory, pose

    def test_aligned_moving_ball_reaches_bounded_preview(self):
        trajectory, pose = self.packets()
        result = joined_observation(trajectory, pose, np.array([.38, -.08, .83]),
                                    self.policy)
        self.assertEqual(result['status'], 'preview')
        np.testing.assert_allclose(result['ball_pelvis_m'], [.68, -.08, .99])
        self.assertAlmostEqual(result['time_to_intercept_s'], .3)
        self.assertLessEqual(max(abs(x) for x in result['target_delta_cm']), 5.)
        self.assertFalse(result['transmitted'])

    def test_stationary_ball_does_not_invoke_policy(self):
        trajectory, pose = self.packets()
        trajectory['status'] = 'stationary'
        result = joined_observation(trajectory, pose, np.array([.38, -.08, .83]),
                                    self.policy)
        self.assertEqual(result['status'], 'stationary_ball')
        self.assertNotIn('proposed_right_pelvis_m', result)

    def test_missing_reference_and_clock_skew_are_rejected(self):
        trajectory, pose = self.packets()
        pose['reference_seen_both_cameras'] = False
        self.assertEqual(joined_observation(trajectory, pose, np.array([.38, -.08, .83]),
                                            self.policy)['status'], 'pose_or_arm_unavailable')
        pose['reference_seen_both_cameras'] = True
        trajectory['timestamp_s'] = 10.25
        self.assertEqual(joined_observation(trajectory, pose, np.array([.38, -.08, .83]),
                                            self.policy)['status'], 'sensor_clock_mismatch')

    def test_collision_before_hand_plane_blocks_preview(self):
        trajectory, pose = self.packets()
        trajectory['collision_events'] = [{'type': 'table', 'time_s': .1}]
        result = joined_observation(trajectory, pose, np.array([.38, -.08, .83]),
                                    self.policy)
        self.assertEqual(result['status'], 'collision_before_hand_plane')
        self.assertAlmostEqual(result['time_to_collision_s'], .1)
        self.assertNotIn('proposed_right_pelvis_m', result)

    def test_untrained_ready_pose_blocks_preview(self):
        trajectory, pose = self.packets()
        pose['right_arm_q_rad'][3] = 1.4
        result = joined_observation(trajectory, pose, np.array([.38, -.08, .83]),
                                    self.policy)
        self.assertEqual(result['status'], 'training_pose_mismatch')
        self.assertNotIn('proposed_right_pelvis_m', result)

    def test_ready_pose_must_be_captured_before_preview(self):
        trajectory, pose = self.packets()
        result = joined_observation(trajectory, pose, np.array([.38, -.08, .83]),
                                    self.policy, ready_valid=False)
        self.assertEqual(result['status'], 'waiting_for_ready_pose')
        self.assertNotIn('proposed_right_pelvis_m', result)


if __name__ == '__main__':
    unittest.main()
