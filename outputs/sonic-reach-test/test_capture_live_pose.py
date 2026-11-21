import unittest

import numpy as np

from capture_live_pose import rotate_for_waist_yaw


class LivePoseFrameTests(unittest.TestCase):
    def test_positive_waist_yaw_rotates_forward_into_left(self):
        point = rotate_for_waist_yaw([1., 0., .3], np.pi / 2)
        np.testing.assert_allclose(point, [0., 1., .3], atol=1e-12)

    def test_negative_waist_yaw_matches_measured_g1_convention(self):
        point = rotate_for_waist_yaw([.2, 0., 0.], -.568)
        self.assertLess(point[1], 0)
        self.assertAlmostEqual(np.linalg.norm(point), .2)


if __name__ == '__main__':
    unittest.main()
