import unittest

import mujoco

from racket_clearance_audit import (
    add_racket_geoms, checked_mount, nearest_attachment_clearance,
)


def sample_mount():
    return {
        'measured_on_installed_racket': True,
        'wrist_to_grip_center_m': [.08, 0, 0],
        'grip_to_face_center_m': [.18, 0, 0],
        'face_diameter_m': .15,
        'face_thickness_m': .01,
        'handle_radius_m': .012,
        'gripper_envelope_radius_m': .035,
        'measurement_margin_m': .01,
    }


class RacketClearanceAuditTests(unittest.TestCase):
    def test_missing_mount_direction_is_rejected(self):
        mount = sample_mount()
        mount['wrist_to_grip_center_m'] = None
        with self.assertRaises(ValueError):
            checked_mount(mount)

    def test_racket_face_clearance_reacts_to_table_movement(self):
        spec = mujoco.MjSpec.from_string('''
          <mujoco><worldbody>
            <geom name="table_top_audit" type="box" pos="0.26 0 -0.12"
                  size="0.2 0.2 0.01"/>
            <body name="right_wrist_yaw_link"><freejoint/>
              <geom name="wrist" type="sphere" size="0.01"/>
            </body>
          </worldbody></mujoco>
        ''')
        add_racket_geoms(spec.body('right_wrist_yaw_link'), checked_mount(sample_mount()))
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        clear = nearest_attachment_clearance(model, data, ['table_top_audit'])
        self.assertGreater(clear['table_m'], 0)
        data.qpos[2] = -.08
        mujoco.mj_forward(model, data)
        touching = nearest_attachment_clearance(model, data, ['table_top_audit'])
        self.assertLess(touching['table_m'], 0)
        self.assertEqual(touching['table_pair'][0], 'racket_face_audit')

    def test_oriented_disc_avoids_sphere_only_false_collision(self):
        xml = '''<mujoco><worldbody>
          <geom name="table_top_audit" type="box" pos="0.26 0.065 0"
                size="0.01 0.01 0.01"/>
          <body name="right_wrist_yaw_link"><freejoint/>
            <geom name="wrist" type="sphere" size="0.01"/>
          </body>
        </worldbody></mujoco>'''
        results = {}
        for shape in ('sphere', 'disc'):
            spec = mujoco.MjSpec.from_string(xml)
            mount = sample_mount()
            mount['face_shape'] = shape
            if shape == 'disc':
                mount['face_normal_wrist'] = [0, 1, 0]
            add_racket_geoms(spec.body('right_wrist_yaw_link'), checked_mount(mount))
            model = spec.compile()
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            results[shape] = nearest_attachment_clearance(model, data, ['table_top_audit'])['table_m']
        self.assertLess(results['sphere'], 0)
        self.assertGreater(results['disc'], 0)


if __name__ == '__main__':
    unittest.main()
