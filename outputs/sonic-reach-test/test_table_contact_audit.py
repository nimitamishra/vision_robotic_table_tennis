import unittest

import mujoco

from table_contact_audit import max_arm_obstacle_penetration


class TableContactAuditTests(unittest.TestCase):
    def test_detects_arm_table_penetration_and_clearance(self):
        model = mujoco.MjModel.from_xml_string('''
            <mujoco><worldbody>
              <geom name="table_top_audit" type="box" size="0.5 0.5 0.02" contype="1" conaffinity="1"/>
              <body name="right_wrist_yaw_link"><freejoint/>
                <geom name="arm" type="sphere" size="0.04" contype="1" conaffinity="1"/>
              </body>
            </worldbody></mujoco>
        ''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        penetration, geoms = max_arm_obstacle_penetration(model, data, ['table_top_audit'])
        self.assertGreater(penetration, .01)
        self.assertIn('table_top_audit', geoms)
        data.qpos[2] = .2
        mujoco.mj_forward(model, data)
        self.assertEqual(max_arm_obstacle_penetration(model, data, ['table_top_audit']), (0., None))


if __name__ == '__main__':
    unittest.main()
