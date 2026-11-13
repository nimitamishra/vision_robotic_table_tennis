"""Read-only static geometry sensitivity check for the installed racket photos.

This deliberately uses uncalibrated mount directions and cannot certify a
physical arm path. It reads a saved robot joint sample; it never connects to
or commands the robot.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from racket_clearance_audit import ATTACHMENT_GEOMS, add_racket_geoms, nearest_attachment_clearance
from table_contact_audit import SCENE


UNITREE_JOINTS = (
    'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee',
    'left_ankle_pitch', 'left_ankle_roll', 'right_hip_pitch',
    'right_hip_roll', 'right_hip_yaw', 'right_knee',
    'right_ankle_pitch', 'right_ankle_roll', 'waist_yaw', 'waist_roll',
    'waist_pitch', 'left_shoulder_pitch', 'left_shoulder_roll',
    'left_shoulder_yaw', 'left_elbow', 'left_wrist_roll',
    'left_wrist_pitch', 'left_wrist_yaw', 'right_shoulder_pitch',
    'right_shoulder_roll', 'right_shoulder_yaw', 'right_elbow',
    'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw',
)


def evaluate(snapshot, wrist_to_grip_m=.12, grip_to_face_values=(.15, .175, .20)):
    q = np.asarray(snapshot['body_q_rad_unitree_order'], dtype=float)
    if q.shape != (29,) or not np.all(np.isfinite(q)):
        raise ValueError('Expected 29 finite Unitree body joint angles.')
    if not .05 <= wrist_to_grip_m <= .25:
        raise ValueError('Wrist-to-grip distance must be 5–25 cm.')
    base = mujoco.MjModel.from_xml_path(str(SCENE))
    base_data = mujoco.MjData(base)
    for name, value in zip(UNITREE_JOINTS, q):
        base_data.qpos[base.joint(name + '_joint').qposadr[0]] = value
    mujoco.mj_forward(base, base_data)
    wrist_id = base.body('right_wrist_yaw_link').id
    wrist_rotation = base_data.xmat[wrist_id].reshape(3, 3)
    wrist_down_vector = wrist_rotation[:, 0].tolist()

    def normal_from_robot_lateral(angle_deg):
        angle = np.deg2rad(angle_deg)
        world = np.array([np.sin(angle), np.cos(angle), 0.])
        local = wrist_rotation.T @ world
        # The racket face contains the handle, so its normal is perpendicular
        # to the approximately wrist-+X handle direction.
        local[0] = 0.
        return local / np.linalg.norm(local)

    rows = []
    scenarios = (
        ('orientation_free_sphere', 'sphere', None),
        ('disc_normal_wrist_y', 'disc', np.array([0., 1., 0.])),
        ('disc_normal_wrist_z', 'disc', np.array([0., 0., 1.])),
        ('disc_normal_robot_lateral', 'disc', normal_from_robot_lateral(0)),
        ('disc_normal_lateral_plus_20deg', 'disc', normal_from_robot_lateral(20)),
        ('disc_normal_lateral_minus_20deg', 'disc', normal_from_robot_lateral(-20)),
    )
    for grip_to_face_m in grip_to_face_values:
        if not .12 <= grip_to_face_m <= .30:
            raise ValueError('Grip-to-face distance outside 12–30 cm.')
        for scenario, shape, normal in scenarios:
            spec = mujoco.MjSpec.from_file(str(SCENE))
            mount = {
                'wrist_to_grip_center_m': np.array([wrist_to_grip_m, 0., 0.]),
                'grip_to_face_center_m': np.array([grip_to_face_m, 0., 0.]),
                'face_diameter_m': .15,
                'face_thickness_m': .01,
                'handle_radius_m': .012,
                'gripper_envelope_radius_m': .035,
                'measurement_margin_m': 0.,
                'face_shape': shape,
            }
            if normal is not None:
                mount['face_normal_wrist'] = normal
            add_racket_geoms(spec.body('right_wrist_yaw_link'), mount)
            model = spec.compile()
            data = mujoco.MjData(model)
            for name, value in zip(UNITREE_JOINTS, q):
                data.qpos[model.joint(name + '_joint').qposadr[0]] = value
            mujoco.mj_forward(model, data)
            parts = {}
            for geom in ATTACHMENT_GEOMS:
                distances = nearest_attachment_clearance(model, data, [], (geom,))
                parts[geom] = {
                    'robot_clearance_cm': distances['robot_m'] * 100,
                    'closest_robot_pair': distances['robot_pair'],
                    'floor_clearance_cm': distances['floor_m'] * 100,
                }
            rows.append({
                'grip_to_face_center_m': grip_to_face_m,
                'scenario': scenario,
                'face_normal_wrist_assumed': normal.tolist() if normal is not None else None,
                'components': parts,
            })
    return {
        'simulation_only': True,
        'read_only': True,
        'robot_joint_sample_unix_time': snapshot['sample_unix_time'],
        'evaluated_unix_time': time.time(),
        'wrist_local_x_in_model_world': wrist_down_vector,
        'measured_wrist_to_grip_distance_m': wrist_to_grip_m,
        'assumed_wrist_to_grip_direction': 'wrist local +X from photo and current FK',
        'assumed_grip_to_face_direction': 'wrist local +X from photo',
        'front_view_observation': 'The user mainly sees the racket edge from the robot front.',
        'face_diameter_m': .15,
        'face_thickness_m_assumed': .01,
        'gripper_envelope_radius_m_assumed': .035,
        'rows': rows,
        'safe_for_physical_motion': False,
        'limitations': ('No measured wrist-frame mount vectors or face normal; no gripper/bracket CAD; '
                        'no current table pose or path dynamics. Negative clearances can be artifacts '
                        'of the deliberately broad attachment envelope.'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(json.loads(args.snapshot.read_text()))
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        'scenarios': len(report['rows']),
        'wrist_local_x_in_model_world': report['wrist_local_x_in_model_world'],
        'safe_for_physical_motion': report['safe_for_physical_motion'],
    }, indent=2))


if __name__ == '__main__':
    main()
