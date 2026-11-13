"""Offline kinematic clearance sweep for tiny right-wrist motions.

Uses a saved read-only G1 joint sample. This is not SONIC dynamics, does not
model the table, and never connects to the physical robot.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from photo_static_sweep import UNITREE_JOINTS
from racket_clearance_audit import add_racket_geoms, nearest_attachment_clearance
from table_contact_audit import ARM, SCENE


def fill_body_q(model, data, snapshot):
    q = np.asarray(snapshot['body_q_rad_unitree_order'], dtype=float)
    if q.shape != (29,) or not np.all(np.isfinite(q)):
        raise ValueError('Expected 29 finite body joint angles.')
    for name, value in zip(UNITREE_JOINTS, q):
        data.qpos[model.joint(name + '_joint').qposadr[0]] = value
    mujoco.mj_forward(model, data)


def solve_short_wrist_path(model, initial_qpos, displacement):
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    mujoco.mj_forward(model, data)
    wrist = model.body('right_wrist_yaw_link').id
    addresses = [model.joint(name + '_joint').qposadr[0] for name in ARM]
    q0 = initial_qpos[addresses].copy()
    position0 = data.xpos[wrist].copy()
    rotation0 = data.xmat[wrist].reshape(3, 3).copy()
    target = position0 + np.asarray(displacement, dtype=float)
    limits = np.asarray([model.jnt_range[model.joint(name + '_joint').id] for name in ARM])
    lower, upper = limits[:, 0] + .01, limits[:, 1] - .01
    if np.any(q0 < lower) or np.any(q0 > upper):
        raise ValueError('Saved arm pose lies outside joint limits.')

    def residual(q):
        data.qpos[:] = initial_qpos
        data.qpos[addresses] = q
        mujoco.mj_forward(model, data)
        rotation = data.xmat[wrist].reshape(3, 3)
        orientation_error = Rotation.from_matrix(rotation0.T @ rotation).as_rotvec()
        return np.r_[data.xpos[wrist] - target, .10 * orientation_error, .01 * (q - q0)]

    solved = least_squares(residual, q0, bounds=(lower, upper), max_nfev=250)
    q1 = solved.x
    data.qpos[:] = initial_qpos
    data.qpos[addresses] = q1
    mujoco.mj_forward(model, data)
    angle = Rotation.from_matrix(rotation0.T @ data.xmat[wrist].reshape(3, 3)).magnitude()
    return {
        'start_q': q0,
        'goal_q': q1,
        'addresses': addresses,
        'goal_position_error_m': float(np.linalg.norm(data.xpos[wrist] - target)),
        'goal_orientation_error_deg': float(np.degrees(angle)),
        'max_joint_change_deg': float(np.degrees(np.max(np.abs(q1 - q0)))),
    }


def face_normal_for_front_edge(wrist_rotation, angle_deg):
    angle = np.deg2rad(angle_deg)
    world = np.array([np.sin(angle), np.cos(angle), 0.])
    local = wrist_rotation.T @ world
    local[0] = 0.
    return local / np.linalg.norm(local)


def arm_body_penetration(model, data):
    """Return deepest right-arm contact against non-arm robot bodies, in mm."""
    worst_mm, pair = 0., None
    arm_prefixes = ('right_shoulder', 'right_elbow', 'right_wrist')
    for contact in data.contact:
        bodies = [int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])]
        if 0 in bodies or contact.dist >= 0:
            continue
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or '' for body in bodies]
        arm = [name.startswith(arm_prefixes) for name in names]
        if sum(arm) != 1 or any(name.startswith('right_hand_') for name in names):
            continue
        depth_mm = -float(contact.dist) * 1000
        if depth_mm > worst_mm:
            worst_mm, pair = depth_mm, names
    return worst_mm, pair


def evaluate(snapshot):
    base = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(base)
    fill_body_q(base, data, snapshot)
    initial_qpos = data.qpos.copy()
    wrist = base.body('right_wrist_yaw_link').id
    wrist_rotation = data.xmat[wrist].reshape(3, 3).copy()
    paths = {
        'outward_2cm': [0., -.02, 0.],
        'up_2cm': [0., 0., .02],
        'outward_and_up_2cm_each': [0., -.02, .02],
    }
    results = []
    for path_name, displacement in paths.items():
        plan = solve_short_wrist_path(base, initial_qpos, displacement)
        arm_depths = []
        arm_pairs = []
        for alpha in np.linspace(0, 1, 31):
            data.qpos[:] = initial_qpos
            data.qpos[plan['addresses']] = (1 - alpha) * plan['start_q'] + alpha * plan['goal_q']
            mujoco.mj_forward(base, data)
            depth, pair = arm_body_penetration(base, data)
            arm_depths.append(depth)
            arm_pairs.append(pair)
        path_arm_contact = {
            'start_penetration_mm': arm_depths[0],
            'maximum_penetration_mm': max(arm_depths),
            'end_penetration_mm': arm_depths[-1],
            'worst_contact_bodies': arm_pairs[int(np.argmax(arm_depths))],
        }
        for grip_to_face in (.15, .175, .20):
            for normal_angle in (-20, 0, 20):
                spec = mujoco.MjSpec.from_file(str(SCENE))
                mount = {
                    'wrist_to_grip_center_m': np.array([.12, 0., 0.]),
                    'grip_to_face_center_m': np.array([grip_to_face, 0., 0.]),
                    'face_diameter_m': .15,
                    'face_thickness_m': .01,
                    'handle_radius_m': .012,
                    'gripper_envelope_radius_m': .035,
                    'measurement_margin_m': 0.,
                    'face_shape': 'disc',
                    'face_normal_wrist': face_normal_for_front_edge(wrist_rotation, normal_angle),
                }
                add_racket_geoms(spec.body('right_wrist_yaw_link'), mount)
                model = spec.compile()
                probe = mujoco.MjData(model)
                worst = {'face_robot_cm': float('inf'), 'handle_robot_cm': float('inf'),
                         'mount_robot_cm': float('inf'), 'face_floor_cm': float('inf')}
                for alpha in np.linspace(0, 1, 31):
                    probe.qpos[:] = initial_qpos
                    probe.qpos[plan['addresses']] = (1 - alpha) * plan['start_q'] + alpha * plan['goal_q']
                    mujoco.mj_forward(model, probe)
                    for label, geom in (('face', 'racket_face_audit'),
                                        ('handle', 'racket_handle_audit'),
                                        ('mount', 'racket_mount_audit')):
                        clearance = nearest_attachment_clearance(model, probe, [], (geom,))
                        worst[label + '_robot_cm'] = min(worst[label + '_robot_cm'],
                                                         clearance['robot_m'] * 100)
                        if label == 'face':
                            worst['face_floor_cm'] = min(worst['face_floor_cm'],
                                                         clearance['floor_m'] * 100)
                results.append({
                    'path': path_name,
                    'requested_wrist_displacement_m': displacement,
                    'grip_to_face_center_m_assumed': grip_to_face,
                    'face_normal_robot_lateral_offset_deg_assumed': normal_angle,
                    'goal_position_error_cm': plan['goal_position_error_m'] * 100,
                    'goal_orientation_error_deg': plan['goal_orientation_error_deg'],
                    'max_joint_change_deg': plan['max_joint_change_deg'],
                    'modeled_arm_body_contact': path_arm_contact,
                    'minimum_clearance': worst,
                })
    return {
        'simulation_only': True,
        'read_only': True,
        'robot_joint_sample_unix_time': snapshot['sample_unix_time'],
        'evaluated_unix_time': time.time(),
        'path_samples_per_scenario': 31,
        'scenarios': results,
        'safe_for_physical_motion': False,
        'limitations': ('Kinematic joint interpolation only. Wrist+X mount directions and face normals '
                        'are estimates from photographs, not calibrated. The gripper body and silver '
                        'bracket lack CAD. No table pose or SONIC dynamics. No physical motion is '
                        'authorized by this sweep.'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(json.loads(args.snapshot.read_text()))
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        'scenarios': len(result['scenarios']),
        'safe_for_physical_motion': result['safe_for_physical_motion'],
        'output': str(args.output.resolve()),
    }, indent=2))


if __name__ == '__main__':
    main()
