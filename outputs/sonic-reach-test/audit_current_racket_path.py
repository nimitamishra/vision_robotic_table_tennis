"""Read-only, assumption-labelled collision check for a short installed-racket path."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from racket_clearance_audit import nearest_attachment_clearance
from racket_path_sweep import arm_body_penetration, fill_body_q, solve_short_wrist_path
from table_contact_audit import DEFAULT_CALIBRATION, SCENE, registered_table_model


def audit(snapshot, pose, calibration, grip_to_face_m, replay_run=None):
    if not .12 <= grip_to_face_m <= .30:
        raise ValueError('Grip-to-face distance must be 12–30 cm.')
    gap_s = abs(float(snapshot['sample_unix_time']) - float(pose['snapshot_unix_time']))
    if gap_s > 30:
        raise ValueError(f'Pose and joint state are {gap_s:.1f} s apart; capture them again.')
    transform = np.asarray(pose['T_world_pelvis'], dtype=float)
    if (transform.shape != (4, 4) or not np.all(np.isfinite(transform))
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=.02)):
        raise ValueError('Invalid table-to-pelvis pose.')
    table = calibration['world']['table']
    base = mujoco.MjModel.from_xml_path(str(SCENE))
    initial = mujoco.MjData(base)
    fill_body_q(base, initial, snapshot)
    qpos = initial.qpos.copy()
    # All mount directions remain photographic estimates. A sphere bounds the
    # entire face irrespective of its unmeasured normal.
    mount = dict(wrist_to_grip_center_m=np.array([.12, 0., 0.]),
                 grip_to_face_center_m=np.array([grip_to_face_m, 0., 0.]),
                 face_diameter_m=.15, face_thickness_m=.01, handle_radius_m=.012,
                 gripper_envelope_radius_m=.035, measurement_margin_m=0.,
                 face_shape='sphere')
    model, table_geoms = registered_table_model(qpos, transform, table, mount)
    probe = mujoco.MjData(model)
    paths = {'start': [0., 0., 0.], 'outward_2cm': [0., -.02, 0.],
             'up_2cm': [0., 0., .02]}
    results = {}
    for name, displacement in paths.items():
        plan = None if name == 'start' else solve_short_wrist_path(base, qpos, displacement)
        smallest = {category: (float('inf'), None) for category in ('table', 'robot', 'floor')}
        arm_depth_mm = 0.
        arm_pair = None
        for fraction in np.linspace(0, 1, 31):
            probe.qpos[:] = qpos
            if plan is not None:
                probe.qpos[plan['addresses']] = ((1 - fraction) * plan['start_q']
                                                   + fraction * plan['goal_q'])
            mujoco.mj_forward(model, probe)
            clearance = nearest_attachment_clearance(model, probe, table_geoms)
            for category in smallest:
                distance = clearance[category + '_m']
                if distance < smallest[category][0]:
                    smallest[category] = (distance, clearance[category + '_pair'])
            depth, pair = arm_body_penetration(model, probe)
            if depth > arm_depth_mm:
                arm_depth_mm, arm_pair = depth, pair
        results[name] = {
            'requested_wrist_displacement_m': displacement,
            'goal_position_error_cm': None if plan is None else plan['goal_position_error_m'] * 100,
            'max_joint_change_deg': None if plan is None else plan['max_joint_change_deg'],
            'minimum_clearance_cm': {key: value[0] * 100 for key, value in smallest.items()},
            'closest_pairs': {key: value[1] for key, value in smallest.items()},
            'max_modeled_arm_body_penetration_mm': arm_depth_mm,
            'worst_arm_body_pair': arm_pair,
        }
    report = {
        'read_only': True, 'simulation_only': True,
        'joint_sample_unix_time': snapshot['sample_unix_time'],
        'pose_snapshot_unix_time': pose['snapshot_unix_time'],
        'pose_joint_time_gap_s': gap_s, 'evaluated_unix_time': time.time(),
        'wrist_to_grip_center_m_measured': .12,
        'grip_to_face_center_m_user_estimate': grip_to_face_m,
        'face_diameter_m_user_estimate': .15,
        'racket_mount_direction_assumed': 'wrist local +X',
        'racket_face_shape_assumed': 'orientation-free bounding sphere',
        'paths': results,
        'safe_for_physical_motion': False,
        'limitations': ('Mount direction, bracket, gripper shape and face normal are not calibrated. '
                        'The old Dex3 hand geom is excluded for racket clearance. '
                        'Joint interpolation is not SONIC tracking or robot balance. '
                        'No real motor command was sent.'),
    }
    if replay_run is not None:
        states = [json.loads(line) for line in (replay_run / 'samples.jsonl').read_text().splitlines()]
        states = [row for row in states if row['phase'] in ('reach', 'complete')]
        if not states:
            raise ValueError('Replay has no reach states.')
        replay_model, replay_table = registered_table_model(
            np.asarray(states[0]['full_qpos']), transform, table, mount)
        replay_probe = mujoco.MjData(replay_model)
        face_id = replay_model.geom('racket_face_audit').id
        pelvis_id = replay_model.body('pelvis').id
        wrist_id = replay_model.body('right_wrist_yaw_link').id
        face_points = []
        wrist_rotations = []
        replay_min = {category: (float('inf'), None, None) for category in ('table', 'robot', 'floor')}
        replay_arm_depth = 0.
        replay_arm_pair = None
        for row in states:
            replay_probe.qpos[:] = row['full_qpos']
            mujoco.mj_forward(replay_model, replay_probe)
            pelvis_rotation = replay_probe.xmat[pelvis_id].reshape(3, 3)
            face_points.append(pelvis_rotation.T @
                               (replay_probe.geom_xpos[face_id] - replay_probe.xpos[pelvis_id]))
            wrist_rotations.append(pelvis_rotation.T @
                                   replay_probe.xmat[wrist_id].reshape(3, 3))
            clearance = nearest_attachment_clearance(replay_model, replay_probe, replay_table)
            for category in replay_min:
                distance = clearance[category + '_m']
                if distance < replay_min[category][0]:
                    replay_min[category] = (distance, clearance[category + '_pair'], row['phase'])
            depth, pair = arm_body_penetration(replay_model, replay_probe)
            if depth > replay_arm_depth:
                replay_arm_depth, replay_arm_pair = depth, pair
        face_points = np.asarray(face_points)
        face_start = face_points[0]
        face_end = face_points[-min(20, len(face_points)):].mean(axis=0)
        wrist_turn = Rotation.from_matrix(wrist_rotations[0].T @ wrist_rotations[-1]).magnitude()
        report['replay'] = {
            'run': str(replay_run.resolve()),
            'states_checked': len(states),
            'minimum_clearance_cm': {key: value[0] * 100 for key, value in replay_min.items()},
            'closest_pairs': {key: value[1] for key, value in replay_min.items()},
            'closest_phases': {key: value[2] for key, value in replay_min.items()},
            'max_modeled_arm_body_penetration_mm': replay_arm_depth,
            'worst_arm_body_pair': replay_arm_pair,
            'modeled_face_center_displacement_cm': ((face_end - face_start) * 100).tolist(),
            'modeled_face_center_displacement_norm_cm': float(np.linalg.norm(face_end - face_start) * 100),
            'modeled_face_max_excursion_cm': float(np.max(np.linalg.norm(face_points - face_start, axis=1)) * 100),
            'modeled_wrist_rotation_deg': float(np.degrees(wrist_turn)),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--joints', type=Path, required=True)
    parser.add_argument('--pose', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument('--grip-to-face-cm', type=float, default=15)
    parser.add_argument('--replay-run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(json.loads(args.joints.read_text()), json.loads(args.pose.read_text()),
                   json.loads(args.calibration.read_text()), args.grip_to_face_cm / 100,
                   args.replay_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({'output': str(args.output.resolve()),
                      'safe_for_physical_motion': False,
                      'paths': report['paths'],
                      'replay': report.get('replay')}, indent=2))


if __name__ == '__main__':
    main()
