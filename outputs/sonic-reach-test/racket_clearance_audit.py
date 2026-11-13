"""Offline racket clearance audit; never connects to or moves the robot.

The wrist-to-racket vectors must come from measurements of the installed
hardware. A sphere overbounds an uncalibrated face; an oriented disc is used
only when its normal has been measured. The existing G1
scene still contains the old Dex3 hand; those geoms are excluded from this
attachment's robot-clearance check.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from table_contact_audit import ARM, DEFAULT_CALIBRATION, DEFAULT_SNAPSHOT, registered_table_model


ATTACHMENT_GEOMS = ('racket_mount_audit', 'racket_handle_audit', 'racket_face_audit')


def checked_mount(raw):
    """Reject missing geometry instead of silently assuming an installation angle."""
    if raw.get('measured_on_installed_racket') is not True:
        raise ValueError('Measure the installed wrist-to-grip and grip-to-face vectors first.')
    mount = {}
    for key in ('wrist_to_grip_center_m', 'grip_to_face_center_m'):
        value = np.asarray(raw.get(key), dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f'{key} must contain three measured wrist-frame metres.')
        mount[key] = value
    shape = raw.get('face_shape', 'sphere')
    if shape not in ('sphere', 'disc'):
        raise ValueError('face_shape must be sphere or disc.')
    mount['face_shape'] = shape
    if shape == 'disc':
        normal = np.asarray(raw.get('face_normal_wrist'), dtype=float)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)) or np.linalg.norm(normal) < .5:
            raise ValueError('A disc needs a measured face_normal_wrist vector.')
        mount['face_normal_wrist'] = normal / np.linalg.norm(normal)
    for key in ('face_diameter_m', 'face_thickness_m', 'handle_radius_m',
                'gripper_envelope_radius_m', 'measurement_margin_m'):
        try:
            value = float(raw[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'{key} needs a measured value in metres.') from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative.')
        mount[key] = value
    if not .10 <= mount['face_diameter_m'] <= .25:
        raise ValueError('Face diameter outside 10–25 cm; check the measurement.')
    if not .12 <= np.linalg.norm(mount['grip_to_face_center_m']) <= .30:
        raise ValueError('Grip-to-face distance outside 12–30 cm; check the wrist-frame vector.')
    if not .02 <= np.linalg.norm(mount['wrist_to_grip_center_m']) <= .35:
        raise ValueError('Wrist-to-grip distance outside 2–35 cm; check the wrist-frame vector.')
    if not 0 <= mount['measurement_margin_m'] <= .08:
        raise ValueError('Measurement margin must be no larger than 8 cm.')
    return mount


def add_racket_geoms(wrist_body, mount):
    """Attach a conservative, zero-mass collision envelope to wrist yaw."""
    grip = mount['wrist_to_grip_center_m']
    face = grip + mount['grip_to_face_center_m']
    margin = mount['measurement_margin_m']
    wrist_body.add_geom(name=ATTACHMENT_GEOMS[0], type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0., 0., 0., *grip],
        size=[mount['gripper_envelope_radius_m'] + margin, 0., 0.],
        density=0, contype=1, conaffinity=1, rgba=[.3, .5, .9, .35])
    wrist_body.add_geom(name=ATTACHMENT_GEOMS[1], type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[*grip, *face], size=[mount['handle_radius_m'] + margin, 0., 0.],
        density=0, contype=1, conaffinity=1, rgba=[.7, .4, .2, .4])
    if mount['face_shape'] == 'sphere':
        # Orientation-free, conservative bound for an uncalibrated racket.
        wrist_body.add_geom(name=ATTACHMENT_GEOMS[2], type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=face.tolist(),
            size=[np.hypot(mount['face_diameter_m'] / 2,
                           mount['face_thickness_m'] / 2) + margin, 0., 0.],
            density=0, contype=1, conaffinity=1, rgba=[.9, .2, .2, .3])
    else:
        normal = mount['face_normal_wrist']
        reference = np.array([1., 0., 0.]) if abs(normal[0]) < .9 else np.array([0., 1., 0.])
        x_axis = np.cross(reference, normal)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(normal, x_axis)
        rotation = np.column_stack((x_axis, y_axis, normal))
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, rotation.reshape(-1))
        wrist_body.add_geom(name=ATTACHMENT_GEOMS[2], type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=face.tolist(), quat=quat.tolist(),
            size=[mount['face_diameter_m'] / 2 + margin,
                  mount['face_thickness_m'] / 2 + margin, 0.],
            density=0, contype=1, conaffinity=1, rgba=[.9, .2, .2, .3])


def nearest_attachment_clearance(model, data, table_names, attachment_names=ATTACHMENT_GEOMS):
    all_attachment_ids = {model.geom(name).id for name in ATTACHMENT_GEOMS}
    attachment_ids = [model.geom(name).id for name in attachment_names]
    mount_id = model.geom(ATTACHMENT_GEOMS[0]).id
    table_ids = {model.geom(name).id for name in table_names}
    wrist_id = model.body('right_wrist_yaw_link').id
    nearest = {'table_m': float('inf'), 'robot_m': float('inf'),
               'floor_m': float('inf'), 'table_pair': None,
               'robot_pair': None, 'floor_pair': None}
    for racket_id in attachment_ids:
        for other_id in range(model.ngeom):
            if other_id in all_attachment_ids or other_id == racket_id:
                continue
            if not model.geom_contype[other_id] or not model.geom_conaffinity[other_id]:
                continue
            body_id = int(model.geom_bodyid[other_id])
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ''
            if body_id == wrist_id or body_name.startswith('right_hand_'):
                continue
            # The mount starts inside the wrist flange; contact with the two
            # immediately adjacent wrist links is intentional in this envelope.
            if (racket_id == mount_id
                    and body_name in ('right_wrist_pitch_link', 'right_wrist_roll_link')):
                continue
            category = 'table' if other_id in table_ids else 'floor' if body_id == 0 else 'robot'
            distance = float(mujoco.mj_geomDistance(model, data, racket_id, other_id, 2., None))
            if distance < nearest[category + '_m']:
                nearest[category + '_m'] = distance
                nearest[category + '_pair'] = [
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, racket_id),
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_id)
                    or f'geom#{other_id} ({body_name})']
    return nearest


def audit(run, mount_path, calibration=DEFAULT_CALIBRATION,
          snapshot=DEFAULT_SNAPSHOT, max_snapshot_age_s=30):
    mount = checked_mount(json.loads(mount_path.read_text()))
    pose = json.loads(snapshot.read_text())
    age_s = time.time() - float(pose['snapshot_unix_time'])
    transform = np.asarray(pose['T_world_pelvis'], dtype=float)
    if (transform.shape != (4, 4) or not np.all(np.isfinite(transform))
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=.02)):
        raise ValueError('Invalid table-to-pelvis pose snapshot.')
    rows = [json.loads(line) for line in (run / 'samples.jsonl').read_text().splitlines()]
    reach = [row for row in rows if row.get('phase') in ('reach', 'complete')]
    if not reach or not all('full_qpos' in row for row in reach):
        raise ValueError('Run has no replayable reach states.')
    table = json.loads(calibration.read_text())['world']['table']
    model, table_names = registered_table_model(reach[0]['full_qpos'], transform, table, mount)
    data = mujoco.MjData(model)
    planned_path = run / 'online_ik.json'
    samples = []
    if planned_path.exists():
        planned = json.loads(planned_path.read_text())
        addresses = [model.joint(name + '_joint').qposadr[0] for name in ARM]
        start = np.asarray(planned['start_q_rad'], dtype=float)
        goal = np.asarray(planned['goal_q_rad'], dtype=float)
        for fraction in np.linspace(0, 1, 41):
            qpos = np.asarray(reach[0]['full_qpos'], dtype=float).copy()
            qpos[addresses] = start + fraction * (goal - start)
            samples.append(('planned', qpos))
    samples.extend(('replay', np.asarray(row['full_qpos'], dtype=float)) for row in reach)
    worst = {'table_m': float('inf'), 'robot_m': float('inf'),
             'floor_m': float('inf'), 'table_pair': None,
             'robot_pair': None, 'floor_pair': None}
    worst_phase = {'table': None, 'robot': None, 'floor': None}
    for phase, qpos in samples:
        if qpos.shape != (model.nq,) or not np.all(np.isfinite(qpos)):
            raise ValueError('Invalid saved MuJoCo state.')
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        current = nearest_attachment_clearance(model, data, table_names)
        for category in ('table', 'robot', 'floor'):
            if current[category + '_m'] < worst[category + '_m']:
                worst[category + '_m'] = current[category + '_m']
                worst[category + '_pair'] = current[category + '_pair']
                worst_phase[category] = phase
    fresh = 0 <= age_s <= max_snapshot_age_s
    report = {
        'simulation_only': True,
        'no_robot_commands': True,
        'run': str(run.resolve()),
        'racket_mount': str(mount_path.resolve()),
        'face_shape': mount['face_shape'],
        'pose_snapshot': str(snapshot.resolve()),
        'pose_snapshot_age_s': age_s,
        'pose_fresh_for_current_setup': fresh,
        'planned_path_checked': planned_path.exists(),
        'states_checked': len(samples),
        'minimum_table_clearance_cm': worst['table_m'] * 100,
        'minimum_robot_clearance_cm': worst['robot_m'] * 100,
        'minimum_floor_clearance_cm': worst['floor_m'] * 100,
        'closest_table_pair': worst['table_pair'],
        'closest_robot_pair': worst['robot_pair'],
        'closest_floor_pair': worst['floor_pair'],
        'closest_table_phase': worst_phase['table'],
        'closest_robot_phase': worst_phase['robot'],
        'closest_floor_phase': worst_phase['floor'],
        'preliminary_clear_at_1cm': fresh and planned_path.exists()
            and all(worst[category + '_m'] >= .01 for category in ('table', 'robot', 'floor')),
        'limitations': ('Racket face is a conservative sphere or a measured-orientation disc; capsule gripper/handle; '
            'old Dex3 hand geoms excluded; saved joint replay, not a dynamics or impact test. '
            'No physical arm motion is authorized by this report.'),
    }
    (run / 'racket_clearance_audit.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--mount', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument('--snapshot', type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()
    try:
        report = audit(args.run, args.mount, args.calibration, args.snapshot)
    except (ValueError, KeyError) as exc:
        raise SystemExit(f'Racket audit unavailable: {exc}') from exc
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
