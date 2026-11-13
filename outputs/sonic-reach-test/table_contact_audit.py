"""Replay a saved reach against the measured tabletop and net geometry.

This is an offline MuJoCo collision audit. It never publishes motor commands.
The table is registered through a saved table-to-pelvis pose, so a fresh pose
is needed before the result can describe a later physical arrangement.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


SCENE = Path('/home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml')
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION = ROOT / 'work/code-export/vision/g1/calibration.json'
DEFAULT_SNAPSHOT = Path(__file__).resolve().parent / 'runs/ready-edge-live-snapshot.json'
ARM = ('right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
       'right_elbow', 'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw')


def registered_table_model(qpos, table_to_pelvis, table_spec, racket_mount=None):
    base = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(base)
    if len(qpos) != base.nq:
        raise ValueError('Saved joint state does not match the G1 scene.')
    data.qpos[:] = qpos
    mujoco.mj_forward(base, data)
    pelvis = base.body('pelvis').id
    rotation_sim_pelvis = data.xmat[pelvis].reshape(3, 3)
    position_sim_pelvis = data.xpos[pelvis]
    rotation_table_pelvis = table_to_pelvis[:3, :3]
    position_table_pelvis = table_to_pelvis[:3, 3]
    rotation_sim_table = rotation_sim_pelvis @ rotation_table_pelvis.T
    position_sim_table = position_sim_pelvis - rotation_sim_table @ position_table_pelvis
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation_sim_table.reshape(-1))

    spec = mujoco.MjSpec.from_file(str(SCENE))
    width = float(table_spec['width_m'])
    length = float(table_spec['length_m'])
    thickness = float(table_spec['thickness_m'])
    net_height = float(table_spec['net_height_m'])
    near_edge = -float(table_spec['tag_center_inset_m'])
    center_y = near_edge + length / 2
    obstacles = {
        'table_top_audit': ([0, center_y, -thickness / 2], [width / 2, length / 2, thickness / 2]),
        'table_net_audit': ([0, center_y, net_height / 2], [(width + .2) / 2, .002, net_height / 2]),
        'table_net_post_left_audit': ([-width / 2 - .1, center_y, net_height / 2], [.011, .011, net_height / 2]),
        'table_net_post_right_audit': ([width / 2 + .1, center_y, net_height / 2], [.011, .011, net_height / 2]),
    }
    for name, (local_position, half_size) in obstacles.items():
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(position_sim_table + rotation_sim_table @ local_position).tolist(),
            quat=quat.tolist(), size=half_size, contype=1, conaffinity=1)
    if racket_mount is not None:
        from racket_clearance_audit import add_racket_geoms
        add_racket_geoms(spec.body('right_wrist_yaw_link'), racket_mount)
    model = spec.compile()
    return model, list(obstacles)


def max_arm_obstacle_penetration(model, data, obstacle_names):
    obstacle_ids = {model.geom(name).id for name in obstacle_names}
    arm_ids = {i for i in range(model.nbody) if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))
               and name.startswith(('right_shoulder', 'right_elbow', 'right_wrist', 'right_hand'))}
    worst = (0., None)
    for contact in data.contact:
        a, b = int(contact.geom1), int(contact.geom2)
        arm_geom = b if a in obstacle_ids else a if b in obstacle_ids else None
        if arm_geom is None or int(model.geom_bodyid[arm_geom]) not in arm_ids:
            continue
        depth = max(0., -float(contact.dist))
        if depth > worst[0]:
            worst = (depth, [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, a) or f'geom#{a}',
                             mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, b) or f'geom#{b}'])
    return worst


def audit(run, calibration, snapshot):
    rows = [json.loads(line) for line in (run / 'samples.jsonl').read_text().splitlines()]
    reach = [row for row in rows if row['phase'] in ('reach', 'complete')]
    if not reach or not all('full_qpos' in row for row in reach):
        raise ValueError('Run has no replayable reach joint states.')
    world = json.loads(calibration.read_text())['world']
    pose = json.loads(snapshot.read_text())
    transform = np.asarray(pose['T_world_pelvis'], dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)) or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=.02):
        raise ValueError('Invalid saved table-to-pelvis pose.')
    model, obstacles = registered_table_model(reach[0]['full_qpos'], transform, world['table'])
    data = mujoco.MjData(model)
    planned = json.loads((run / 'online_ik.json').read_text()) if (run / 'online_ik.json').exists() else None
    qadr = [model.joint(name + '_joint').qposadr[0] for name in ARM]
    worst_planned = (0., None)
    if planned is not None:
        q0 = np.asarray(planned['start_q_rad'], dtype=float)
        q1 = np.asarray(planned['goal_q_rad'], dtype=float)
        for fraction in np.linspace(0, 1, 21):
            data.qpos[:] = reach[0]['full_qpos']
            data.qpos[qadr] = q0 + fraction * (q1 - q0)
            mujoco.mj_forward(model, data)
            candidate = max_arm_obstacle_penetration(model, data, obstacles)
            if candidate[0] > worst_planned[0]:
                worst_planned = candidate
    worst_replay = (0., None)
    for row in reach:
        data.qpos[:] = row['full_qpos']
        mujoco.mj_forward(model, data)
        candidate = max_arm_obstacle_penetration(model, data, obstacles)
        if candidate[0] > worst_replay[0]:
            worst_replay = candidate
    report = {
        'simulation_only': True,
        'run': str(run.resolve()),
        'calibration': str(calibration.resolve()),
        'pose_snapshot': str(snapshot.resolve()),
        'pose_snapshot_unix_time': pose.get('snapshot_unix_time'),
        'planned_path_checked': planned is not None,
        'planned_max_table_or_net_penetration_mm': worst_planned[0] * 1000,
        'planned_contact_geoms': worst_planned[1],
        'replay_max_table_or_net_penetration_mm': worst_replay[0] * 1000,
        'replay_contact_geoms': worst_replay[1],
        'clear_at_1mm_threshold': planned is not None and worst_planned[0] <= .001 and worst_replay[0] <= .001,
        'checked_bodies': 'right shoulder, arm, wrist and hand',
        'limitations': 'Saved table pose; no racket; only right-arm/hand contacts checked; offline replay does not include table forces during simulation.',
    }
    (run / 'table_contact_audit.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument('--snapshot', type=Path, default=DEFAULT_SNAPSHOT)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.calibration, args.snapshot), indent=2))


if __name__ == '__main__':
    main()
