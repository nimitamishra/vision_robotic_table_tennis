"""Read-only G1 pose snapshot and preliminary right-arm IK for a table-frame target."""
import argparse
import json
import sys
import time
from pathlib import Path

import msgpack
import mujoco
import numpy as np
import zmq
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rl-interception'))
from fixed_arm_env import ARM, OFFSET, build_model


def rotate_for_waist_yaw(point_zero_waist, waist_yaw):
    """Map a zero-waist pelvis-frame point into the live yawed pelvis frame."""
    c, s = np.cos(waist_yaw), np.sin(waist_yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return rotation @ np.asarray(point_zero_waist, dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', default='tcp://127.0.0.1:5565')
    parser.add_argument('--world-target', nargs=3, type=float, default=[0, 0, .20])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    target_world = np.asarray(args.world_target, dtype=float)
    if not np.all(np.isfinite(target_world)):
        raise SystemExit('World target must be finite.')

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 6000)
    socket.connect(args.pose)
    try:
        packet = msgpack.unpackb(socket.recv(), raw=False)
    finally:
        socket.close(0)
        context.term()
    if not all(packet.get(key) for key in ('pose_fresh', 'right_arm_fresh', 'reference_seen_both_cameras')):
        raise SystemExit('Live pose, arm state, and stereo table reference must all be fresh.')
    transform = np.asarray(packet.get('T_world_pelvis'), dtype=float)
    current_q = np.asarray(packet.get('right_arm_q_rad'), dtype=float)
    measured_hand = np.asarray(packet.get('right_hand_pelvis_m'), dtype=float)
    waist_yaw = packet.get('waist_yaw_rad')
    if (transform.shape != (4, 4) or current_q.shape != (7,) or measured_hand.shape != (3,)
            or not isinstance(waist_yaw, (int, float)) or not np.isfinite(waist_yaw)
            or not all(np.all(np.isfinite(x)) for x in (transform, current_q, measured_hand))
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=.02)):
        raise SystemExit('Invalid pose packet.')

    model = build_model()
    data = mujoco.MjData(model)
    wrist_id = model.body('right_wrist_yaw_link').id
    pelvis_id = model.body('pelvis').id
    limits = np.asarray([model.jnt_range[model.joint(name + '_joint').id] for name in ARM])
    lower, upper = limits[:, 0] + .02, limits[:, 1] - .02
    def point(q):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        # The seven-joint IK model fixes waist yaw at zero; the live hand point
        # is expressed in the pelvis frame after the measured waist rotation.
        zero_waist_point = (data.xpos[wrist_id]
                            + data.xmat[wrist_id].reshape(3, 3) @ OFFSET
                            - data.xpos[pelvis_id])
        return rotate_for_waist_yaw(zero_waist_point, waist_yaw)

    model_hand = point(current_q)
    model_gap = float(np.linalg.norm(model_hand - measured_hand))
    if model_gap > .02:
        raise SystemExit(f'Observed/model hand gap is {model_gap * 100:.1f} cm; IK snapshot rejected.')
    target_local = transform[:3, :3].T @ (target_world - transform[:3, 3])
    rng = np.random.default_rng(0)
    seeds = [current_q, np.array([-.632, .009, -.012, .475, -.003, .027, -.038])]
    seeds.extend(np.clip(current_q + rng.normal(0, .5, 7), lower, upper) for _ in range(16))
    candidates = []
    for seed in seeds:
        def residual(q):
            return np.r_[point(q) - target_local, .01 * (q - current_q)]
        solution = least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper),
                                 max_nfev=250).x
        candidates.append((float(np.linalg.norm(point(solution) - target_local)),
                           float(np.linalg.norm(solution - current_q)), solution.copy()))
    feasible = [item for item in candidates if item[0] <= .005]
    error, joint_change, goal_q = (min(feasible, key=lambda item: (item[1], item[0]))
                                   if feasible else min(candidates, key=lambda item: item[0]))
    result = {
        'simulation_only': True,
        'read_only': True,
        'frame_id': packet.get('frame_id'),
        'waist_yaw_rad': waist_yaw,
        'capture_monotonic_ns': packet.get('capture_monotonic_ns'),
        'snapshot_unix_time': time.time(),
        'target_world_m': target_world.tolist(),
        'T_world_pelvis': transform.tolist(),
        'current_hand_local_m': measured_hand.tolist(),
        'model_hand_local_m': model_hand.tolist(),
        'model_vs_observation_cm': model_gap * 100,
        'observation_basis': ('Live joint angles transformed by the Viser URDF; '
                              'this comparison checks coordinate conventions, '
                              'not independent optical hand localization.'),
        'target_hand_local_m': target_local.tolist(),
        'required_hand_displacement_cm': float(np.linalg.norm(target_local - measured_hand) * 100),
        'current_right_q_rad': current_q.tolist(),
        'ik_right_q_rad': goal_q.tolist(),
        'ik_position_error_cm': error * 100,
        'joint_change_norm_rad': joint_change,
        'max_joint_change_rad': float(np.max(np.abs(goal_q - current_q))),
        'limits': 'Position-only IK; no racket orientation, table collision, or dynamic validation.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ('frame_id', 'model_vs_observation_cm',
        'required_hand_displacement_cm', 'target_hand_local_m', 'ik_right_q_rad',
        'ik_position_error_cm', 'max_joint_change_rad')}, indent=2))


if __name__ == '__main__':
    main()
