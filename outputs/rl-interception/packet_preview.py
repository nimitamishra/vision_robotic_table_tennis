"""Preview a ball-policy SONIC planner packet without opening a socket.

The sample comes from the fixed-pelvis MuJoCo training environment. This is
only a format and coordinate-conversion check, not a live robot integration.
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from ball_proxy_env import BallProxyReach
from policy_adapter import encode_sonic_planner, load_policy, right_hand_target


UPSTREAM = Path('/home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl')
sys.path.insert(0, str(UPSTREAM))
OFFSETS = np.array([[.18, -.025, 0.], [.18, .025, 0.], [0., 0., .35]])
LINKS = ('left_wrist_yaw_link', 'right_wrist_yaw_link', 'torso_link')


def preview(checkpoint, seed):
    env = BallProxyReach(seed=seed)
    policy = load_policy(checkpoint)
    pelvis = env.pelvis_id
    root_pos = env.data.xpos[pelvis].copy()
    root_rot = env.data.xmat[pelvis].reshape(3, 3).copy()
    root_inv = env.data.xquat[pelvis].copy()
    root_inv[1:] *= -1
    points = []
    quaternions = []
    for link, offset in zip(LINKS, OFFSETS):
        i = env.model.body(link).id
        world = env.data.xpos[i] + env.data.xmat[i].reshape(3, 3) @ offset
        points.append((world - root_pos) @ root_rot)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, root_inv, env.data.xquat[i])
        quaternions.append(q)

    ball_world, velocity_world = env.ball()
    ball_local = (ball_world - root_pos) @ root_rot
    velocity_local = velocity_world @ root_rot
    ready_right = points[1].copy()
    hand_target = right_hand_target(policy, env.data.qpos.copy(),
                                   env.data.qvel.copy(), points[1], ready_right,
                                   ball_local, velocity_local, .5)
    packet = encode_sonic_planner(points[0], hand_target, points[2],
                                  *quaternions)
    if not packet.startswith(b'planner'):
        raise RuntimeError('Unexpected SONIC packet topic')
    header = json.loads(packet[7:1287].rstrip(b'\0'))
    shapes = {item['name']: item['shape'] for item in header['fields']}
    if shapes.get('vr_position') != [9] or shapes.get('vr_orientation') != [12]:
        raise RuntimeError('Incorrect SONIC VR vector layout')
    return {
        'source': 'MuJoCo fixed-pelvis arm and analytic ball; no robot data',
        'checkpoint': str(Path(checkpoint).resolve()),
        'seed': seed,
        'ball_pelvis_m': ball_local.tolist(),
        'ball_velocity_pelvis_m_s': velocity_local.tolist(),
        'ready_right_pelvis_m': ready_right.tolist(),
        'proposed_right_pelvis_m': hand_target.tolist(),
        'right_target_delta_cm': ((hand_target - ready_right) * 100).tolist(),
        'packet_topic': 'planner',
        'packet_bytes': len(packet),
        'vr_position_count': shapes['vr_position'][0],
        'vr_orientation_count': shapes['vr_orientation'][0],
        'transmitted': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path,
                        default=Path(__file__).parent / 'runs/ball-proxy-02/policy.pt')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parent / 'runs/packet-preview.json')
    parser.add_argument('--seed', type=int, default=20261107)
    args = parser.parse_args()
    result = preview(args.checkpoint, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
