"""Record a few deterministic MuJoCo ball-proxy policy rollouts for viewing.

This uses only the fixed-pelvis, seven-joint right arm and an analytic ball.
It never starts SONIC or connects to the robot.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from ball_proxy_env import BallProxyReach
from policy_adapter import load_policy
from train_ball_proxy import features


def record_rollouts(checkpoint: Path, output: Path, seed: int, episodes: int):
    torch.set_num_threads(1)
    policy = load_policy(checkpoint)
    env = BallProxyReach(seed=seed)
    names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, i)
             for i in range(env.model.nbody)]
    result = {
        'mode': 'MuJoCo fixed-pelvis G1 right arm; analytic ball, no impact physics',
        'checkpoint': str(checkpoint.resolve()),
        'seed': seed,
        'body_names': names,
        'parents': env.model.body_parentid.tolist(),
        'episodes': [],
    }
    for episode_id in range(episodes):
        obs = env.reset()
        frames = []
        for step in range(env.steps + 1):
            with torch.no_grad():
                action = policy(features(obs))[0].numpy()
            ball, velocity = env.ball()
            frames.append({
                'step': step,
                'time_s': step * .05,
                'body_positions_m': env.data.xpos.tolist(),
                'ball_m': ball.tolist(),
                'ball_velocity_mps': velocity.tolist(),
                'hand_m': env.point().tolist(),
                'target_m': (env.ready + np.clip(action, -1, 1) * .05).tolist(),
                'crossing_m': env.goal.tolist(),
            })
            if step < env.steps:
                obs, _, _, info = env.step(action)
        error_cm = float(np.linalg.norm(env.point() - env.ball()[0]) * 100)
        result['episodes'].append({'id': episode_id + 1,
                                   'final_proxy_error_cm': error_cm,
                                   'frames': frames})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result))
    print(json.dumps({'output': str(output.resolve()), 'episodes': episodes,
                      'final_proxy_errors_cm': [round(e['final_proxy_error_cm'], 2)
                                                for e in result['episodes']]}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path,
                        default=Path(__file__).parent / 'runs/ball-proxy-02/policy.pt')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parent / 'runs/demo-rollouts.json')
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--episodes', type=int, default=6)
    args = parser.parse_args()
    record_rollouts(args.checkpoint, args.output, args.seed, args.episodes)
