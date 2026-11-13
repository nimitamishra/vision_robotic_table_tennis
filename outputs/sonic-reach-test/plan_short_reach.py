"""Offline position-only IK for a short reach from a saved SONIC simulation hold."""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rl-interception'))
from fixed_arm_env import ARM, OFFSET, build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-run', type=Path, required=True)
    parser.add_argument('--delta', type=float, nargs=3, default=[.02, 0, 0])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    delta = np.asarray(args.delta, dtype=float)
    if not np.all(np.isfinite(delta)) or np.linalg.norm(delta) > .05:
        raise SystemExit('Finite reach of at most 5 cm required.')

    with (args.from_run / 'samples.jsonl').open() as stream:
        holds = [row for row in map(json.loads, stream) if row['phase'] == 'hold']
    if not holds:
        raise SystemExit('No hold phase in source run.')
    baseline = holds[-1]
    if baseline['virtual_support'] or baseline['feet_in_contact'] != 2:
        raise SystemExit('Source hold was not grounded without support.')
    q0 = np.asarray(baseline['right_joint_positions_rad'], dtype=float)
    measured = np.asarray(baseline['actual_local_m'][1], dtype=float)
    target = measured + delta

    model = build_model()
    data = mujoco.MjData(model)
    wrist = model.body('right_wrist_yaw_link').id
    pelvis = model.body('pelvis').id
    limits = np.asarray([model.jnt_range[model.joint(name + '_joint').id] for name in ARM])
    lower, upper = limits[:, 0] + .02, limits[:, 1] - .02

    def point(q):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        return data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ OFFSET - data.xpos[pelvis]

    model_gap = float(np.linalg.norm(point(q0) - measured))
    if model_gap > .02:
        raise SystemExit(f'Model and saved hand point differ by {model_gap * 100:.2f} cm.')
    solution = least_squares(
        lambda q: np.r_[point(q) - target, .003 * (q - q0)],
        np.clip(q0, lower, upper), bounds=(lower, upper), max_nfev=400,
    ).x
    error = float(np.linalg.norm(point(solution) - target))
    if error > .005:
        raise SystemExit(f'IK position error {error * 100:.2f} cm exceeds 0.5 cm.')
    result = {
        'simulation_only': True,
        'source_run': str(args.from_run.resolve()),
        'delta_m': delta.tolist(),
        'target_hand_local_m': target.tolist(),
        'initial_right_q_rad': q0.tolist(),
        'ik_right_q_rad': solution.tolist(),
        'model_vs_logged_cm': model_gap * 100,
        'ik_position_error_cm': error * 100,
        'max_joint_change_rad': float(np.max(np.abs(solution - q0))),
        'limitation': 'Position-only IK; no table or racket collision model.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
