"""Post-training contact audit and sensitivity tests, no checkpoint selection."""
import argparse
import hashlib
import json
import numpy as np
import mujoco
import torch
from decoder_env import ROOT
from impact_env import ImpactEnv, sample_cases
from impact_runtime import load_policy, predict

def evaluate(env, policy, cases, latency=0):
    trials = []
    for case in cases:
        obs = env.reset(case); start = env.face(); peak_move = 0.; events = []; queue = []
        def contact_check():
            env._contacts()
            for i in range(env.data.ncon):
                c = env.data.contact[i]
                pair = (c.geom1, c.geom2)
                racket = [g for g in pair if env.model.geom(g).name in ('training_face', 'training_handle')]
                if not racket:continue
                other = pair[1] if pair[0] in racket else pair[0]
                body = env.model.body(env.model.geom_bodyid[other]).name
                if other == env.ball_geom or body == 'training_racket':continue
                force = np.zeros(6); mujoco.mj_contactForce(env.model, env.data, i, force)
                if force[0] > .01 and not events:
                    events.append({'time_s': float(env.data.time), 'other_body': body, 'force_n': float(force[0])})
        env.base.on_substep = contact_check
        for _ in range(env.steps):
            queue.append(obs.copy())
            observed = queue[max(0, len(queue)-1-latency)]
            obs, _, done, info = env.step(predict(policy, observed))
            peak_move = max(peak_move, float(np.linalg.norm(env.face()-start)))
            if done:break
        trials.append({'case': case, 'hit': info['hit'], 'unsafe': info['unsafe'],
            'racket_body_contacts': events, 'peak_face_displacement_cm': peak_move*100,
            'outgoing_normal_speed_m_s': info['outgoing_normal_speed_m_s']})
    return {'episodes': len(trials), 'returns': sum(t['hit'] for t in trials),
            'return_fraction': float(np.mean([t['hit'] for t in trials])),
            'unsafe_episodes': sum(t['unsafe'] for t in trials),
            'racket_body_contact_episodes': sum(bool(t['racket_body_contacts']) for t in trials),
            'median_peak_face_displacement_cm': float(np.median([t['peak_face_displacement_cm'] for t in trials])),
            'maximum_peak_face_displacement_cm': max(t['peak_face_displacement_cm'] for t in trials),
            'trials': trials}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run', type=__import__('pathlib').Path); a = ap.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(a.run/'policy.pt', map_location='cpu', weights_only=True)
    env = ImpactEnv(residual_scale=checkpoint['residual_scale']); policy, _ = load_policy(a.run/'policy.pt', env)
    cases = sample_cases(np.random.default_rng(511712), 60); report = {}
    report['nominal'] = evaluate(env, policy, cases)
    env.model.opt.timestep = .001; env.base.substeps = 20
    report['physics_1ms'] = evaluate(env, policy, cases)
    env.model.opt.timestep = .002; env.base.substeps = 10
    report['observations_delayed_20ms'] = evaluate(env, policy, cases, latency=1)
    alternate = dict(np.load(ROOT/'runs/decoder-validation/seed.npz'))
    for key in ('qpos', 'qvel', 'history'):env.base.seed[key] = alternate[key]
    report['alternate_initial_state'] = evaluate(env, policy, cases)
    report['simulation_only'] = True; report['hardware_approved'] = False
    report['checkpoint_sha256'] = hashlib.sha256((a.run/'policy.pt').read_bytes()).hexdigest()
    report['note'] = 'Fresh cases, not used for model selection. Delay affects all actor observations; SONIC low-level state remains current. No camera-noise model.'
    (a.run/'sensitivity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {n: v for n, v in r.items() if n != 'trials'} for k, r in report.items() if isinstance(r, dict)}, indent=2))

if __name__ == '__main__':main()
