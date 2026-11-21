"""PPO for physical racket contact through the frozen SONIC token decoder."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from decoder_env import ROOT, decoder_hash
from impact_env import CONFIG, ImpactEnv, config_hash, sample_cases
from train import TokenPolicy

class ImpactPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(141, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 64))
        self.critic = nn.Sequential(nn.Linear(141, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.log_std = nn.Parameter(torch.full((64,), -2.4))
    def forward(self, x):
        return self.actor(x), self.critic(x).squeeze(-1)

def initialize(path):
    source = torch.load(path, map_location='cpu', weights_only=True)
    if source['decoder_sha256'] != decoder_hash():raise ValueError('Decoder mismatch')
    policy = ImpactPolicy()
    if source.get('environment_sha256'):
        if source['environment_sha256'] != config_hash():raise ValueError('Environment mismatch')
        policy.load_state_dict(source['state_dict'])
    else:
        old = TokenPolicy(); old.load_state_dict(source['state_dict'])
        with torch.no_grad():
            policy.actor[0].weight.zero_()
            policy.actor[0].weight[:, :131].copy_(old.actor[0].weight)
            policy.actor[0].bias.copy_(old.actor[0].bias)
            for layer in (2, 4):policy.actor[layer].load_state_dict(old.actor[layer].state_dict())
    return policy

def predict(policy, observation):
    observation = np.asarray(observation, dtype=np.float32)
    if observation.shape != (141,) or not np.isfinite(observation).all():raise ValueError('Invalid impact observations')
    with torch.no_grad():action = policy(torch.from_numpy(observation))[0].numpy()
    if action.shape != (64,) or not np.isfinite(action).all():raise ValueError('Invalid token residual')
    return np.clip(action, -1, 1)

def evaluate(env, policy, cases, zero=False, record=False):
    rows, episodes = [], []
    for case in cases:
        obs = env.reset(case); frames = []
        for _ in range(env.steps):
            obs, _, done, info = env.step(np.zeros(64) if zero else predict(policy, obs))
            if record:frames.append({'time_s': float(env.data.time), 'qpos': env.data.qpos.tolist(),
                                     'target_m': env.target.tolist(), 'hit': info['hit']})
            if done:break
        info = {k: v for k, v in info.items() if k != 'token'}
        info['case'] = case; rows.append(info)
        if record:episodes.append({'metrics': info, 'frames': frames, 'contact_events': env.contact_events})
    return {
        'episodes': len(rows), 'face_contacts': sum(r['face_contact'] for r in rows),
        'successful_returns': sum(r['hit'] for r in rows),
        'return_fraction': float(np.mean([r['hit'] for r in rows])),
        'other_contacts': sum(r['other_contact'] for r in rows),
        'unsafe_episodes': sum(r['unsafe'] for r in rows),
        'mean_target_error_cm': float(np.mean([r['face_error_cm'] for r in rows])),
        'max_lower_body_drift_rad': max(r['lower_body_peak_rad'] for r in rows),
        'max_root_drift_cm': max(r['root_drift_peak_cm'] for r in rows),
        'trials': rows,
    }, episodes

def score(report):
    return 100*report['return_fraction'] - 1000*report['unsafe_episodes'] - .2*report['mean_target_error_cm']

def save(out, policy, env, update):
    torch.save({'state_dict': policy.state_dict(), 'architecture': '141-128-128-64 actor',
        'action': '64D residual token through frozen SONIC',
        'base_token': env.base.base_token.tolist(), 'residual_scale': env.base.residual_scale,
        'decoder_sha256': decoder_hash(), 'environment_sha256': config_hash(),
        'environment': CONFIG, 'training_stage': CONFIG['stage'],
        'simulation_only': True, 'hardware_approved': False, 'update': update}, out/'policy.pt')

def compact(report):return {k: v for k, v in report.items() if k != 'trials'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--init', type=Path, default=ROOT/'runs/reach-v2/policy.pt')
    ap.add_argument('--updates', type=int, default=100)
    ap.add_argument('--episodes', type=int, default=16)
    ap.add_argument('--seed', type=int, default=120)
    ap.add_argument('--actor-lr', type=float, default=1e-4)
    ap.add_argument('--residual-scale', type=float, default=.15)
    a = ap.parse_args(); out = a.output; out.mkdir(parents=True, exist_ok=True)
    if (out/'policy.pt').exists():raise SystemExit('Refusing to overwrite existing run')
    torch.set_num_threads(1); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    env = ImpactEnv(residual_scale=a.residual_scale); policy = initialize(a.init)
    initial_scale = float(torch.load(a.init, map_location='cpu', weights_only=True)['residual_scale'])
    if initial_scale != a.residual_scale:
        # Distill the clipped old token into the wider action space. Simply
        # scaling the final layer would change already-saturated actions.
        source = copy.deepcopy(policy).eval()
        clone_opt = torch.optim.Adam(policy.actor.parameters(), lr=2e-4)
        states = []
        for case in sample_cases(rng, 80):
            obs = env.reset(case)
            for _ in range(env.steps):
                states.append(obs.copy())
                action = predict(source, obs)*initial_scale/a.residual_scale
                obs, _, done, _ = env.step(action)
                if done:break
        x_clone = torch.tensor(np.array(states), dtype=torch.float32)
        with torch.no_grad():target = source(x_clone)[0].clamp(-1, 1)*initial_scale/a.residual_scale
        for _ in range(500):
            ix = torch.randint(len(states), (256,))
            loss = (policy.actor(x_clone[ix])-target[ix]).square().mean()
            clone_opt.zero_grad(); loss.backward(); clone_opt.step()
    reference = copy.deepcopy(policy).eval()
    opt = torch.optim.Adam([
        {'params': list(policy.actor.parameters())+[policy.log_std], 'lr': a.actor_lr},
        {'params': policy.critic.parameters(), 'lr': 3e-4}])
    validation = sample_cases(np.random.default_rng(612601), 40)
    warm, _ = evaluate(env, policy, validation)
    best = copy.deepcopy(policy.state_dict()); best_score = score(warm); best_update = 0
    save(out, policy, env, 0)
    print('INITIAL', json.dumps(compact(warm)), flush=True)
    history = []; total_steps = 0; started = time.monotonic()
    for update in range(1, a.updates+1):
        observations = []; actions = []; logps = []; advantages = []; returns = []; outcomes = []
        for case in sample_cases(rng, a.episodes):
            obs = env.reset(case); trajectory = []
            for _ in range(env.steps):
                x = torch.from_numpy(obs)
                with torch.no_grad():
                    mean, value = policy(x); dist = Normal(mean, policy.log_std.exp())
                    action = dist.sample(); logp = dist.log_prob(action).sum()
                obs, reward, done, info = env.step(action.numpy())
                trajectory.append((x, action, logp, float(value), reward)); total_steps += 1
                if done:break
            outcomes.append(info['hit']); adv = 0.; next_value = 0.
            for x, action, logp, value, reward in reversed(trajectory):
                td = reward + .99*next_value-value; adv = td+.99*.95*adv
                observations.append(x); actions.append(action); logps.append(logp)
                advantages.append(adv); returns.append(adv+value); next_value = value
        x = torch.stack(observations); actions = torch.stack(actions); old_logp = torch.stack(logps)
        adv = torch.tensor(advantages, dtype=torch.float32); ret = torch.tensor(returns, dtype=torch.float32)
        adv = (adv-adv.mean())/(adv.std()+1e-8)
        with torch.no_grad():anchor = reference(x)[0]
        for _ in range(4):
            for ix in torch.randperm(len(x)).split(256):
                mean, value = policy(x[ix]); dist = Normal(mean, policy.log_std.exp())
                ratio = (dist.log_prob(actions[ix]).sum(-1)-old_logp[ix]).exp()
                actor_loss = -torch.minimum(ratio*adv[ix], ratio.clamp(.8, 1.2)*adv[ix]).mean()
                loss = actor_loss + .25*(value-ret[ix]).square().mean() + .01*(mean-anchor[ix]).square().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(policy.actor.parameters())+[policy.log_std], .5)
                nn.utils.clip_grad_norm_(policy.critic.parameters(), .5); opt.step()
        row = {'update': update, 'train_return_fraction': float(np.mean(outcomes)), 'steps': total_steps}
        if update % 5 == 0 or update == a.updates:
            val, _ = evaluate(env, policy, validation); row['validation'] = compact(val)
            if score(val) > best_score:
                best_score = score(val); best_update = update; best = copy.deepcopy(policy.state_dict())
                save(out, policy, env, update)
            print(json.dumps(row), flush=True)
        history.append(row)
        (out/'progress.json').write_text(json.dumps({'update': update, 'updates': a.updates,
            'best_update': best_update, 'seconds': time.monotonic()-started, 'history': history}, indent=2))
    policy.load_state_dict(best)
    cases = sample_cases(np.random.default_rng(881337), 120)
    result, _ = evaluate(env, policy, cases)
    baseline, _ = evaluate(env, policy, cases, zero=True)
    previous, _ = evaluate(env, reference, cases)
    _, episodes = evaluate(env, policy, cases[:12], record=True)
    report = {'simulation_only': True, 'hardware_approved': False, 'environment': CONFIG,
        'decoder_sha256': decoder_hash(), 'environment_sha256': config_hash(),
        'algorithm': 'PPO initialized from prior policy; validation-selected checkpoint',
        'residual_scale': a.residual_scale, 'initial_residual_scale': initial_scale,
        'initial_checkpoint': str(a.init.resolve()), 'seed': a.seed,
        'training_steps': total_steps, 'training_episodes': a.updates*a.episodes,
        'best_update': best_update, 'seconds': time.monotonic()-started,
        'warm_start_validation': warm, 'test_policy': result,
        'test_constant_token': baseline, 'test_initial_policy': previous,
        'history': history, 'limitations': [
            'Side-fed 4-12 cm interception offsets, 0.65-0.95 s flights; not table tennis rallies',
            'Assumed edge-on racket at 0.27 m wrist offset; normal, mass and contact parameters uncalibrated',
            'Perfect simulated ball/body state and known gravity; no camera noise or latency',
            'Returning means a forced face contact followed by outward normal speed >0.2 m/s; no placement objective',
            'Stock Dex3 retained with additional 0.20 kg racket; actual gripper/bracket not modeled',
            'All 29 joints driven by SONIC; lower-body motion penalized, not locked',
            'No real-robot command output or physical safety validation']}
    (out/'report.json').write_text(json.dumps(report, indent=2))
    (out/'rollouts.json').write_text(json.dumps({'episodes': episodes, 'simulation_only': True}))
    print('FINAL', json.dumps({k: compact(report[k]) for k in ('test_policy', 'test_constant_token', 'test_initial_policy')}), flush=True)

if __name__ == '__main__':main()
