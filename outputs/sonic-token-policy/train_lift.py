"""Learn a 64D-token lifting policy from SONIC-encoded demonstrations."""
import json
import numpy as np
import torch
from torch import nn
from decoder_env import TokenReachEnv, ROOT, DEFAULT
from front_model import make_front_model
from lift_teacher import MotionEncoder, ready_pose, reference

class LiftPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(132, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 64))
    def forward(self, x):return self.net(x)

def face(env):
    return env.data.xpos[env.wrist]+env.data.xmat[env.wrist].reshape(3, 3)@np.array([.27, 0, 0])

def observation(env, last_token, time_s):
    d = env.data
    return np.r_[face(env)/.5, d.qpos[env.bqa]-DEFAULT, d.qvel[env.bva]*.1,
                 -d.xmat[env.pelvis].reshape(3, 3)[2], last_token,
                 (np.array([.80, 0, .95])-face(env))/.5, min(time_s/3, 1)].astype(np.float32)

def new_env():return TokenReachEnv(model=make_front_model(), point_offset=[.27, 0, 0], goal_radius=1, steps=250)

def main():
    out = ROOT/'runs/front-lift-v1'; out.mkdir(parents=True, exist_ok=True)
    if (out/'policy.pt').exists():raise SystemExit('Run already exists')
    torch.set_num_threads(1); torch.manual_seed(812); rng = np.random.default_rng(812)
    env = new_env(); encoder = MotionEncoder(); q0 = env.data.qpos[env.bqa].copy(); q1, ik = ready_pose(env)
    r0 = env.data.xmat[env.pelvis].reshape(3, 3).copy(); xs = []; ys = []
    for episode in range(16):
        env.reset([0, 0, 0]); last = env.base_token.copy()
        # Small reference variations teach feedback around the lifting path.
        target = q1.copy(); target[22:] += rng.normal(0, .008, 7)
        for k in range(225):
            x = observation(env, last, k*.02)
            future = [reference(k*.02+i*.1, q0, target) for i in range(10)]
            token = encoder.encode(np.array([q for q, dq in future]), np.array([dq for q, dq in future]),
                                   env.data.xmat[env.pelvis].reshape(3, 3).T@r0)
            xs.append(x); ys.append(token); env.base_token = token
            _, _, _, info = env.step(np.zeros(64)); last = token
            if info['unsafe']:raise RuntimeError('Invalid demonstration')
        print('demonstration', episode+1, face(env).tolist(), flush=True)
    x = torch.tensor(np.array(xs)); y = torch.tensor(np.array(ys)); policy = LiftPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=.001)
    for update in range(2500):
        ix = torch.randint(len(x), (256,)); loss = (policy(x[ix])-y[ix]).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    frames = []; env.reset([0, 0, 0]); last = env.base_token.copy(); peak_lower = 0.
    # Reset the initial token to the original recorded seed, not the last demo.
    last = env.seed['token'].copy()
    for k in range(250):
        with torch.no_grad():token = policy(torch.from_numpy(observation(env, last, k*.02))).numpy().clip(-.6, .6)
        env.base_token = token; _, _, _, info = env.step(np.zeros(64)); last = token
        peak_lower = max(peak_lower, info['lower_body_max_rad'])
        frames.append({'time_s': float(env.data.time), 'qpos': env.data.qpos.tolist(),
                       'face_m': face(env).tolist(), 'unsafe': info['unsafe']})
        if info['unsafe']:break
    normal = -env.data.xmat[env.wrist].reshape(3, 3)[:, 1]
    passed = not info['unsafe'] and face(env)[2] > .88 and face(env)[0] > .65 and normal[0] > .6
    report = {'method': 'Behavior cloning of SONIC-encoded lifting demonstrations', 'samples': len(x),
        'final_token_mse': float(loss), 'final_face_m': face(env).tolist(), 'normal': normal.tolist(),
        'passed_simulation_lift_check': bool(passed), 'peak_lower_rad': peak_lower, 'ik_reference': ik,
        'simulation_only': True, 'hardware_approved': False}
    torch.save({'state_dict': policy.state_dict(), 'simulation_only': True, 'hardware_approved': False,
                'input_dim': 132, 'output_dim': 64}, out/'policy.pt')
    np.savez(out/'ready_seed.npz', qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
             history=np.array(env.history), token=env.base_token)
    (out/'report.json').write_text(json.dumps(report, indent=2))
    (out/'rollout.json').write_text(json.dumps(frames))
    print('FINAL', json.dumps(report), flush=True)

if __name__ == '__main__':main()
