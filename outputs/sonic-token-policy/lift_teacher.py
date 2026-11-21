"""Build lifting demonstrations: reference pose -> SONIC encoder -> token -> decoder.

IK is used only to construct reference demonstrations. It does not issue joint
commands. Every simulated actuator target is produced by the frozen decoder.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
import onnxruntime as ort
from scipy.optimize import least_squares
from decoder_env import TokenReachEnv, DECODER, TO_ISAAC, ROOT
from front_model import make_front_model

class MotionEncoder:
    def __init__(self):
        o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(DECODER.parent/'model_encoder.onnx'), o, providers=['CPUExecutionProvider'])
        assert self.session.get_inputs()[0].shape == [1, 1762]
    def encode(self, q_future, dq_future, relative_root):
        x = np.zeros(1762, dtype=np.float32)
        # G1 encoder mode 0; positions are absolute Isaac-order joint angles.
        x[4:294] = q_future[:, TO_ISAAC].ravel()
        x[294:584] = dq_future[:, TO_ISAAC].ravel()
        orientation = relative_root[:, :2].ravel()
        x[595:601] = orientation; x[601:661] = np.tile(orientation, 10)
        return self.session.run(None, {'obs_dict': x[None]})[0][0]

def ready_pose(env, goal=(.70, -.25, .98)):
    d, m = env.data, env.model
    initial = d.qpos.copy(); q0 = d.qpos[env.bqa[22:]].copy()
    ids = m.actuator_trnid[env.body_act[22:], 0]
    bounds = (m.jnt_range[ids, 0]+.01, m.jnt_range[ids, 1]-.01)
    def residual(q):
        d.qpos[env.bqa[22:]] = q; mujoco.mj_forward(m, d)
        rot = d.xmat[env.wrist].reshape(3, 3)
        face = d.xpos[env.wrist]+rot@np.array([.27, 0, 0])
        normal = -rot[:, 1]
        return np.r_[(face-goal)*8, (normal-[1, 0, 0])*1.5, .10*(q-q0)]
    starts = [q0, np.array([-.6, -.4, 0, 1.4, -1.3, 0, 0]),
              np.array([-.6, -.4, 0, 1.4, 1.3, 0, 0])]
    rng = np.random.default_rng(405)
    starts.extend(q0+rng.normal(0, .8, 7) for _ in range(15))
    solutions = [least_squares(residual, np.clip(start, bounds[0]+.001, bounds[1]-.001),
                               bounds=bounds, max_nfev=150) for start in starts]
    result = min(solutions, key=lambda x: float(x.fun @ x.fun))
    target = d.qpos[env.bqa].copy(); target[22:] = result.x
    error = residual(result.x)
    d.qpos[:] = initial; mujoco.mj_forward(m, d)
    return target, {'goal_m': list(goal), 'position_error_cm': float(np.linalg.norm(error[:3])/8*100),
                    'normal_error': float(np.linalg.norm(error[3:6])/1.5), 'right_q': result.x.tolist()}

def reference(t, q0, q1, duration=3.):
    u = np.clip(t/duration, 0, 1)
    f = u*u*(3-2*u)
    df = (6*u*(1-u)/duration) if 0 < t < duration else 0.
    return q0+f*(q1-q0), df*(q1-q0)

def run(output):
    output.mkdir(parents=True, exist_ok=True)
    env = TokenReachEnv(model=make_front_model(), point_offset=[.27, 0, 0], goal_radius=1., steps=350)
    encoder = MotionEncoder(); q0 = env.data.qpos[env.bqa].copy(); q1, ik = ready_pose(env)
    r0 = env.data.xmat[env.pelvis].reshape(3, 3).copy(); frames = []; peak_lower = 0.
    for k in range(350):
        future = [reference(k*.02+i*.1, q0, q1) for i in range(10)]
        token = encoder.encode(np.array([q for q, dq in future]), np.array([dq for q, dq in future]),
                               env.data.xmat[env.pelvis].reshape(3, 3).T@r0)
        env.base_token = token
        _, _, done, info = env.step(np.zeros(64))
        r = env.data.xmat[env.wrist].reshape(3, 3)
        face = env.data.xpos[env.wrist]+r@np.array([.27, 0, 0])
        peak_lower = max(peak_lower, info['lower_body_max_rad'])
        frames.append({'time_s': env.data.time, 'qpos': env.data.qpos.tolist(), 'face_m': face.tolist(),
                       'normal': (-r[:, 1]).tolist(), 'token': token.tolist(), 'unsafe': info['unsafe']})
        if k%50 == 0:print(k, face, 'normal', -r[:, 1], 'lower', peak_lower, flush=True)
        if info['unsafe']:break
    report = {'ik': ik, 'final': frames[-1], 'peak_lower_rad': peak_lower, 'simulation_only': True}
    (output/'report.json').write_text(json.dumps(report, indent=2))
    (output/'rollout.json').write_text(json.dumps(frames))
    np.savez(output/'ready_seed.npz', qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
             history=np.array(env.history), token=env.base_token)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, default=ROOT/'runs/front-lift-probe')
    run(p.parse_args().output)
