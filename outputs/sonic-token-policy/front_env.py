"""Front feed over a physical net, own-side bounce, hit and legal return."""
import hashlib
import json
import numpy as np
import mujoco
from decoder_env import ROOT, TokenReachEnv
from front_model import TABLE, make_front_model

READY = ROOT/'runs/front-lift-v1/ready_seed.npz'
CONFIG = {'stage': 'front_table_v1', 'table': TABLE, 'residual_scale': .35,
          'control_dt_s': .02, 'physics_dt_s': .001, 'forecast_restitution': .8, 'contact_solref': [.016, .1],
          'lift_checkpoint': str(ROOT/'runs/front-lift-v1/policy.pt'),
          'racket_geometry_provisional': True, 'hardware_approved': False}

def config_hash():
    return hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()+READY.read_bytes()).hexdigest()

def sample_cases(rng, n):
    return [{'y_m': float(y), 'vx_m_s': float(v), 'vz_m_s': float(z)}
            for y, v, z in zip(rng.uniform(-.10, .06, n), rng.uniform(-2.75, -2.5, n), rng.uniform(1.25, 1.5, n))]

class FrontEnv:
    def __init__(self):
        self.base = TokenReachEnv(model=make_front_model(), seed_path=READY, steps=175,
                                  point_offset=[.27, 0, 0], goal_radius=1., residual_scale=CONFIG['residual_scale'])
        self.model, self.data = self.base.model, self.base.data; self.dt = self.base.dt
        self.ball_qa = int(self.model.joint('training_ball_joint').qposadr[0])
        self.ball_va = int(self.model.joint('training_ball_joint').dofadr[0])
        self.ball_geom = self.model.geom('training_ball_geom').id
        self.face_geom = self.model.geom('training_face').id
        self.racket_geoms = {self.face_geom, self.model.geom('training_handle').id}
        self.table_geom = self.model.geom('table_top').id
        self.net_geom = self.model.geom('table_net').id
        self.base.on_substep = self._contacts
        self.steps = 175
        self.reset({'y_m': -.025, 'vx_m_s': -2.6, 'vz_m_s': 1.35})

    def face(self):return self.data.geom_xpos[self.face_geom].copy()
    def normal(self):return -self.data.xmat[self.base.wrist].reshape(3, 3)[:, 1]
    def ball_position(self):return self.data.qpos[self.ball_qa:self.ball_qa+3].copy()
    def ball_velocity(self):return self.data.qvel[self.ball_va:self.ball_va+3].copy()

    def reset(self, case, reset_robot=True):
        if not (-.3 <= case['y_m'] <= .3 and -4 <= case['vx_m_s'] <= -1 and 0 <= case['vz_m_s'] <= 3):
            raise ValueError('Invalid front-feed case')
        if reset_robot:self.base.reset([0, 0, 0])
        self.start_time = float(self.data.time); self.base.k = 0
        self.start_face = self.face(); self.target = self.start_face.copy(); self.hit_time = None
        self.bounces = 0; self.hit = False; self.crossed_net = False; self.landed = False
        self.failure = None; self.contact_events = []; self.last_table_contact = -10.
        self.initial_ball = np.array([2.65, case['y_m'], 1.20])
        self.data.qpos[self.ball_qa:self.ball_qa+3] = self.initial_ball
        self.data.qpos[self.ball_qa+3:self.ball_qa+7] = [1, 0, 0, 0]
        self.data.qvel[self.ball_va:self.ball_va+6] = [case['vx_m_s'], 0, case['vz_m_s'], 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.previous_ball = self.ball_position(); self.previous_velocity = self.ball_velocity()
        self.peak_lower = 0.; self.peak_root = 0.; self.min_miss = 10.; self.peak_face_z = self.face()[2]
        self._forecast()
        return self.observation()

    @property
    def time(self):return float(self.data.time-self.start_time)

    def _forecast(self):
        p = self.ball_position(); v = self.ball_velocity(); g = self.model.opt.gravity[2]
        self.tti = max(0., min(2., (self.start_face[0]-p[0])/v[0])) if v[0] < -.05 else 0.
        if self.hit or self.failure:return
        if self.tti > 0:
            t = self.tti
            # Include the first own-side bounce in the observation forecast.
            roots = np.roots([.5*g, v[2], p[2]-(TABLE['height']+.02)])
            bounce = [float(r.real) for r in roots if abs(r.imag) < 1e-7 and 0 < r.real < t
                      and v[2]+g*r.real < 0 and TABLE['near_x'] < p[0]+v[0]*r.real < TABLE['net_x']]
            q = p+v*t; q[2] += .5*g*t*t
            if not self.bounces and bounce:
                tb = min(bounce); after = t-tb
                vz = -(v[2]+g*tb)*CONFIG['forecast_restitution']
                q[2] = TABLE['height']+.02+vz*after+.5*g*after**2
            self.target = q
        r = self.data.xmat[self.base.pelvis].reshape(3, 3)
        self.base.goal = (self.target-self.data.xpos[self.base.pelvis]) @ r

    def observation(self):
        core = self.base.observation(); core[:6] *= .2
        return np.r_[core, self.ball_position()-self.face(), self.ball_velocity()/5,
                     self.normal(), self.tti].astype(np.float32)

    def _contacts(self):
        t = self.time; pos = self.ball_position(); vel = self.ball_velocity()
        for i in range(self.data.ncon):
            c = self.data.contact[i]; pair = (c.geom1, c.geom2)
            if self.ball_geom not in pair and not self.racket_geoms.intersection(pair):continue
            force = np.zeros(6); mujoco.mj_contactForce(self.model, self.data, i, force)
            if force[0] <= 1e-5:continue
            if self.ball_geom not in pair:
                names = [self.model.geom(g).name for g in pair]
                if any(n in ('training_face', 'training_handle') for n in names):
                    self.failure = self.failure or 'racket_obstacle_contact'
                continue
            other = pair[1] if pair[0] == self.ball_geom else pair[0]
            if other == self.table_geom:
                if t-self.last_table_contact < .08:continue
                self.last_table_contact = t
                if self.hit:
                    if self.crossed_net and TABLE['net_x'] < pos[0] < TABLE['far_x'] and abs(pos[1]) < TABLE['width']/2:
                        self.landed = True
                    else:self.failure = self.failure or 'return_wrong_side'
                else:
                    if TABLE['near_x'] < pos[0] < TABLE['net_x']:
                        self.bounces += 1
                        if self.bounces > 1:self.failure = self.failure or 'second_bounce'
                    else:self.failure = self.failure or 'invalid_feed_bounce'
                self.contact_events.append({'type': 'table', 'time_s': t, 'position_m': pos.tolist()})
            elif other == self.face_geom:
                n = self.normal()
                if abs(float(c.frame[:3] @ n)) < .8:
                    self.failure = self.failure or 'racket_rim'
                elif self.bounces != 1:self.failure = self.failure or 'hit_before_bounce'
                elif not self.hit:
                    self.hit = True; self.hit_time = t
                    self.contact_events.append({'type': 'face', 'time_s': t, 'position_m': pos.tolist(),
                                                'velocity_before_m_s': self.previous_velocity.tolist()})
            else:
                name = self.model.geom(other).name
                self.failure = self.failure or ('net_contact' if other == self.net_geom else 'ball_obstacle_'+name)
        if self.hit and self.previous_ball[0] < TABLE['net_x'] <= pos[0] and vel[0] > 0:
            fraction = (TABLE['net_x']-self.previous_ball[0])/(pos[0]-self.previous_ball[0])
            at_net = self.previous_ball+fraction*(pos-self.previous_ball)
            if at_net[2] > TABLE['height']+TABLE['net_height']+.02 and abs(at_net[1]) <= TABLE['width']/2:
                self.crossed_net = True
        self.previous_ball = pos; self.previous_velocity = vel

    def step(self, action):
        previous = (self.hit, self.crossed_net, self.landed)
        self._forecast(); _, _, done, info = self.base.step(action); self._forecast()
        error = float(np.linalg.norm(self.target-self.face()))
        self.min_miss = min(self.min_miss, float(np.linalg.norm(self.ball_position()-self.face())))
        self.peak_lower = max(self.peak_lower, info['lower_body_max_rad'])
        self.peak_root = max(self.peak_root, info['root_drift_cm'])
        self.peak_face_z = max(self.peak_face_z, self.face()[2])
        reward = (-error/.15 if not self.hit else 0.)-.3*(info['lower_body_max_rad']/.15)**2
        reward -= .01*float(np.mean(self.base.last_residual**2))
        if self.hit and not previous[0]:reward += 15.
        if self.crossed_net and not previous[1]:reward += 30.
        if self.landed and not previous[2]:reward += 60.
        if self.hit:reward += .1*np.clip(self.ball_velocity()[0], -4, 5)
        if info['unsafe']:self.failure = self.failure or 'posture_limit'; reward -= 40.
        if self.failure:reward -= 15.; done = True
        if self.landed:done = True
        if self.ball_position()[0] < .3 or self.ball_position()[2] < .04:done = True
        if done and not self.hit:reward -= 10.*min(self.min_miss/.1, 3)
        info.update({'face_contact': self.hit, 'crossed_net': self.crossed_net,
            'legal_return': self.landed and self.failure is None, 'failure': self.failure,
            'own_bounces': self.bounces, 'minimum_miss_cm': self.min_miss*100,
            'lower_body_peak_rad': self.peak_lower, 'root_drift_peak_cm': self.peak_root,
            'face_error_cm': error*100, 'peak_face_height_cm': self.peak_face_z*100})
        return self.observation(), float(reward), done, info
