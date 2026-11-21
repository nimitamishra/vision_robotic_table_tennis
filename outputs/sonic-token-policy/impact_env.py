"""Side-fed ball contact curriculum. Offline SONIC/MuJoCo, no robot transport.

The assumed edge-on mount is deliberately NOT rotated to look like a ready
forehand. This is a contact exercise beside the robot, below table height.
It is not regulation table tennis and does not validate the installed mount.
"""
import hashlib
import json
import numpy as np
import mujoco
from decoder_env import MODEL, TokenReachEnv

CONFIG = {
    'stage': 'side_feed_contact_v2', 'point_offset_m': [.27, 0, 0],
    'forecast_freezes_on_any_contact': True,
    'wrist_to_grip_m': .12, 'grip_to_face_m': .15,
    'face_normal_wrist': [0, -1, 0], 'face_radius_m': .075,
    'face_thickness_m': .01, 'face_mass_kg': .17, 'handle_mass_kg': .03,
    'ball_radius_m': .02, 'ball_mass_kg': .0027,
    'physics_dt_s': .002, 'control_dt_s': .02, 'residual_scale': .15,
    'goal_radius_m': .145, 'flight_time_s': [.65, .95],
    'incoming_speed_m_s': [.9, 1.4], 'target_radius_m': [.04, .12],
    'solref': [.008, .3], 'solimp': [.95, .99, .001, .5, 2],
    'mount_orientation_measured': False, 'dynamics_measured': False,
    'stock_dex3_retained': True, 'hardware_approved': False,
}

def config_hash():
    return hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()

def make_spec():
    spec = mujoco.MjSpec.from_file(str(MODEL))
    spec.option.timestep = CONFIG['physics_dt_s']
    wrist = spec.body('right_wrist_yaw_link')
    # Separate welded body gives the attachment real mass and inertia, even
    # though the parent wrist has an explicit inertial block in the stock XML.
    paddle = wrist.add_body(name='training_racket')
    paddle.add_geom(name='training_handle', type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    fromto=[.09, 0, 0, .20, 0, 0], size=[.012, 0, 0],
                    mass=CONFIG['handle_mass_kg'], rgba=[.6, .35, .15, 1])
    paddle.add_geom(name='training_face', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=CONFIG['point_offset_m'], quat=[2**-.5, 2**-.5, 0, 0],
                    size=[CONFIG['face_radius_m'], CONFIG['face_thickness_m']/2, 0],
                    mass=CONFIG['face_mass_kg'], rgba=[.8, .08, .08, 1])
    ball = spec.worldbody.add_body(name='training_ball', pos=[0, 0, 2])
    ball.add_freejoint(name='training_ball_joint')
    ball.add_geom(name='training_ball_geom', type=mujoco.mjtGeom.mjGEOM_SPHERE,
                  size=[CONFIG['ball_radius_m'], 0, 0], mass=CONFIG['ball_mass_kg'],
                  contype=2, conaffinity=1, rgba=[1, .9, .2, 1],
                  solref=CONFIG['solref'], solimp=CONFIG['solimp'])
    spec.add_pair(name='ball_face_contact', geomname1='training_face',
                  geomname2='training_ball_geom', condim=3,
                  solref=CONFIG['solref'], solimp=CONFIG['solimp'],
                  friction=[.3, .3, .005, .0001, .0001])
    return spec

def make_model():
    return make_spec().compile()

def sample_cases(rng, n, radius=(.04, .12)):
    angle = rng.uniform(0, 2*np.pi, n)
    r = rng.uniform(*radius, n)
    return [{'offset_uv_m': [float(a), float(b)],
             'flight_time_s': float(t), 'speed_m_s': float(v)}
            for a, b, t, v in zip(r*np.cos(angle), r*np.sin(angle),
                                  rng.uniform(*CONFIG['flight_time_s'], n),
                                  rng.uniform(*CONFIG['incoming_speed_m_s'], n))]

class ImpactEnv:
    """141 observations, 64 token residuals; impacts resolved by MuJoCo."""
    def __init__(self, decoder=None, residual_scale=.15):
        if not .05 <= residual_scale <= .3:raise ValueError('Simulation residual scale must be 0.05–0.30')
        self.base = TokenReachEnv(decoder=decoder, model=make_model(), steps=85,
                                  point_offset=CONFIG['point_offset_m'],
                                  goal_radius=CONFIG['goal_radius_m'], residual_scale=residual_scale)
        self.model, self.data = self.base.model, self.base.data
        self.ball = self.model.body('training_ball').id
        joint = self.model.joint('training_ball_joint')
        self.ball_qa, self.ball_va = int(joint.qposadr[0]), int(joint.dofadr[0])
        self.ball_geom = self.model.geom('training_ball_geom').id
        self.face_geom = self.model.geom('training_face').id
        self.base.on_substep = self._contacts
        self.dt = self.base.dt
        self.reset({'offset_uv_m': [0, 0], 'flight_time_s': .8, 'speed_m_s': 1.1})

    def face(self):
        return self.data.geom_xpos[self.face_geom].copy()

    def ball_position(self):
        return self.data.qpos[self.ball_qa:self.ball_qa+3].copy()

    def ball_velocity(self):
        return self.data.qvel[self.ball_va:self.ball_va+3].copy()

    def reset(self, case):
        uv = np.asarray(case['offset_uv_m'], dtype=float)
        t, speed = float(case['flight_time_s']), float(case['speed_m_s'])
        if (uv.shape != (2,) or not np.isfinite(uv).all() or np.linalg.norm(uv) > .14
                or not .4 <= t <= 1.2 or not .5 <= speed <= 2):
            raise ValueError('Invalid side-feed case')
        self.base.reset([0, 0, 0])
        r = self.data.xmat[self.base.wrist].reshape(3, 3)
        self.normal = r @ np.array(CONFIG['face_normal_wrist'])
        self.u, self.v = r[:, 0].copy(), r[:, 2].copy()
        self.start_face = self.face()
        self.target = self.start_face + self.u*uv[0] + self.v*uv[1]
        self.arrival = t
        self.steps = int(np.ceil((t+.4)/self.dt))
        self.base.steps = self.steps
        g = self.model.opt.gravity.copy()
        p = self.target + self.normal*speed*t
        velocity = -self.normal*speed - .5*g*t
        self.data.qpos[self.ball_qa:self.ball_qa+3] = p
        self.data.qpos[self.ball_qa+3:self.ball_qa+7] = [1, 0, 0, 0]
        self.data.qvel[self.ball_va:self.ball_va+6] = np.r_[velocity, [0, 0, 0]]
        mujoco.mj_forward(self.model, self.data)
        self.contact_events = []; self.other_contact = False
        self.contact_time = None; self.hit = False; self.outgoing_speed = 0.
        self.min_distance = float('inf'); self.peak_lower = 0.; self.peak_root = 0.
        self.previous_velocity = self.ball_velocity()
        self._update_goal()
        return self.observation()

    def _update_goal(self):
        # Intersect the observed ballistic path with the initial racket plane.
        # No launch-time oracle or future simulator state is supplied to policy.
        g = self.model.opt.gravity
        a = float(.5*g @ self.normal)
        b = float(self.ball_velocity() @ self.normal)
        c = float((self.ball_position()-self.start_face) @ self.normal)
        roots = np.roots([a, b, c]) if abs(a) > 1e-9 else ([-c/b] if abs(b) > 1e-9 else [])
        future = [float(np.real(t)) for t in roots if abs(np.imag(t)) < 1e-7
                  and 0 <= np.real(t) <= 2 and b+2*a*np.real(t) < 0]
        self.time_to_plane = min(future) if future else 0.
        if self.contact_time is None and not self.other_contact and future:
            remain = self.time_to_plane
            self.target = (self.ball_position() + self.ball_velocity()*remain
                           + .5*g*remain**2)
        r = self.data.xmat[self.base.pelvis].reshape(3, 3)
        self.base.goal = (self.target-self.data.xpos[self.base.pelvis]) @ r

    def observation(self):
        r = self.data.xmat[self.base.pelvis].reshape(3, 3)
        return np.r_[self.base.observation(),
                     (self.ball_position()-self.face()) @ r,
                     self.ball_velocity() @ r / 5,
                     self.normal @ r,
                     self.time_to_plane].astype(np.float32)

    def _contacts(self):
        # Inspect EVERY 2 ms physics step, not just the 20 ms controller ticks.
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self.ball_geom not in (c.geom1, c.geom2):continue
            other = c.geom2 if c.geom1 == self.ball_geom else c.geom1
            force = np.zeros(6); mujoco.mj_contactForce(self.model, self.data, i, force)
            if force[0] <= 1e-5:continue
            if other != self.face_geom:
                # A later floor landing does not erase a completed return.
                if self.model.geom_bodyid[other] != 0 or not self.hit:
                    self.other_contact = True
                continue
            normal = self.data.geom_xmat[self.face_geom].reshape(3, 3)[:, 2]
            if abs(float(c.frame[:3] @ normal)) < .8:
                self.other_contact = True  # Rim contact is not a face hit.
                continue
            if self.contact_time is None:
                self.contact_time = float(self.data.time)
                self.contact_events.append({'time_s': self.contact_time,
                    'position_m': c.pos.tolist(), 'normal_force_n': float(force[0]),
                    'velocity_before_m_s': self.previous_velocity.tolist()})
        velocity = self.ball_velocity()
        if self.contact_time is not None:
            outward = float(velocity @ self.normal)
            self.outgoing_speed = max(self.outgoing_speed, outward)
            incoming = float(np.array(self.contact_events[0]['velocity_before_m_s']) @ self.normal)
            # Require actual contact, an incoming ball, then outward velocity.
            self.hit |= incoming < -.1 and outward > .2
        self.previous_velocity = velocity

    def step(self, action):
        previous_hit = self.hit
        self._update_goal()
        _, _, done, info = self.base.step(action)
        self._update_goal()
        self.min_distance = min(self.min_distance, float(np.linalg.norm(self.ball_position()-self.face())))
        self.peak_lower = max(self.peak_lower, info['lower_body_max_rad'])
        self.peak_root = max(self.peak_root, info['root_drift_cm'])
        error = float(np.linalg.norm(self.target-self.face()))
        before = self.data.time <= self.arrival+.04 and self.contact_time is None
        reward = (-error/.08 if before else 0.) - .25*(info['lower_body_max_rad']/.1)**2
        reward -= .01*float(np.mean(self.base.last_residual**2))
        if self.hit and not previous_hit:reward += 30.
        if self.other_contact:reward -= 10.; done = True
        if info['unsafe']:reward -= 30.
        if self.contact_time is not None and self.data.time-self.contact_time >= .10:done = True
        if done and not self.hit:reward -= 10.+10.*min(error/.1, 3)
        info.update({'hit': self.hit and not self.other_contact,
            'face_contact': self.contact_time is not None,
            'other_contact': self.other_contact, 'contact_time_s': self.contact_time,
            'outgoing_normal_speed_m_s': self.outgoing_speed,
            'face_error_cm': error*100, 'minimum_ball_face_distance_cm': self.min_distance*100,
            'lower_body_peak_rad': self.peak_lower, 'root_drift_peak_cm': self.peak_root})
        return self.observation(), float(reward), done, info
