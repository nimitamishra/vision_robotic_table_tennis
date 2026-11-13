"""MuJoCo right-arm-only reach environment; never connects to SONIC or real hardware."""
from pathlib import Path
import mujoco
import numpy as np

SCENE = Path('/home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml')
ARM = ('right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw',
       'right_elbow', 'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw')
OFFSET = np.array([.18, .025, 0.])  # SONIC right VR tracking point, not racket center.


def build_model():
    spec = mujoco.MjSpec.from_file(str(SCENE))
    allowed = set(ARM)
    for sensor in list(spec.sensors): spec.delete(sensor)
    for actuator in list(spec.actuators):
        if actuator.name not in allowed: spec.delete(actuator)
    for joint in list(spec.joints):
        if joint.name not in {name + '_joint' for name in ARM}: spec.delete(joint)
    model = spec.compile()
    if model.nq != 7 or model.nv != 7 or model.nu != 7:
        raise RuntimeError('Expected seven right-arm joints and actuators')
    return model


class FixedArmReach:
    """A preliminary target-following simulator, not an impact or SONIC model."""
    def __init__(self, seed=0, steps=30):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.steps = steps
        self.body_id = self.model.body('right_wrist_yaw_link').id
        self.pelvis_id = self.model.body('pelvis').id
        self.joint_limits = np.array([self.model.jnt_range[self.model.joint(name + '_joint').id] for name in ARM])
        self.force_limits = np.array([self.model.actuator_ctrlrange[self.model.actuator(name).id] for name in ARM])
        self.ready = None
        self.goal = None
        self.joint_target = np.zeros(7)
        self.t = 0
        self.reset()

    def point(self):
        return self.data.xpos[self.body_id].copy() + self.data.xmat[self.body_id].reshape(3,3) @ OFFSET

    def _joint_drive(self):
        q, dq = self.data.qpos.copy(), self.data.qvel.copy()
        tau = 35 * (self.joint_target - q) - 3 * dq + self.data.qfrc_bias
        self.data.ctrl[:] = np.clip(tau, self.force_limits[:,0], self.force_limits[:,1])

    def reset(self, goal=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.
        self.data.qpos[ARM.index('right_elbow')] = .45
        self.data.qpos[ARM.index('right_shoulder_pitch')] = -.2
        self.joint_target = self.data.qpos.copy()
        mujoco.mj_forward(self.model, self.data)
        self.ready = self.point()
        displacement = np.asarray(goal, float) if goal is not None else self.rng.uniform(-.035,.035,3)
        if displacement.shape != (3,) or not np.all(np.isfinite(displacement)) or np.linalg.norm(displacement) > .07:
            raise ValueError('Finite 3D goal within 7 cm required')
        self.goal = self.ready + displacement
        self.t = 0
        return self.observe()

    def observe(self):
        return np.r_[self.data.qpos, self.data.qvel,
                     self.point() - self.ready, self.goal - self.ready,
                     (self.steps-self.t)/self.steps].astype(np.float32)

    def step(self, action):
        action = np.asarray(action, float)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError('Finite 3D action required')
        # Policy requests a pelvis-relative hand target around the captured ready point.
        target = self.ready + np.clip(action, -1, 1) * .05
        world_point = self.point()
        jacp = np.zeros((3,self.model.nv)); jacr = np.zeros((3,self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, world_point, self.body_id)
        error = np.clip(target - world_point, -.06, .06)
        step_q = jacp.T @ np.linalg.solve(jacp @ jacp.T + .005 * np.eye(3), error)
        self.joint_target = np.clip(self.data.qpos + np.clip(step_q, -.04,.04),
                                    self.joint_limits[:,0]+.02,self.joint_limits[:,1]-.02)
        n = max(1,round(.05/self.model.opt.timestep))
        for _ in range(n):
            self._joint_drive()
            mujoco.mj_step(self.model,self.data)
        self.t += 1
        distance = float(np.linalg.norm(self.point() - self.goal))
        reward = -distance*20 - .005*float(np.dot(action,action))
        done = self.t >= self.steps
        return self.observe(),reward,done,{'error_m':distance,'point_m':self.point().tolist()}
