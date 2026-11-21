"""Offline MuJoCo + frozen SONIC decoder. No DDS, sockets, or robot commands.

Full floating-base 43-DOF stock model; 29 body targets all come from SONIC.
This is a short-reach curriculum, NOT a ping-pong impact environment.
"""
from pathlib import Path
from collections import deque
import json
import hashlib
import numpy as np
import mujoco
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl')
MODEL = REPO/'gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml'
DECODER = REPO/'gear_sonic_deploy/policy/release/model_decoder.onnx'
SEED = ROOT/'runs/decoder-seed/seed.npz'
# Matches policy_parameters.hpp, hardware body order <-> IsaacLab.
TO_MJ = np.array([0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28])
TO_ISAAC = np.argsort(TO_MJ)
DEFAULT = np.array([-.312,0,0,.669,-.363,0]*2+[0,0,0,.2,.2,0,.6,0,0,0,.2,-.2,0,.6,0,0,0])
ARMATURE = np.array([.025101925,.025101925,.010177520,.025101925,.003609725,.003609725]*2 + [.010177520,.003609725,.003609725]+[.003609725]*5+[.00425]*2+[.003609725]*5+[.00425]*2)
STIFFNESS = ARMATURE*(20*np.pi)**2
KP = STIFFNESS.copy()
KD = 4*ARMATURE*20*np.pi
KP[[4,5,10,11,13,14]] *= 2
KD[[4,5,10,11,13,14]] *= 2
SCALE = .25*np.array([139,139,88,139,25,25]*2+[88,25,25]+[25]*5+[5]*2+[25]*5+[5]*2)/STIFFNESS

class Decoder:
    def __init__(self):
        options=ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(DECODER),options,providers=['CPUExecutionProvider'])
        assert self.session.get_inputs()[0].shape == [1,994]
    def __call__(self, obs):
        obs=np.asarray(obs,dtype=np.float32)
        if obs.shape != (994,) or not np.isfinite(obs).all():
            raise ValueError('Decoder requires 994 finite observations')
        out=self.session.run(None,{'obs_dict':obs[None]})[0][0]
        if not np.isfinite(out).all(): raise ValueError('Nonfinite decoder action')
        return out

def split_history(obs):
    p=64; groups=[]
    for width in (3,29,29,29,3):
        groups.append(obs[p:p+10*width].reshape(10,width)); p+=10*width
    return np.concatenate(groups,axis=1)

def pack_observation(token,history):
    h=np.asarray(history)
    return np.concatenate([token,h[:,:3].ravel(),h[:,3:32].ravel(),h[:,32:61].ravel(),h[:,61:90].ravel(),h[:,90:93].ravel()]).astype(np.float32)

class TokenReachEnv:
    dt=.02
    def __init__(self,decoder=None,seed_path=SEED,steps=75,residual_scale=.15,
                 model=None,point_offset=(.18,.025,0),goal_radius=.051):
        self.decoder=decoder or Decoder()
        self.model=model if model is not None else mujoco.MjModel.from_xml_path(str(MODEL))
        if model is None:self.model.opt.timestep=.005
        self.substeps=round(self.dt/self.model.opt.timestep)
        if not np.isclose(self.substeps*self.model.opt.timestep,self.dt):raise ValueError('Physics timestep must divide controller timestep')
        self.point_offset=np.asarray(point_offset,dtype=float);self.goal_radius=goal_radius
        self.on_substep=None
        self.before_substep=None
        self.command_filter=None
        self.data=mujoco.MjData(self.model)
        self.body_act=np.array([i for i in range(self.model.nu) if '_hand_' not in self.model.actuator(i).name],dtype=int)
        self.hand_act=np.array([i for i in range(self.model.nu) if '_hand_' in self.model.actuator(i).name],dtype=int)
        joints=self.model.actuator_trnid[:,0]
        self.qa=self.model.jnt_qposadr[joints]; self.va=self.model.jnt_dofadr[joints]
        self.bqa=self.qa[self.body_act]; self.bva=self.va[self.body_act]
        self.wrist=self.model.body('right_wrist_yaw_link').id
        self.pelvis=self.model.body('pelvis').id
        self.seed=dict(np.load(seed_path)); self.steps=steps; self.residual_scale=residual_scale
        # Match seed joints by name: custom attachment models can omit the Dex3 fingers.
        self.seed_model=mujoco.MjModel.from_xml_path(str(MODEL))
        self.base_token=self.seed['token'].astype(np.float32)
        self.reset(np.zeros(3))
    def point(self):
        d=self.data
        world=d.xpos[self.wrist]+d.xmat[self.wrist].reshape(3,3)@self.point_offset
        return (world-d.xpos[self.pelvis])@d.xmat[self.pelvis].reshape(3,3)
    def reset(self,delta):
        delta=np.asarray(delta,dtype=float)
        if delta.shape != (3,) or not np.isfinite(delta).all() or np.linalg.norm(delta)>self.goal_radius:
            raise ValueError('Reach goal must be finite and within the configured radius')
        mujoco.mj_resetData(self.model,self.data)
        for j in range(self.seed_model.njnt):
            name=self.seed_model.joint(j).name
            dst=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,name)
            if dst<0:continue
            nq,nv=(7,6) if self.seed_model.jnt_type[j]==0 else (4,3) if self.seed_model.jnt_type[j]==1 else (1,1)
            a,b=int(self.seed_model.jnt_qposadr[j]),int(self.model.jnt_qposadr[dst])
            c,e=int(self.seed_model.jnt_dofadr[j]),int(self.model.jnt_dofadr[dst])
            self.data.qpos[b:b+nq]=self.seed['qpos'][a:a+nq]
            self.data.qvel[e:e+nv]=self.seed['qvel'][c:c+nv]
        mujoco.mj_forward(self.model,self.data)
        self.history=deque(self.seed['history'].copy(),maxlen=10)
        self.last_action=self.history[-1][61:90].copy()
        self.last_residual=np.zeros(64); self.k=0
        self.start=self.point().copy(); self.goal=self.start+delta
        self.start_joints=self.data.qpos[self.bqa].copy(); self.start_root=self.data.qpos[:3].copy()
        self.hand_q=self.data.qpos[self.qa[self.hand_act]].copy()
        return self.observation()
    def observation(self):
        # Goal error, body positions/speeds, gravity, previous token residual.
        d=self.data
        return np.r_[(self.goal-self.start)/.05,(self.goal-self.point())/.05,
                     (d.qpos[self.bqa]-DEFAULT)[TO_ISAAC],d.qvel[self.bva][TO_ISAAC]*.1,
                     -d.xmat[self.pelvis].reshape(3,3)[2,:],self.last_residual].astype(np.float32)
    def step(self,residual):
        residual=np.asarray(residual,dtype=float)
        if residual.shape!=(64,) or not np.isfinite(residual).all(): raise ValueError('Expected finite 64D token residual')
        residual=np.clip(residual,-1,1)
        # Bound token slew as well as amplitude; exact same rule in deployment.
        residual=np.clip(residual,self.last_residual-.10,self.last_residual+.10)
        token=self.base_token+self.residual_scale*residual
        d=self.data
        frame=np.r_[d.qvel[3:6],(d.qpos[self.bqa]-DEFAULT)[TO_ISAAC],d.qvel[self.bva][TO_ISAAC],self.last_action,-d.xmat[self.pelvis].reshape(3,3)[2,:]]
        self.history.append(frame)
        decoder_obs=pack_observation(token,self.history)
        action=self.decoder(decoder_obs)
        target=DEFAULT+SCALE*action[TO_MJ]
        for _ in range(self.substeps):
            if self.before_substep is not None:self.before_substep()
            # Optional diagnostic command envelope, evaluated at physics rate.
            # Default path is unchanged. Policy/history still use the raw decoder action.
            command,kp,kd=(target,KP,KD) if self.command_filter is None else self.command_filter(target)
            d.ctrl[self.body_act]=kp*(command-d.qpos[self.bqa])-kd*d.qvel[self.bva]
            d.ctrl[self.hand_act]=5*(self.hand_q-d.qpos[self.qa[self.hand_act]])-.2*d.qvel[self.va[self.hand_act]]
            d.ctrl[:]=np.clip(d.ctrl,self.model.actuator_ctrlrange[:,0],self.model.actuator_ctrlrange[:,1])
            mujoco.mj_step(self.model,d)
            if self.on_substep is not None:self.on_substep()
        mujoco.mj_forward(self.model,d)
        self.last_action=action.copy(); self.k+=1
        error=float(np.linalg.norm(self.point()-self.goal))
        lower=float(np.max(np.abs(d.qpos[self.bqa[:15]]-self.start_joints[:15])))
        root_drift=float(np.linalg.norm(d.qpos[:2]-self.start_root[:2]))
        upright=float(d.xmat[self.pelvis].reshape(3,3)[2,2])
        unsafe=bool(d.qpos[2]<.55 or upright<.9 or root_drift>.10 or lower>.30 or not np.isfinite(d.qpos).all())
        smooth=float(np.mean((residual-self.last_residual)**2))
        reward=-error/.05-.5*(lower/.1)**2-.05*np.mean(residual**2)-.5*smooth
        if unsafe: reward-=20
        self.last_residual=residual
        info={'error_cm':100*error,'lower_body_max_rad':lower,'root_drift_cm':100*root_drift,'unsafe':unsafe,'token':token.tolist()}
        return self.observation(),float(reward),bool(unsafe or self.k>=self.steps),info
    def frame(self):
        return {'time_s':self.k*self.dt,'qpos':self.data.qpos.tolist(),'body_positions':self.data.xpos.tolist(),
                'hand_local_m':self.point().tolist(),'goal_local_m':self.goal.tolist(),
                'root_position':self.data.xpos[self.pelvis].tolist(),'root_rotation':self.data.xmat[self.pelvis].reshape(3,3).tolist()}

def decoder_hash():
    return hashlib.sha256(DECODER.read_bytes()).hexdigest()
