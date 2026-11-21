"""Reference forehand -> SONIC tokens; all code is offline and simulation-only."""
from pathlib import Path
import sys, hashlib, json
from collections import deque
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from decoder_env import TokenReachEnv, DEFAULT, SCALE, TO_ISAAC, DECODER
from lift_teacher import MotionEncoder
from impact_env import make_spec
SOURCE = HERE.parents[2]/'work/table-tennis-motion-kit'
GRID = SOURCE/'examples/forehand_grid'
RUN = HERE/'runs/prior-v1'
CONTACT = np.array([.2365, .0062, 0.])

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def motions():
    result=[]
    for entry in json.loads((GRID/'motion_grid.json').read_text())['entries']:
        folder=GRID/entry['directory']
        with np.load(folder/'reference.npz', allow_pickle=False) as a: data={k:a[k] for k in a.files}
        data['meta']=json.loads((folder/'metadata.json').read_text()); data['name']=entry['directory']
        data['target']=np.array(data['meta']['target_hit_position_world'])-data['root_pos'][50]
        data['dq']=np.gradient(data['dof'], data['time'], axis=0, edge_order=2)
        result.append(data)
    return result

def future(m, t, stretch=1.):
    times=np.clip((t+np.arange(10)*.1)/stretch,0,2.5)
    q=np.stack([np.interp(times,m['time'],m['dof'][:,j]) for j in range(29)],axis=1)
    dq=np.stack([np.interp(times,m['time'],m['dq'][:,j]) for j in range(29)],axis=1)/stretch
    dq[(t+np.arange(10)*.1<0)|(t+np.arange(10)*.1>2.5*stretch)]=0
    return q,dq

def model(dt=.001, ball_contacts=False, supported=False):
    # Dynamics model: stock G1 with an estimated racket matching reference geometry.
    # VICAR is visual-only. Mass/contact values remain provisional, not hardware calibration.
    spec=make_spec();spec.option.timestep=dt
    spec.geom('training_face').pos=[.2365,0,0]
    spec.geom('training_face').size=[.075,.006,0]
    spec.geom('training_handle').fromto=[.09,0,0,.17,0,0]
    spec.pair('ball_face_contact').solref=[.016,.1]
    for name,pos,size,color in [
        ('table_top',[1.73,0,.7475],[1.37,.7625,.0125],[.07,.24,.42,1]),
        ('table_net',[1.73,0,.83625],[.004,.80,.07625],[.8,.85,.9,.6])]:
        spec.worldbody.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_BOX,pos=pos,size=size,rgba=color)
    for i,x in enumerate((.66,2.8)):
        for j,y in enumerate((-.60,.60)):
            spec.worldbody.add_geom(name=f'table_leg_{i}_{j}',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[x,y,.36],size=[.025,.025,.36],rgba=[.2,.2,.2,1])
    if ball_contacts:
        spec.add_pair(name='ball_table_contact',geomname1='table_top',geomname2='training_ball_geom',
            condim=3,solref=[.016,.1],solimp=[.95,.99,.001,.5,2],friction=[.1,.1,.001,.0001,.0001])
    if supported:
        # Diagnostic rigid support only, not a calibrated physical hoist model.
        pelvis=spec.body('pelvis')
        spec.worldbody.add_body(name='transition_support',mocap=True,pos=pelvis.pos,quat=pelvis.quat)
        spec.add_equality(name='transition_pelvis_support',type=mujoco.mjtEq.mjEQ_WELD,
                          name1='pelvis',name2='transition_support',objtype=mujoco.mjtObj.mjOBJ_BODY,
                          solref=[.01,1])
    return spec.compile()

def make_env(m, dt=.001, existing=None):
    env=existing or TokenReachEnv(model=model(dt),steps=1000,point_offset=CONTACT,goal_radius=1.)
    if existing is not None:env.reset(np.zeros(3))
    names=[env.model.joint(int(env.model.actuator_trnid[i,0])).name for i in env.body_act]
    if names!=list(m['joint_names']):raise ValueError('Joint order does not match: map by name before continuing')
    # Only initial conditions are assigned. All subsequent movement uses frozen SONIC + torque dynamics.
    env.data.qpos[:3]=m['root_pos'][0];env.data.qpos[3:7]=m['root_wxyz'][0]
    env.data.qpos[env.bqa]=m['dof'][0];env.data.qvel[:]=0
    ball=env.model.joint('training_ball_joint');qa=int(ball.qposadr[0]);env.data.qpos[qa:qa+3]=[8,0,2]
    env.data.qpos[qa+3:qa+7]=[1,0,0,0]
    mujoco.mj_forward(env.model,env.data)
    action=((m['dof'][0]-DEFAULT)/SCALE)[TO_ISAAC]
    frame=np.r_[np.zeros(3),(m['dof'][0]-DEFAULT)[TO_ISAAC],np.zeros(29),action,[0,0,-1]]
    env.history=deque([frame.copy() for _ in range(10)],maxlen=10);env.last_action=action.copy()
    env.start_joints=env.data.qpos[env.bqa].copy();env.start_root=env.data.qpos[:3].copy()
    env.start=env.point().copy();env.goal=env.start.copy()
    return env

def face(env):
    return env.data.xpos[env.wrist]+env.data.xmat[env.wrist].reshape(3,3)@CONTACT

def rollout(m, predictor=None, stretch=1., dt=.001, warmup=1.):
    env=make_env(m,dt);encoder=MotionEncoder();records=[];initial=env.start_root.copy()
    for k in range(round((warmup+2.5*stretch)/.02)+1):
        t=k*.02-warmup
        q,dq=future(m,t,stretch)
        rot=env.data.xmat[env.pelvis].reshape(3,3)
        if predictor is None:token=encoder.encode(q,dq,rot.T)
        elif t<0:token=encoder.encode(q,dq,rot.T) # common reference warm-up, separately identified
        else:token=predictor(np.r_[m['target'],1-t/stretch])
        env.base_token=np.asarray(token,dtype=np.float32)
        _,_,_,info=env.step(np.zeros(64))
        r=env.data.xmat[env.pelvis].reshape(3,3)
        legs=float(np.max(abs(env.data.qpos[env.bqa[:12]]-m['dof'][0,:12])))
        drift=float(np.linalg.norm(env.data.qpos[:2]-initial[:2]))
        unsafe=bool(env.data.qpos[2]<.55 or r[2,2]<.9 or drift>.10 or legs>.30 or not np.isfinite(env.data.qpos).all())
        collisions=[]
        for ci,c in enumerate(env.data.contact):
            ns=[env.model.geom(int(g)).name or '' for g in (c.geom1,c.geom2)]
            if any(n in ('training_face','training_handle') for n in ns) and not 'training_ball_geom' in ns:
                force=np.zeros(6);mujoco.mj_contactForce(env.model,env.data,ci,force)
                if force[0]>1.:collisions.append(ns)
        phase=np.clip((t+.02)/stretch,0,2.5)
        qi=np.stack([np.interp(phase,m['time'],m['dof'][:,j]) for j in range(29)])
        reference=np.array([np.interp(phase,m['time'],m['actual_contact_pos'][:,j]) for j in range(3)])
        actual=face(env);local=(actual-env.data.xpos[env.pelvis])@r
        records.append(dict(time_s=t+.02,phase_s=float(phase),qpos=env.data.qpos.tolist(),face_m=actual.tolist(),
            face_error_cm=float(100*np.linalg.norm(local-(reference-m['root_pos'][0]))),
            arm_rmse_rad=float(np.sqrt(np.mean((env.data.qpos[env.bqa[22:]]-qi[22:])**2))),
            normal=env.data.xmat[env.wrist].reshape(3,3)[:,1].tolist(),legs_max_rad=legs,root_drift_cm=100*drift,
            unsafe=unsafe,obstacle_contacts=collisions,token=np.asarray(token).tolist()))
        if unsafe:break
    active=[r for r in records if r['time_s']>=0] or records
    hit=min(active,key=lambda r:abs(r['phase_s']-1))
    report=dict(motion=m['name'],stretch=stretch,physics_dt=dt,completed=records[-1]['phase_s']>=2.5,
        unsafe=any(r['unsafe'] for r in records),obstacle_contact_frames=sum(bool(r['obstacle_contacts']) for r in records),
        mean_face_error_cm=float(np.mean([r['face_error_cm'] for r in active])),hit_face_error_cm=hit['face_error_cm'],
        arm_rmse_rad=float(np.mean([r['arm_rmse_rad'] for r in active])),peak_leg_rad=max(r['legs_max_rad'] for r in records),
        root_drift_cm=max(r['root_drift_cm'] for r in records),hit_face_m=hit['face_m'],hit_normal=hit['normal'])
    return records,report
