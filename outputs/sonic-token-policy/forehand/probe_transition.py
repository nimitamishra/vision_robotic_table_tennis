"""Offline transition audit from measured joints through frozen SONIC, no transport."""
import argparse
from common import *
from decoder_env import TO_MJ


def reference(times,start,goal,duration,delay=1.):
    t=np.asarray(times);u=np.clip((t-delay)/duration,0,1)
    f=10*u**3-15*u**4+6*u**5
    df=(30*u**2-60*u**3+30*u**4)/duration
    return start+f[...,None]*(goal-start),df[...,None]*(goal-start)


def initialize(snapshot,supported=True,dt=.001,history_mode='pose_hold',ground_feet=False,stock_hand_contacts=True):
    env=TokenReachEnv(model=model(dt,supported=supported),steps=1000,point_offset=CONTACT,goal_radius=1.)
    m,d=env.model,env.data
    # Sensitivity experiment only: retain stock hand inertia, disable collisions
    # from the Dex3 fingers absent on this robot. This is not a gripper model.
    if not stock_hand_contacts:
        for i in range(m.ngeom):
            name=m.body(int(m.geom_bodyid[i])).name or ''
            if name.startswith(('left_hand_','right_hand_')):
                m.geom_contype[i]=m.geom_conaffinity[i]=0
    q=np.asarray(snapshot['measured_q_rad']);dq=np.asarray(snapshot['measured_dq_rad_s'])
    quat=np.asarray(snapshot['imu_quaternion_wxyz']);gyro=np.asarray(snapshot['imu_gyroscope_rad_s'])
    if q.shape!=(29,) or dq.shape!=(29,) or quat.shape!=(4,) or gyro.shape!=(3,) or not np.isfinite(np.r_[q,dq,quat,gyro]).all() or abs(np.linalg.norm(quat)-1)>.02:
        raise ValueError('Invalid measured snapshot')
    r=Rotation.from_quat(quat[[1,2,3,0]]);yaw=r.as_euler('xyz')[2]
    # Table is the training scene, not a registered real table. Remove global yaw.
    qq=(Rotation.from_euler('z',-yaw)*r).as_quat()[[3,0,1,2]]
    d.qpos[:3]=[0,0,.79];d.qpos[3:7]=qq
    d.qpos[env.bqa]=q;d.qvel[:]=0;d.qvel[env.bva]=dq;d.qvel[3:6]=gyro
    ball=int(m.joint('training_ball_joint').qposadr[0]);d.qpos[ball:ball+7]=[8,0,2,1,0,0,0]
    if supported:d.mocap_pos[0]=d.qpos[:3];d.mocap_quat[0]=d.qpos[3:7]
    mujoco.mj_forward(m,d)
    env.initial_foot_bottoms_m={}
    if ground_feet:
        bottoms=[];foot_names=[]
        for i in range(m.ngeom):
            name=m.body(int(m.geom_bodyid[i])).name or ''
            if 'ankle_roll' in name and m.geom_type[i]==mujoco.mjtGeom.mjGEOM_BOX and m.geom_contype[i]:
                bottoms.append(d.geom_xpos[i,2]-np.abs(d.geom_xmat[i].reshape(3,3)[2])@m.geom_size[i])
                foot_names.append(name)
        if len(bottoms)!=2:raise ValueError('Expected two foot collision boxes')
        shift=.001-min(bottoms)
        d.qpos[2]+=shift
        env.initial_foot_bottoms_m=dict(zip(foot_names,[float(x+shift) for x in bottoms]))
        if supported:d.mocap_pos[0]=d.qpos[:3]
        mujoco.mj_forward(m,d)
    action=((q-DEFAULT)/SCALE)[TO_ISAAC] if history_mode=='pose_hold' else np.zeros(29)
    frame=np.r_[gyro,(q-DEFAULT)[TO_ISAAC],dq[TO_ISAAC],action,-d.xmat[env.pelvis].reshape(3,3)[2,:]]
    env.history=deque([frame.copy() for _ in range(10)],maxlen=10);env.last_action=action.copy()
    env.start_joints=q.copy();env.start_root=d.qpos[:3].copy();env.start=env.point().copy();env.goal=env.start.copy()
    return env


def geometry_pairs(env):
    m=env.model;attachments=[m.geom(n).id for n in ('training_face','training_handle')]
    pairs=[]
    for a in attachments:
        for b in range(m.ngeom):
            body=int(m.geom_bodyid[b]);name=m.body(body).name or ''
            if b in attachments or not m.geom_contype[b] or name.startswith('right_hand_') or name=='right_wrist_yaw_link' or name=='training_ball':continue
            pairs.append((a,b))
    return pairs


def run(snapshot,name,duration,hold=False,supported=True,dt=.001,history_mode='pose_hold',record=True,goal_fraction=1.,goal_override=None,round_trip=False,ground_feet=False,stock_hand_contacts=True):
    env=initialize(snapshot,supported,dt,history_mode,ground_feet,stock_hand_contacts);m,d=env.model,env.data;enc=MotionEncoder()
    q0=d.qpos[env.bqa].copy();q1=q0.copy()
    if not hold:q1[22:]=q0[22:]+goal_fraction*(np.load(HERE/'runs/return-v4/forward_reference.npz')['q'][0,22:]-q0[22:])
    if goal_override is not None:q1=np.asarray(goal_override).copy()
    pairs=geometry_pairs(env);frames=[];peak_speed=0.;peak_target_jump=0.;peak_leg=0.;min_distance=10.;closest=None;collisions=[]
    initial_root=d.qpos[:3].copy();initial_rotation=d.xmat[env.pelvis].reshape(3,3).copy();first_fault=None;previous_target=q0.copy()
    n=round((1+duration*(2 if round_trip else 1)+1)/.02) if not hold else 100
    def substep():
        nonlocal peak_speed
        peak_speed=max(peak_speed,float(np.abs(d.qvel[env.bva[22:]]).max()))
        for c in d.contact:
            pair=tuple(sorted((int(c.geom1),int(c.geom2))))
            if pair in checked_pairs and c.dist<-.001:
                if len(collisions)<25:collisions.append({'time_s':float(d.time),'pair':[m.geom(g).name or m.body(int(m.geom_bodyid[g])).name for g in pair],'penetration_m':float(-c.dist)})
    checked_pairs={tuple(sorted(p)) for p in pairs};env.on_substep=substep
    for k in range(n):
        times=d.time+np.arange(10)*.1
        q,dq=reference(times,q0,q1,duration)
        if round_trip:
            returning=times>1+duration
            qr,dqr=reference(times,q1,q0,duration,delay=1+duration)
            q[returning]=qr[returning];dq[returning]=dqr[returning]
        token=enc.encode(q,dq,d.xmat[env.pelvis].reshape(3,3).T@initial_rotation)
        env.base_token=token;env.step(np.zeros(64))
        target=DEFAULT+SCALE*env.last_action[TO_MJ]
        peak_target_jump=max(peak_target_jump,float(np.abs(target[22:]-previous_target[22:]).max()));previous_target=target.copy()
        peak_leg=max(peak_leg,float(np.abs(d.qpos[env.bqa[:15]]-q0[:15]).max()))
        for a,b in pairs:
            dist=float(mujoco.mj_geomDistance(m,d,a,b,1.,None))
            if dist<min_distance:min_distance=dist;closest=[m.geom(g).name or m.body(int(m.geom_bodyid[g])).name for g in (a,b)]
        if peak_speed>1.0 and first_fault is None:first_fault={'time_s':float(d.time),'reason':'right-arm speed exceeds provisional slow-trial limit of 1 rad/s'}
        if collisions and first_fault is None:first_fault={'time_s':float(d.time),'reason':'provisional racket collision geometry penetrates an obstacle'}
        if record:frames.append({'time_s':float(d.time),'qpos':d.qpos.tolist(),'token':token.tolist(),'right_q_rad':d.qpos[env.bqa[22:]].tolist(),'right_dq_rad_s':d.qvel[env.bva[22:]].tolist(),'reference_right_q_rad':q[0,22:].tolist(),'face_m':d.geom_xpos[m.geom('training_face').id].tolist(),'min_distance_so_far_m':min_distance})
        if getattr(env,'stop_requested',False):break
        if (not np.isfinite(d.qpos).all() or d.qpos[2]<.45 or np.linalg.norm(d.qpos[:2]-initial_root[:2])>.3 or np.abs(d.qvel[env.bva]).max()>12):break
    goal_err=float(np.degrees(np.abs(d.qpos[env.bqa[22:]]-(q0 if round_trip else q1)[22:]).max()))
    # Translating the root to the lowest foot does not guarantee BOTH feet touch.
    ground_consistent=not ground_feet or all(abs(x)<=.002 for x in env.initial_foot_bottoms_m.values())
    report={'name':name,'simulation_only':True,'hardware_ready':False,'supported':supported,'ground_feet':ground_feet,'initial_foot_bottoms_m':env.initial_foot_bottoms_m,'both_feet_near_ground':ground_consistent if ground_feet else None,'stock_hand_contacts':stock_hand_contacts,'stock_hand_inertia_retained':True,'physics_dt_s':dt,'history_mode':history_mode,'transition_duration_s':duration,'goal_fraction':goal_fraction,'simulated_s':float(d.time),
        'completed':k==n-1,'peak_right_speed_rad_s':peak_speed,'peak_target_step_deg':float(np.degrees(peak_target_jump)),
        'peak_lower_body_change_deg':float(np.degrees(peak_leg)),'minimum_racket_distance_cm':min_distance*100,'closest_pair':closest,
        'final_max_right_error_deg':goal_err,'first_fault':first_fault,'collision_samples':collisions,
        'diagnostic_pass':bool(ground_consistent and k==n-1 and peak_speed<=1.0 and min_distance>=.02 and goal_err<5 and peak_leg<.15),
        'assumptions':['Measured 29 joint positions, velocities and IMU; synthesized policy history.','Rigid pelvis support approximation, not a calibrated hoist.' if supported else 'Unassisted floating base; root height assumed 0.79 m.', 'Training table pose is unregistered to real robot; racket mount and retained Dex3 hands are provisional.','No motor commands or token packets are transmitted.']}
    return report,frames


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    snapshot=json.loads(a.snapshot.read_text());a.output.mkdir(parents=True,exist_ok=True)
    configs=[('hold_pose_history',2,True,True,.001,'pose_hold',1),('hold_zero_history',2,True,True,.001,'zero',1),
             ('lift_2s',2,False,True,.001,'pose_hold',1),('lift_4s',4,False,True,.001,'pose_hold',1),('lift_6s',6,False,True,.001,'pose_hold',1),
             ('lift_6s_halfstep',6,False,True,.0005,'pose_hold',1),('lift_6s_free_base',6,False,False,.001,'pose_hold',1),('lift_small_6s',6,False,True,.001,'pose_hold',.1)]
    reports=[]
    for name,duration,hold,support,dt,h,f in configs:
        report,frames=run(snapshot,name,duration,hold,support,dt,h,goal_fraction=f)
        (a.output/(name+'.json')).write_text(json.dumps({'report':report,'frames':frames}))
        reports.append(report);print(json.dumps({k:report[k] for k in ('name','simulated_s','peak_right_speed_rad_s','minimum_racket_distance_cm','final_max_right_error_deg','diagnostic_pass')}),flush=True)
    (a.output/'report.json').write_text(json.dumps({'simulation_only':True,'snapshot_sha256':sha(a.snapshot),'encoder_sha256':sha(DECODER.parent/'model_encoder.onnx'),'decoder_sha256':sha(DECODER),'results':reports},indent=2))
if __name__=='__main__':main()
