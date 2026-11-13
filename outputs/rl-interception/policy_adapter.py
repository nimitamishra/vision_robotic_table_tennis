"""Pure, bounded ball-policy inference to SONIC VR targets. No network or motors."""
from pathlib import Path
import numpy as np
import torch
from ball_proxy_env import G
from train_ball_proxy import ActorCritic,features


def load_policy(checkpoint):
    payload=torch.load(Path(checkpoint),map_location='cpu',weights_only=True)
    if payload.get('architecture')!='ActorCritic 27-64-64-(3,1)':
        raise ValueError('Expected the 27-input ball policy checkpoint')
    policy=ActorCritic();policy.load_state_dict(payload['state_dict']);policy.eval()
    return policy


def right_hand_target(policy, q, dq, wrist, ready, ball_position, ball_velocity,
                      time_to_intercept, previous_target=None, dt=None):
    q=np.asarray(q,float);dq=np.asarray(dq,float)
    wrist=np.asarray(wrist,float);ready=np.asarray(ready,float)
    ball_position=np.asarray(ball_position,float);ball_velocity=np.asarray(ball_velocity,float)
    if any(x.shape!=shape or not np.all(np.isfinite(x)) for x,shape in
           [(q,(7,)),(dq,(7,)),(wrist,(3,)),(ready,(3,)),(ball_position,(3,)),(ball_velocity,(3,))]):
        raise ValueError('Finite 7-joint state and 3D points required')
    t=float(time_to_intercept)
    if not np.isfinite(t) or not 0<t<=.5:raise ValueError('Fresh intercept within 0.5 s required')
    crossing=ball_position+ball_velocity*t+.5*G*t*t
    obs=np.r_[q,dq,wrist-ready,ball_position-ready,ball_velocity,t,crossing-ready].astype(np.float32)
    with torch.no_grad():action=policy(features(obs))[0].numpy()
    target=ready+np.clip(action,-1.,1.)*.05
    if previous_target is not None:
        previous=np.asarray(previous_target,float)
        if previous.shape!=(3,) or not np.all(np.isfinite(previous)) or dt is None or not np.isfinite(dt) or dt<=0:
            raise ValueError('Valid previous target and elapsed time required')
        target=previous+np.clip(target-previous,-.03*min(.1,dt),.03*min(.1,dt))
    return target


def vr_target_vectors(left, right, head, left_q, right_q, head_q):
    points=[np.asarray(x,float) for x in (left,right,head)]
    quats=[np.asarray(x,float) for x in (left_q,right_q,head_q)]
    if any(x.shape!=(3,) or not np.all(np.isfinite(x)) for x in points):
        raise ValueError('Three finite pelvis-relative points required')
    if any(x.shape!=(4,) or not np.all(np.isfinite(x)) or abs(np.linalg.norm(x)-1)>.02 for x in quats):
        raise ValueError('Three unit wxyz quaternions required')
    return np.concatenate(points).tolist(),np.concatenate(quats).tolist()


def encode_sonic_planner(left,right,head,left_q,right_q,head_q):
    """Serialize targets for the existing SONIC ZMQ sender; never transmits."""
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_planner_message
    positions,orientations=vr_target_vectors(left,right,head,left_q,right_q,head_q)
    return build_planner_message(0,[0,0,0],[1,0,0],speed=0,
                                 vr_3pt_position=positions,vr_3pt_orientation=orientations)
