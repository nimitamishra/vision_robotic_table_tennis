"""Finite-difference local token response, entirely in MuJoCo."""
import json,time
import numpy as np
from decoder_env import *

def outcome(env,action):
    env.reset(np.zeros(3))
    for _ in range(env.steps):
        _,_,done,info=env.step(action)
        if done:break
    return np.r_[env.point()-env.start,env.data.qpos[env.bqa[:22]]-env.start_joints[:22]], info

def main():
    out=ROOT/'runs/probe';out.mkdir(parents=True,exist_ok=True)
    env=TokenReachEnv();zero,_=outcome(env,np.zeros(64));jac=np.zeros((25,64));t=time.time()
    for i in range(64):
        a=np.zeros(64);a[i]=.5
        plus,ip=outcome(env,a);minus,im=outcome(env,-a)
        if ip['unsafe'] or im['unsafe']:raise RuntimeError('Unsafe local probe')
        jac[:,i]=plus-minus
        if i%16==15:print('token probe',i+1,'/64',round(time.time()-t,1),flush=True)
    # Penalize disturbance to all legs, waist and the left arm.
    weights=np.r_[np.ones(3),np.full(22,.08)]
    a=jac*weights[:,None]
    inverse=np.linalg.solve(a.T@a+np.eye(64)*1e-6,a.T*weights)
    np.savez(out/'response.npz',jacobian=jac,zero=zero,inverse=inverse)
    results=[]
    for delta in ([.02,0,0],[0,-.02,0],[0,0,.02],[-.02,0,0],[0,.02,0],[0,0,-.02]):
        target=np.r_[delta,np.zeros(22)]-zero
        action=np.clip(inverse@target,-1,1)
        value,info=outcome(env,action)
        results.append({'delta':delta,'action_max':float(abs(action).max()),'error_cm':float(np.linalg.norm(value[:3]-delta)*100),'lower_rad':info['lower_body_max_rad']})
    print(json.dumps(results,indent=2));(out/'report.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
