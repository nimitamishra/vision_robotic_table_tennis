"""Offline local token-response probe, no hardware transport."""
import numpy as np
from decoder_env import TokenReachEnv,ROOT
from front_env import READY
from front_model import make_front_model

def main():
 e=TokenReachEnv(model=make_front_model(),seed_path=READY,steps=50,point_offset=[.27,0,0],residual_scale=.35,goal_radius=1.)
 def rollout(action,steps=40):
  e.reset([0,0,0]);peak=0
  for _ in range(steps):
   _,_,_,i=e.step(action);peak=max(peak,i['lower_body_max_rad'])
  p=e.data.geom_xpos[e.model.geom('training_face').id].copy();n=-e.data.xmat[e.wrist].reshape(3,3)[:,1]
  return np.r_[p,n],peak
 origin,_=rollout(np.zeros(64));jac=np.empty((6,64))
 for j in range(64):
  action=np.zeros(64);action[j]=.15;value,_=rollout(action);jac[:,j]=(value-origin)/.15
 np.savez(ROOT/'runs/front-token-response.npz',origin=origin,jacobian=jac)
 print('origin',origin,flush=True)
 for height in [1.,1.05,1.10]:
  target=np.array([.84,-.025,height,.94,0,.34]);action=np.zeros(64)
  for _ in range(5):
   value,peak=rollout(action);residual=target-value
   change=jac.T@np.linalg.solve(jac@jac.T+np.eye(6)*.001,residual)
   action=np.clip(action+.5*change,-1,1)
  value,peak=rollout(action)
  print('goal',target,'result',value,'lower',peak,'action_max',abs(action).max(),flush=True)
  np.savez(ROOT/f'runs/front-probe-{height}.npz',action=action,target=target,result=value)
if __name__=='__main__':main()
