"""Bounded simulation search for a warm-start token stroke, not hardware control."""
import json
import numpy as np
from decoder_env import ROOT
from front_env import FrontEnv

def main():
 e=FrontEnv();jac=np.load(ROOT/'runs/front-token-response.npz')['jacobian']
 inv=jac.T@np.linalg.inv(jac@jac.T+np.eye(6)*.001)
 ready=np.load(ROOT/'runs/front-probe-1.0.npz')['action']
 forward=inv@np.array([.12,0,0,0,0,0])
 cases=[{'y_m':-.025,'vx_m_s':-2.6,'vz_m_s':1.25}, {'y_m':-.025,'vx_m_s':-2.6,'vz_m_s':1.35}]
 rng=np.random.default_rng(990)
 best=[]
 for trial in range(150):
  # Smooth frontward pulse timed from the observed flight; all actions are tokens.
  gain=rng.uniform(.3,1.);pulse=rng.uniform(.1,1.);center=rng.uniform(.55,.77);width=rng.uniform(.08,.22)
  c=cases[trial%2];e.reset(c)
  for k in range(e.steps):
   t=e.time
   a=gain*ready+pulse*forward*np.exp(-.5*((t-center)/width)**2)
   if t>.88:a*=max(0.,1-(t-.88)/.18)
   _,r,d,i=e.step(a)
   if d:break
  score=100*i['legal_return']+20*i['crossed_net']+5*i['face_contact']-100*i['unsafe']-.1*i['minimum_miss_cm']
  row={'gain':gain,'pulse':pulse,'center':center,'width':width,'score':score,'info':{k:v for k,v in i.items() if k!='token'}}
  best.append(row);best=sorted(best,key=lambda x:x['score'],reverse=True)[:10]
  if trial%25==0:print(trial,json.dumps(best[0]),flush=True)
 (ROOT/'runs/front-stroke-search.json').write_text(json.dumps(best,indent=2));print('FINAL',json.dumps(best[:3]),flush=True)
if __name__=='__main__':main()
