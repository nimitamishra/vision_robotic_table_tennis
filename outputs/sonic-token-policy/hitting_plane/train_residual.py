"""Episodic CEM RL on a learned 64D token head. Simulation only; no online encoder."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,time,multiprocessing
from pathlib import Path
import numpy as np
from environment import HittingPlaneEnv
from policy import HittingPlanePolicy
from mount import CONFIG
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'forehand/runs/forward-v1/policy.pt'

class ResidualPolicy:
 def __init__(self,base,basis,params):
  self.base=base;self.basis=basis;self.params=np.asarray(params);self.checkpoint_sha256=base.checkpoint_sha256
 def predict(self,target,t):
  if not np.isfinite(t) or not -1.501<=t<=1.001:raise ValueError('Time outside trained interval')
  # Bounded phase lead and low-rank token head depend only on XYZ and remaining time.
  tt=float(np.clip(t-self.params[-1]*.18,-1.5,1))
  gate=np.exp(-.5*((t-.20)/.40)**4)
  return (self.base.predict(target,tt)+gate*(self.params[:-1]@self.basis)).astype(np.float32)

def score(r):
 error=r['face_error_at_deadline_cm'] or 40
 value=-min(r['minimum_ball_face_distance_cm'] or 100,100)-.15*error
 value+=30*r['face_contact']+100*r['net_crossed']+300*r['legal_return']
 if r['rim_contact']:value-=20
 if r['failure'] in ('posture_limit','racket_obstacle'):value-=150
 if r['outgoing']:
  p=np.array(r['outgoing']['position']);v=np.array(r['outgoing']['velocity'])
  if v[0]>.2:
   t=(v[2]+np.sqrt(max(0,v[2]**2+19.62*(p[2]-.78))))/9.81
   landing=p[:2]+t*v[:2]
   value+=30*np.exp(-np.sum(((landing-[2.3,-.3])/[.65,.4])**2))
 return float(value)

def init_worker(basis):
 global env,base,B,envs
 env=HittingPlaneEnv(BASE);base=env.policy;B=basis;envs={.001:env}
 # Warm feed planner cache; costs no robot-policy training samples.
 env.planner.plan([.43,-.38,.781987965106964+.178])

def evaluate(job):
 params,cases=job;rows=[]
 for item in cases:
  case=dict(item);dt=case.pop('physics_dt',.001)
  if dt not in envs:envs[dt]=HittingPlaneEnv(BASE,dt)
  e=envs[dt];e.policy=ResidualPolicy(base,B,params);r=e.run(**case);r['physics_dt_s']=dt;rows.append(r)
 scores=[score(r) for r in rows]
 return float(np.mean(scores)+.25*min(scores)),rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--iterations',type=int,default=30);p.add_argument('--population',type=int,default=32);p.add_argument('--workers',type=int,default=8);p.add_argument('--output',type=Path,default=HERE/'runs/bracket-v1');p.add_argument('--resume',type=Path);p.add_argument('--robust',action='store_true');a=p.parse_args()
 out=a.output;out.mkdir(parents=True,exist_ok=True)
 data=np.load(BASE.parent/'paired.npz');tokens=data['token'];_,sv,vt=np.linalg.svd(tokens-tokens.mean(0),full_matrices=False)
 basis=vt[:12]*.20
 np.save(out/'basis.npy',basis);(out/'mount.json').write_text(json.dumps(CONFIG,indent=2))
 rng=np.random.default_rng(20261120);mean=np.zeros(13);std=np.full(13,.20);best=mean.copy();best_score=-np.inf;history=[];started=time.monotonic()
 if a.resume:mean=np.array(json.loads(a.resume.read_text())['params']);best=mean.copy();std[:]=.08
 cases=[{'y_m':-.38,'z_m':.178,'deadline_s':1.}]
 if a.robust:cases += [dict(y_m=-.38,z_m=.178,deadline_s=1.,physics_dt=.0005),dict(y_m=-.395,z_m=.168,deadline_s=.95),dict(y_m=-.365,z_m=.188,deadline_s=.90)]
 with multiprocessing.get_context('spawn').Pool(a.workers,initializer=init_worker,initargs=(basis,)) as pool:
  baseline=pool.map(evaluate,[(mean,cases)])[0];print('BASELINE',baseline,flush=True)
  for it in range(a.iterations):
   pop=np.clip(rng.normal(mean,std,(a.population,13)),-1,1);pop[0]=best;pop[1]=mean
   pop[-4:]=np.clip(rng.normal(best,.25,(4,13)),-1,1)
   results=pool.map(evaluate,[(v,cases) for v in pop]);scores=np.array([x[0] for x in results]);idx=int(scores.argmax())
   if scores[idx]>best_score:best_score=float(scores[idx]);best=pop[idx].copy();best_report=results[idx][1]
   elite=pop[np.argsort(scores)[-max(4,a.population//5):]]
   mean=.2*mean+.8*elite.mean(0);std=np.clip(.2*std+.8*elite.std(0),.025,.35)
   row={'iteration':it+1,'episodes':(it+1)*a.population*len(cases),'seconds':time.monotonic()-started,'score':best_score,'best':best_report,'std':float(std.mean())};history.append(row)
   cp={'algorithm':'CEM episodic reinforcement learning of low-rank SONIC token residual head','params':best.tolist(),'basis_file':'basis.npy','base_checkpoint':str(BASE),'simulation_only':True,'hardware_approved':False,'training_cases':cases,'training_episodes':(it+1)*a.population,'report':best_report}
   cp['training_episodes']=row['episodes'];cp['resume_from']=str(a.resume) if a.resume else None
   (out/'candidate.json').write_text(json.dumps(cp,indent=2));(out/'progress.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
  (out/'policy.json').write_text(json.dumps(cp,indent=2))
if __name__=='__main__':main()
