"""Reproducible full-pipeline audit, including held-out goals/times and sensitivity."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from environment import HittingPlaneEnv
from train_residual import ResidualPolicy
from mount import CONFIG
from export_policy import ExportedPolicy

def load_run(run,dt=.001,mass=1.):
 cp=json.loads((run/'policy.json').read_text())
 e=HittingPlaneEnv(dt=dt,mass_scale=mass,policy=ExportedPolicy(run/'policy.ts'))
 digest=hashlib.sha256()
 for path in (run/'policy.json',run/cp['basis_file'],Path(cp['base_checkpoint'])):digest.update(path.read_bytes())
 return e

def summary(rows):
 return {'episodes':len(rows),'completed':sum(r['completed'] for r in rows),'face_contacts':sum(r['face_contact'] for r in rows),'net_crossings':sum(r['net_crossed'] for r in rows),'legal_returns':sum(r['legal_return'] for r in rows),'posture_failures':sum(r['failure']=='posture_limit' for r in rows),'racket_obstacle_failures':sum(r['failure']=='racket_obstacle' for r in rows),'mean_deadline_error_cm':float(np.mean([r['face_error_at_deadline_cm'] for r in rows if r['face_error_at_deadline_cm'] is not None]))}

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();e=load_run(a.run)
 # Inference must never call the motion encoder, even during startup.
 import common
 def no_encoder(*a,**kw):raise AssertionError('Online encoder is forbidden in this audit')
 common.MotionEncoder.__init__=no_encoder
 groups={'training_replay':[dict(y_m=-.38,z_m=.178,deadline_s=1.)],
  'held_out_nearby':[dict(y_m=y,z_m=z,deadline_s=t) for y,z,t in [(-.39,.173,1.),(-.37,.183,1.),(-.385,.188,.96),(-.375,.168,.94),(-.40,.178,.98),(-.36,.178,.92)]],
  'wider_targets':[dict(y_m=y,z_m=z,deadline_s=t) for y,z,t in [(-.48,.128,1.),(-.43,.178,1.),(-.33,.228,1.),(-.38,.078,1.),(-.38,.178,.8)]]}
 rng=np.random.default_rng(81114)
 groups['fresh_test']=[dict(y_m=float(y),z_m=float(z),deadline_s=float(t)) for y,z,t in zip(rng.uniform(-.41,-.35,20),rng.uniform(.158,.198,20),rng.uniform(.90,1.,20))]
 result={'simulation_only':True,'mount':CONFIG,'groups':{},'source_hashes':{}}
 for name,cases in groups.items():
  rows=[]
  for i,case in enumerate(cases):
   r=e.run(**case);rows.append(r)
   (a.run/f'{name}_{i:02d}.json').write_text(json.dumps({'report':r,'events':e.events,'frames':e.frames}))
   print(name,i,json.dumps(r),flush=True)
  result['groups'][name]={'summary':summary(rows),'trials':rows}
 result['sensitivity']=[]
 for dt,mass in [(.0005,1.),(.00025,1.),(.001,.75),(.001,1.25)]:
  probe=load_run(a.run,dt,mass);r=probe.run();result['sensitivity'].append({'physics_dt_s':dt,'attachment_mass_scale':mass,'report':r});print('SENSITIVITY',dt,mass,r['legal_return'],flush=True)
 for f in Path(__file__).parent.glob('*.py'):result['source_hashes'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
 (a.run/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v['summary'] for k,v in result['groups'].items()},indent=2))
if __name__=='__main__':main()
