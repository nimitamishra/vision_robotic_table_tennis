"""Offline forward-facing demonstrations -> target/time -> 64 SONIC tokens."""
import sys,argparse,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'forehand'))
from common import *
HERE=Path(__file__).resolve().parent
from mount import model as bracket_model
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from train_prior import train

def build(out):
 out.mkdir(parents=True,exist_ok=True)
 ms=motions();e=make_env(ms[0],existing=TokenReachEnv(model=bracket_model(),point_offset=CONTACT,goal_radius=1.));encoder=MotionEncoder();xs=[];ys=[];ids=[];references=[];checks=[]
 limits=e.model.jnt_range[e.model.actuator_trnid[e.body_act[22:],0]].copy();limits[:,0]+=.015;limits[:,1]-=.015
 for index,m in enumerate(ms):
  q=m['dof'].copy()
  for i,t in enumerate(m['time']):
   blend=np.exp(-.5*((t-1.)/.33)**4)
   original=m['wrist_rotation'][i,:,1];normal=(1-blend)*original+blend*np.array([1.,0.,.04]);normal/=np.linalg.norm(normal)
   e.data.qpos[e.bqa]=m['dof'][i];e.data.qpos[:3]=m['root_pos'][i]
   def residual(a):
    e.data.qpos[e.bqa[22:]]=a;mujoco.mj_forward(e.model,e.data)
    return np.r_[12*(face(e)-m['actual_contact_pos'][i]),2*(e.data.xmat[e.wrist].reshape(3,3)[:,1]-normal),.07*(a-m['dof'][i,22:])]
   fit=least_squares(residual,np.clip(m['dof'][i,22:],limits[:,0],limits[:,1]),bounds=(limits[:,0],limits[:,1]),max_nfev=60)
   q[i,22:]=fit.x
  q[:,22:]=gaussian_filter1d(q[:,22:],1,axis=0,mode='nearest');q[-1]=q[0]
  corrected=dict(m,dof=q,dq=np.gradient(q,m['time'],axis=0,edge_order=2))
  e.data.qpos[e.bqa]=q[50];e.data.qpos[:3]=m['root_pos'][50];mujoco.mj_forward(e.model,e.data)
  checks.append({'motion':m['name'],'kinematic_hit_error_cm':float(np.linalg.norm(face(e)-m['meta']['target_hit_position_world'])*100),'hit_face_forward_component':float(e.data.xmat[e.wrist].reshape(3,3)[0,1])})
  for t in m['time']:
   qq,dq=future(corrected,float(t));xs.append(np.r_[m['target'],1-t]);ys.append(encoder.encode(qq,dq,np.eye(3)));ids.append(index)
  references.append(q)
  print('Encoded forward-facing motion',index+1,'/',len(ms),checks[-1],flush=True)
 original=np.load(RUN/'paired.npz',allow_pickle=False)
 np.savez_compressed(out/'paired.npz',x=np.array(xs,np.float32),token=np.array(ys,np.float32),motion_id=np.array(ids),names=np.array([m['name'] for m in ms]),**{k:original[k] for k in ['train_ids','validation_ids','test_ids']})
 np.savez_compressed(out/'references.npz',q=np.array(references),time=ms[0]['time'],targets=np.array([m['target'] for m in ms]))
 meta=json.loads((RUN/'dataset.json').read_text());meta.update(teacher='Public forehand clips with local IK face-orientation correction; labels from frozen SONIC encoder',kinematic_checks=checks,reference_sha256=sha(out/'references.npz'),online_encoder_required=False)
 (out/'dataset.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'runs/forward-v1');p.add_argument('--epochs',type=int,default=1600);p.add_argument('--reuse-data',action='store_true');a=p.parse_args()
 if (a.output/'policy.pt').exists():raise SystemExit('Use a new run directory; checkpoints are immutable')
 if not a.reuse_data:build(a.output)
 train(a.output,a.epochs)
