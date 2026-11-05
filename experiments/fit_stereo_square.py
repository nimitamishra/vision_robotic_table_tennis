import sys,json,msgpack,cv2,numpy as np
from pathlib import Path
from scipy.optimize import least_squares
sys.path.insert(0,'/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/repo/g1')
from stereo_viser import Detector,detect_camera_tags,make_T,average_poses
out=Path('outputs/g1-ball-session');world=json.loads((out/'session-calibration.json').read_text())['world'];det=Detector(families='tag36h11',quad_decimate=1,refine_edges=1)
rows=[]
with (out/'tag-corner-diagnostic-frames.msgpack').open('rb') as f:
 for p in msgpack.Unpacker(f,raw=False,max_buffer_size=10000000):
  cal=p['calibration'];im=cv2.imdecode(np.frombuffer(p['jpg'],np.uint8),0);w=im.shape[1]//2;Tlr=np.array(cal['T_left_right']);tags=[{t.tag_id:t for t in detect_camera_tags(det,im[:,i*w:(i+1)*w],cal[key],world['tag_size_m'],1)} for i,key in [(0,'left_intrinsics'),(1,'right_intrinsics')]]
  if any(3 not in t or 5 not in t for t in tags):continue
  def proj(q,intr):
   K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
   return cv2.projectPoints(q,np.zeros(3),np.zeros(3),K,np.array(intr['distortion']))[0].reshape(-1,2)
  fits={}
  for tid,size in [(3,.0865),(5,.0875)]:
   obj=np.array([[-1,1,0],[1,1,0],[1,-1,0],[-1,-1,0]],float)*size/2
   t0=tags[0][tid];rvec=cv2.Rodrigues(t0.pose_R)[0].ravel();trans=t0.pose_t.ravel()*size/world['tag_size_m'];x0=np.r_[rvec,trans]
   # Determine corner object ordering from the existing decoded tag pose, not stereo matching guesses.
   options=[np.roll(obj[::-1] if rev else obj,k,axis=0) for rev in (False,True) for k in range(4)]
   obj=min(options,key=lambda v:np.sum((proj(v@t0.pose_R.T+trans,cal['left_intrinsics'])-t0.corners)**2))
   def residual(x):
    R=cv2.Rodrigues(x[:3])[0];q=obj@R.T+x[3:];qr=(q-Tlr[:3,3])@Tlr[:3,:3]
    return np.r_[(proj(q,cal['left_intrinsics'])-tags[0][tid].corners).ravel(),(proj(qr,cal['right_intrinsics'])-tags[1][tid].corners).ravel()]
   fit=least_squares(residual,x0,max_nfev=100,loss='soft_l1',f_scale=.5)
   fits[tid]=(make_T(cv2.Rodrigues(fit.x[:3])[0],fit.x[3:]),float(np.sqrt(np.mean(residual(fit.x)**2))))
  current_pose5=average_poses([make_T(tags[0][5].pose_R,tags[0][5].pose_t),Tlr@make_T(tags[1][5].pose_R,tags[1][5].pose_t)])
  Tw=np.array(world['T_world_tag'])@np.linalg.inv(current_pose5)
  Tjoint=np.array(world['T_world_tag'])@np.linalg.inv(fits[5][0])
  rows.append({'tag3_pixel_rms':fits[3][1],'tag5_pixel_rms':fits[5][1],'current_frame_height_cm':float((Tw@fits[3][0])[2,3]*100),'joint_frame_height_cm':float((Tjoint@fits[3][0])[2,3]*100)})
summary={k:float(np.median([r[k] for r in rows])) for k in rows[0]} if rows else {}
summary['n']=len(rows);summary['note']='Diagnostic only: known-square constraint, 8.65cm tag3 midpoint; no live changes. 6.9cm height not used in fitting.'
(out/'joint-stereo-square-diagnosis.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2));print(json.dumps(summary,indent=2))
