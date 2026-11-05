import sys,json,time,zmq,msgpack,cv2,numpy as np
from pathlib import Path
sys.path.insert(0,'/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/repo/g1')
from stereo_viser import Detector,detect_camera_tags,make_T,average_poses
from ball_geometry import camera_ray,project_point
out=Path('outputs/g1-ball-session');world=json.loads((out/'session-calibration.json').read_text())['world'];det=Detector(families=world['tag_family'],quad_decimate=1,refine_edges=1)
c=zmq.Context();s=c.socket(zmq.SUB);s.setsockopt(zmq.SUBSCRIBE,b'');s.setsockopt(zmq.CONFLATE,1);s.connect('tcp://192.168.123.164:5560');rows=[];start=time.monotonic()
while len(rows)<60 and time.monotonic()-start<12:
 if not s.poll(500):continue
 p=msgpack.unpackb(s.recv(),raw=False);cal=p['calibration'];im=cv2.imdecode(np.frombuffer(p['jpg'],np.uint8),0);w=im.shape[1]//2;Tlr=np.array(cal['T_left_right'])
 tags=[{t.tag_id:t for t in detect_camera_tags(det,im[:,i*w:(i+1)*w],cal[key],world['tag_size_m'],1)} for i,key in [(0,'left_intrinsics'),(1,'right_intrinsics')]]
 if any(4 not in ts or 5 not in ts for ts in tags):continue
 poses={tid:average_poses([make_T(tags[0][tid].pose_R,tags[0][tid].pose_t),Tlr@make_T(tags[1][tid].pose_R,tags[1][tid].pose_t)]) for tid in (4,5)}
 Tw=np.array(world['T_world_tag'])@np.linalg.inv(poses[5]);centers={};sides={};valid=True
 for tid in (4,5):
  pts=[]
  for l,r in zip(tags[0][tid].corners,tags[1][tid].corners):
   a=camera_ray(l,cal['left_intrinsics']);b=Tlr[:3,:3]@camera_ray(r,cal['right_intrinsics']);o=Tlr[:3,3];ds=np.linalg.lstsq(np.column_stack((a,-b)),o,rcond=None)[0];q=(a*ds[0]+o+b*ds[1])/2;qr=Tlr[:3,:3].T@(q-o)
   err=max(np.linalg.norm(project_point(q,cal['left_intrinsics'])-l),np.linalg.norm(project_point(qr,cal['right_intrinsics'])-r))
   if min(ds)<=0 or err>1:valid=False;break
   pts.append(q)
  if not valid:break
  pts=np.array(pts);centers[tid]=np.mean(pts,axis=0);sides[tid]=np.linalg.norm(pts-np.roll(pts,1,axis=0),axis=1).tolist()
 if not valid:continue
 rows.append({'frame':p['left_id'],'pnp_xyz':(Tw@poses[4])[:3,3].tolist(),'stereo_xyz':(Tw@np.r_[centers[4],1])[:3].tolist(),'stereo_delta':(Tw[:3,:3]@(centers[4]-centers[5])).tolist(),'sides':sides})
s.close(0);c.term();result={'assumed_tape_xyz_m':[-.356,.398,0],'tag_size_m':world['tag_size_m'],'n':len(rows),'frames':rows}
for key in ('pnp_xyz','stereo_xyz','stereo_delta'):
 if rows:result[key]={'median_cm':(np.median([r[key] for r in rows],axis=0)*100).tolist(),'std_mm':(np.std([r[key] for r in rows],axis=0)*1000).tolist()}
if rows:result['stereo_side_lengths_cm']={str(tid):(np.median([r['sides'][tid] for r in rows],axis=0)*100).tolist() for tid in (4,5)}
(out/'tag4-live-check.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='frames'},indent=2))
