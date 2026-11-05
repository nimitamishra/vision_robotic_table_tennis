import sys,json,time,zmq,msgpack,cv2,numpy as np
from pathlib import Path
sys.path.insert(0,'/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/repo/g1')
from stereo_viser import Detector,detect_camera_tags,make_T,average_poses
from ball_geometry import camera_ray,project_point,best_stereo_ball
sys.path.insert(0,str(Path("outputs/g1-ball-session").resolve()))
from ball_center import refine_boxes
out=Path('outputs/g1-ball-session');world=json.loads((out/'session-calibration.json').read_text())['world'];det=Detector(families='tag36h11',quad_decimate=1,refine_edges=1)
c=zmq.Context();s=c.socket(zmq.SUB);s.setsockopt(zmq.SUBSCRIBE,b'');s.setsockopt(zmq.CONFLATE,1);s.connect('tcp://127.0.0.1:5561');rows=[];start=time.monotonic()
with (out/'ball-after-move-source.msgpack').open('wb') as f:
 while len(rows)<60 and time.monotonic()-start<15:
  if not s.poll(500):continue
  raw=s.recv();p=msgpack.unpackb(raw,raw=False)
  if 'source_jpg' not in p:continue
  cal=p['calibration'];im=cv2.imdecode(np.frombuffer(p['source_jpg'],np.uint8),0);w=im.shape[1]//2;T=np.array(cal['T_left_right'])
  tags=[{t.tag_id:t for t in detect_camera_tags(det,im[:,i*w:(i+1)*w],cal[key],world['tag_size_m'],1)} for i,key in [(0,'left_intrinsics'),(1,'right_intrinsics')]]
  if not all(t in tags[0] and t in tags[1] for t in (0,5)):continue
  pose=average_poses([make_T(tags[0][5].pose_R,tags[0][5].pose_t),T@make_T(tags[1][5].pose_R,tags[1][5].pose_t)])
  Tw=np.array(world['T_world_tag'])@np.linalg.inv(pose);pts={};valid=True
  for tid in (0,5):
   qs=[]
   for l,r in zip(tags[0][tid].corners,tags[1][tid].corners):
    a=camera_ray(l,cal['left_intrinsics']);b=T[:3,:3]@camera_ray(r,cal['right_intrinsics']);o=T[:3,3];ds=np.linalg.lstsq(np.column_stack((a,-b)),o,rcond=None)[0];q=(a*ds[0]+o+b*ds[1])/2;qr=T[:3,:3].T@(q-o)
    if min(ds)<=0 or max(np.linalg.norm(project_point(q,cal['left_intrinsics'])-l),np.linalg.norm(project_point(qr,cal['right_intrinsics'])-r))>1:valid=False;break
    qs.append((Tw@np.r_[q,1])[:3])
   if not valid:break
   pts[tid]=np.array(qs)
  if not valid:continue
  fit=np.concatenate([pts[t] for t in (0,5)]);center=fit.mean(0);_,sv,V=np.linalg.svd(fit-center);normal=V[-1];normal*=1 if normal[2]>0 else -1
  target=pts[5].mean(0)
  row={'frame':p['frame_id'],'tag3_original_z_mm':float(target[2]*1000),'tag3_plane_height_mm':float((target-center)@normal*1000),'fit_rmse_mm':float(np.sqrt(np.mean(((fit-center)@normal)**2))*1000),'plane_tilt_deg':float(np.degrees(np.arccos(np.clip(normal[2],-1,1)))),'fit_y_extent_cm':float(np.ptp(fit[:,1])*100),'tag3_xyz_cm':(target*100).tolist()}
  row['balls']={}
  for mode in ['box','contour']:
   boxes=p['boxes'] if mode=='box' else [refine_boxes(im[:,i*w:(i+1)*w],p['boxes'][i])[0] for i in (0,1)]
   candidates=[]
   for lb in boxes[0]:
    for rb in boxes[1]:
     ball=best_stereo_ball([lb],[rb],cal)
     if ball is None:continue
     xyz=(Tw@np.r_[ball['xyz_left'],1])[:3]
     if np.all(xyz>=world['play_volume_min_m']) and np.all(xyz<=world['play_volume_max_m']):
      candidates.append((np.prod(ball['confidence'])/(.5+ball['reprojection_error_px']),xyz))
   if candidates:
    xyz=max(candidates,key=lambda x:x[0])[1]
    row['balls'][mode]={'original_cm':float(xyz[2]*100),'xyz_cm':(xyz*100).tolist()}
  if not rows:cv2.imwrite(str(out/'ball-after-move.jpg'),im)
  rows.append(row);f.write(msgpack.packb(p,use_bin_type=True))
s.close(0);c.term();result={'n':len(rows),'fit_tags':[0,5],'held_out_tag':3,'assumption':'All listed tags lie flat on the same tabletop','frames':rows}
if rows:
 result['ball_summary']={}
 for mode in ['box','contour']:
  bs=[r['balls'][mode] for r in rows if mode in r['balls']]
  result['ball_summary'][mode]={'n':len(bs),'median':{k:np.median([b[k] for b in bs],axis=0).tolist() for k in bs[0]} if bs else None}
result.pop('held_out_tag',None)
(out/'ball-after-move-check.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='frames'},indent=2))
