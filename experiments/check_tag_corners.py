import sys,time,json,msgpack,zmq,cv2,numpy as np
from pathlib import Path
sys.path.insert(0,'/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/repo/g1')
from stereo_viser import Detector,detect_camera_tags,make_T,average_poses
from ball_geometry import camera_ray,project_point
out=Path('outputs/g1-ball-session');world=json.loads((out/'session-calibration.json').read_text())['world']
det=Detector(families='tag36h11',quad_decimate=1,refine_edges=1)
c=zmq.Context();s=c.socket(zmq.SUB);s.setsockopt(zmq.SUBSCRIBE,b'');s.setsockopt(zmq.CONFLATE,1);s.connect('tcp://192.168.123.164:5560')
packets=[];start=time.monotonic()
while len(packets)<24 and time.monotonic()-start<15:
 if s.poll(500):packets.append(msgpack.unpackb(s.recv(),raw=False))
s.close(0);c.term()
with (out/'tag-corner-diagnostic-frames.msgpack').open('wb') as f:
 for p in packets:f.write(msgpack.packb(p,use_bin_type=True))
def tri(l,r,cal):
 T=np.array(cal['T_left_right']);a=camera_ray(l,cal['left_intrinsics']);b=T[:3,:3]@camera_ray(r,cal['right_intrinsics']);o=T[:3,3];ds=np.linalg.lstsq(np.column_stack((a,-b)),o,rcond=None)[0]
 if min(ds)<=0:return None
 return (a*ds[0]+o+b*ds[1])/2
results={};order=[];sizes=[]
for p in packets:
 cal=p['calibration'];im=cv2.imdecode(np.frombuffer(p['jpg'],np.uint8),0);w=im.shape[1]//2;imgs=[im[:,:w],im[:,w:]];Tlr=np.array(cal['T_left_right']);tags=[{t.tag_id:t for t in detect_camera_tags(det,imgs[i],cal[key],world['tag_size_m'],1)} for i,key in [(0,'left_intrinsics'),(1,'right_intrinsics')]]
 if any(3 not in ts or 5 not in ts for ts in tags):continue
 pose5=average_poses([make_T(tags[0][5].pose_R,tags[0][5].pose_t),Tlr@make_T(tags[1][5].pose_R,tags[1][5].pose_t)])
 Tw=np.array(world['T_world_tag'])@np.linalg.inv(pose5)
 methods={'current':[{tid:ts[tid].corners for tid in (3,5)} for ts in tags]}
 for scale in (1,2,3,4):
  maps=[]
  for img in imgs:
   enlarged=img if scale==1 else cv2.resize(img,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
   found=det.detect(enlarged,estimate_tag_pose=False)
   maps.append({t.tag_id:(t.corners+.5)/scale-.5 for t in found if t.tag_id in (3,5)})
  if all(3 in m and 5 in m for m in maps):methods[f'scale{scale}_pixel_center']=maps
 base=methods['current'];sub=[];ok=True
 for img,m in zip(imgs,base):
  refined={}
  for tid in (3,5):
   pts=m[tid].astype(np.float32).reshape(-1,1,2)
   pts=cv2.cornerSubPix(img,pts,(2,2),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,30,.001)).reshape(-1,2)
   if np.max(np.linalg.norm(pts-m[tid],axis=1))>1.5:ok=False
   refined[tid]=pts
  sub.append(refined)
 if ok:methods['cornerSubPix_2']=sub
 for name,maps in methods.items():
  pts={tid:np.array([tri(l,r,cal) for l,r in zip(maps[0][tid],maps[1][tid])]) for tid in (3,5)}
  if any(v.dtype==object for v in pts.values()):continue
  xyz=(Tw@np.r_[pts[3].mean(0),1])[:3];delta=Tw[:3,:3]@(pts[3].mean(0)-pts[5].mean(0))
  sides=np.linalg.norm(pts[3]-np.roll(pts[3],1,axis=0),axis=1)
  _,sv,_=np.linalg.svd(pts[3]-pts[3].mean(0))
  results.setdefault(name,[]).append({'xyz':xyz.tolist(),'delta':delta.tolist(),'sides':sides.tolist(),'square_side_rms_mm':float(np.sqrt(np.mean((sides-.0865)**2))*1000),'nonplanarity_mm':float(sv[-1]*1000),'corners_left':maps[0][3].tolist(),'corners_right':maps[1][3].tolist()})
 # Epipolar angular coplanarity residual for the 8 order-preserving/reversing permutations.
 a=np.array([camera_ray(x,cal['left_intrinsics']) for x in base[0][3]])
 b=np.array([Tlr[:3,:3]@camera_ray(x,cal['right_intrinsics']) for x in base[1][3]])
 normals=np.cross(a,Tlr[:3,3]);normals/=np.linalg.norm(normals,axis=1)[:,None]
 scores=[]
 for reverse in (False,True):
  for shift in range(4):
   ids=np.roll(np.arange(4)[::-1] if reverse else np.arange(4),shift)
   scores.append((float(np.sqrt(np.mean(np.sum(normals*b[ids],axis=1)**2))),ids.tolist()))
 order.append({'best':min(scores,key=lambda x:x[0]),'identity':scores[0]})
 sizes.append(np.linalg.norm(base[0][3]-np.roll(base[0][3],1,axis=0),axis=1).tolist())
summary={'captured_frames':len(packets),'matched_frames':len(order),'identity_best_count':sum(o['best'][1]==[0,1,2,3] for o in order),'tag3_sides_pixels':np.median(sizes,axis=0).tolist() if sizes else [],'methods':{}}
for name,rows in results.items():
 summary['methods'][name]={'n':len(rows),'height_cm':float(np.median([r['xyz'][2] for r in rows])*100),'relative_height_cm':float(np.median([r['delta'][2] for r in rows])*100),'sides_cm':(np.median([r['sides'] for r in rows],axis=0)*100).tolist(),'side_rms_error_mm':float(np.median([r['square_side_rms_mm'] for r in rows])),'nonplanarity_mm':float(np.median([r['nonplanarity_mm'] for r in rows]))}
(out/'tag-corner-diagnosis.json').write_text(json.dumps({'summary':summary,'results':results,'ordering':order},indent=2));print(json.dumps(summary,indent=2))
