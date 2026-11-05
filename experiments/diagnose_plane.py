import sys,json,glob,msgpack,cv2,numpy as np
from pathlib import Path
sys.path.insert(0,'/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/repo/g1')
from stereo_viser import Detector,detect_camera_tags,make_T
from ball_geometry import camera_ray,project_point
from scipy.spatial.transform import Rotation
out=Path('outputs/g1-ball-session');world=json.loads((out/'session-calibration.json').read_text())['world']
det=Detector(families=world['tag_family'],quad_decimate=1,refine_edges=1)
labels=['static_height_sweep_20','static_height_sweep_40','static_height_sweep_60','static_height_sweep_80','static_height_sweep_20_return']
counts={k:0 for k in labels};results={k:[] for k in labels}
def triang(l,r,cal):
 T=np.array(cal['T_left_right']);a=camera_ray(l,cal['left_intrinsics']);b=T[:3,:3]@camera_ray(r,cal['right_intrinsics']);o=T[:3,3]
 ds=np.linalg.lstsq(np.column_stack((a,-b)),o,rcond=None)[0]
 if min(ds)<=0:return None
 p=(a*ds[0]+o+b*ds[1])/2
 pr=T[:3,:3].T@(p-o)
 err=max(np.linalg.norm(project_point(p,cal['left_intrinsics'])-l),np.linalg.norm(project_point(pr,cal['right_intrinsics'])-r))
 return p if err<1 else None
for file in ['frames-20261016-103533.msgpack','frames-20261016-110554.msgpack']:
 with (out/file).open('rb') as f:
  for pkt in msgpack.Unpacker(f,raw=False,max_buffer_size=10000000):
   label=pkt.get('measurement_label')
   if label not in counts:continue
   counts[label]+=1
   if counts[label]%25 or len(results[label])>=16:continue
   cal=pkt['calibration'];im=cv2.imdecode(np.frombuffer(pkt['source_jpg'],np.uint8),0);w=im.shape[1]//2
   tags=[]
   for side,key in [(0,'left_intrinsics'),(1,'right_intrinsics')]:
    intr=cal[key]
    if np.any(intr['distortion']):raise ValueError('Nonzero distortion requires consistent rectification')
    tags.append({t.tag_id:t for t in detect_camera_tags(det,im[:,side*w:(side+1)*w],intr,world['tag_size_m'],1)})
   if 5 not in tags[0] or 5 not in tags[1] or pkt.get('T_world_left') is None:continue
   Tw=np.array(pkt['T_world_left']);pts=[];tagheights={}
   for tid in set(tags[0])&set(tags[1])&{0,1,2,4,5}:
    points=[]
    for l,r in zip(tags[0][tid].corners,tags[1][tid].corners):
     p=triang(l,r,cal)
     if p is not None:points.append((Tw@np.r_[p,1])[:3])
    if len(points)==4:
     pts+=points;tagheights[str(tid)]=np.mean(points,axis=0).tolist()
   if len(pts)<12:continue
   pts=np.array(pts);A=np.c_[pts[:,:2],np.ones(len(pts))];coef=np.linalg.lstsq(A,pts[:,2],rcond=None)[0]
   residual=pts[:,2]-A@coef
   # Refit after rejecting gross corner outliers; report resulting support.
   good=np.abs(residual-np.median(residual))<max(.004,3*1.4826*np.median(np.abs(residual-np.median(residual))))
   if good.sum()>=12:coef=np.linalg.lstsq(A[good],pts[good,2],rcond=None)[0]
   n=np.r_[-coef[:2],1];norm=np.linalg.norm(n)
   ball=pkt.get('ball_refined');xyz=pkt.get('xyz_world_refined')
   if ball is None or xyz is None:continue
   b=np.array(xyz);height=(np.dot(n,b)-coef[2])/norm
   t0=tags[0][5];t1=tags[1][5];Tlr=np.array(cal['T_left_right'])
   tl=np.array(world['T_world_tag'])@np.linalg.inv(make_T(t0.pose_R,t0.pose_t))
   tr=np.array(world['T_world_tag'])@np.linalg.inv(Tlr@make_T(t1.pose_R,t1.pose_t))
   bc=np.r_[ball['xyz_left'],1]
   entry={'frame':pkt['frame_id'],'xyz':xyz,'plane_abc':coef.tolist(),'plane_height':height,'plane_rmse_mm':float(np.sqrt(np.mean((pts[good,2]-A[good]@coef)**2))*1000),'tag_centers':tagheights,'n_corners':int(good.sum()),'left_tag_ball_z':float((tl@bc)[2]),'right_tag_ball_z':float((tr@bc)[2]),'tag_pose_angle_deg':float(np.degrees((Rotation.from_matrix(tl[:3,:3]).inv()*Rotation.from_matrix(tr[:3,:3])).magnitude())),'camera_xyz':ball['xyz_left'],'tag_y_extent':np.ptp(pts[:,1]).item()}
   results[label].append(entry)
summary={}
for label,rs in results.items():
 if not rs:summary[label]={'n':0};continue
 summary[label]={'n':len(rs)}
 for key in ['xyz','plane_abc','plane_height','plane_rmse_mm','left_tag_ball_z','right_tag_ball_z','tag_pose_angle_deg','tag_y_extent','camera_xyz']:
  summary[label][key]=np.median([r[key] for r in rs],axis=0).tolist()
 summary[label]['tag_centers']={str(tid):np.median([r['tag_centers'][str(tid)] for r in rs if str(tid) in r['tag_centers']],axis=0).tolist() for tid in (0,1,2,4,5) if any(str(tid) in r['tag_centers'] for r in rs)}
(out/'plane-diagnosis.json').write_text(json.dumps({'summary':summary,'frames':results},indent=2));print(json.dumps(summary,indent=2))
