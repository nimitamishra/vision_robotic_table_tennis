import sys,json,cv2,msgpack,numpy as np
from pathlib import Path
sys.path.insert(0,'/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/repo/g1')
from stereo_viser import Detector,detect_camera_tags,make_T,average_poses
from ball_geometry import camera_ray,project_point,best_stereo_ball
sys.path.insert(0,str(Path('outputs/g1-ball-session').resolve()))
from ball_center import refine_center
out=Path('outputs/g1-ball-session');world=json.loads((out/'session-calibration.json').read_text())['world'];det=Detector(families='tag36h11',quad_decimate=1,refine_edges=1)
results={}
for label,file in [('before','ball-plane-before-robot-move-source.msgpack'),('after','ball-after-move-source.msgpack')]:
 rows=[]
 with (out/file).open('rb') as f:
  for p in msgpack.Unpacker(f,raw=False):
   cal=p['calibration'];im=cv2.imdecode(np.frombuffer(p['source_jpg'],np.uint8),0);w=im.shape[1]//2;T=np.array(cal['T_left_right'])
   tags=[{t.tag_id:t for t in detect_camera_tags(det,im[:,i*w:(i+1)*w],cal[key],world['tag_size_m'],1)} for i,key in [(0,'left_intrinsics'),(1,'right_intrinsics')]]
   if not all(t in tags[0] and t in tags[1] for t in [0,5]):continue
   pose=average_poses([make_T(tags[0][5].pose_R,tags[0][5].pose_t),T@make_T(tags[1][5].pose_R,tags[1][5].pose_t)])
   Tw=np.array(world['T_world_tag'])@np.linalg.inv(pose);pts={}
   for tid in [0,5]:
    qs=[]
    for l,r in zip(tags[0][tid].corners,tags[1][tid].corners):
     a=camera_ray(l,cal['left_intrinsics']);b=T[:3,:3]@camera_ray(r,cal['right_intrinsics']);o=T[:3,3];ds=np.linalg.lstsq(np.column_stack((a,-b)),o,rcond=None)[0];q=(a*ds[0]+o+b*ds[1])/2
     qs.append(q)
    pts[tid]=np.array(qs)
   cs={t:pts[t].mean(0) for t in [0,5]};opts=[]
   for lb in p['boxes'][0]:
    for rb in p['boxes'][1]:
     b=best_stereo_ball([lb],[rb],cal)
     if b is None:continue
     xyz=(Tw@np.r_[b['xyz_left'],1])[:3]
     if np.all(xyz>=world['play_volume_min_m']) and np.all(xyz<=world['play_volume_max_m']):opts.append((np.prod(b['confidence'])/(.5+b['reprojection_error_px']),b,lb,rb))
   if not opts:continue
   _,ball,lb,rb=max(opts,key=lambda v:v[0]);q=np.array(ball['xyz_left']);traces=[];fits=[]
   for side,box in enumerate([lb,rb]):
    tr={}
    def trace(frame,event,arg):
     if frame.f_code.co_name=='refine_center' and event=='return':tr.update(return_line=frame.f_lineno,n_fits=len(frame.f_locals.get('fits',[])),spread=frame.f_locals.get('spread'))
     return trace
    sys.settrace(trace);fit=refine_center(im[:,side*w:(side+1)*w],box);sys.settrace(None);traces.append(tr);fits.append(fit)
   row={'frame':p['frame_id'],'tag0_world_cm':((Tw@np.r_[cs[0],1])[:3]*100).tolist(),'tag5_world_cm':((Tw@np.r_[cs[5],1])[:3]*100).tolist(),'tag0_minus5_cm':(Tw[:3,:3]@(cs[0]-cs[5])*100).tolist(),'tag_distance_cm':float(np.linalg.norm(cs[0]-cs[5])*100),'ball_world_cm':((Tw@np.r_[q,1])[:3]*100).tolist(),'ball_to_tag_cm':[float(np.linalg.norm(q-cs[t])*100) for t in [0,5]],'refine_trace':traces,'refine':fits,'centers':[ball['left_center_px'],ball['right_center_px']],'points':np.concatenate([pts[0],pts[5]]).tolist(),'ball_camera':q.tolist()}
   if all(fits):row['refine_offsets_px']=[(np.array(fits[i]['center_px'])-row['centers'][i]).tolist() for i in [0,1]]
   rows.append(row)
 summary={'n':len(rows)}
 for key in ['tag0_world_cm','tag5_world_cm','tag0_minus5_cm','tag_distance_cm','ball_world_cm','ball_to_tag_cm']:
  vs=[r[key] for r in rows];summary[key]={'median':np.median(vs,axis=0).tolist(),'std':np.std(vs,axis=0).tolist()}
 summary['refine_valid']=[sum(r['refine'][i] is not None for r in rows) for i in [0,1]]
 summary['refine_failure_examples']=rows[0]['refine_trace'];results[label]={'summary':summary,'rows':rows}
# Fit a rigid camera-frame alignment using only the same two fixed tags.
A=np.median([r['points'] for r in results['before']['rows']],axis=0);B=np.median([r['points'] for r in results['after']['rows']],axis=0)
ac=A.mean(0);bc=B.mean(0);U,S,Vt=np.linalg.svd((B-bc).T@(A-ac));R=Vt.T@U.T
if np.linalg.det(R)<0:Vt[-1]*=-1;R=Vt.T@U.T
t=ac-R@bc;qa=np.median([r['ball_camera'] for r in results['before']['rows']],axis=0);qb=np.median([r['ball_camera'] for r in results['after']['rows']],axis=0)
results['alignment']={'tag_corner_rms_mm':float(np.sqrt(np.mean(np.sum(((R@B.T).T+t-A)**2,axis=1)))*1000),'ball_position_difference_mm':float(np.linalg.norm(R@qb+t-qa)*1000),'caution':'Only two nearby tags; not independent ground truth and sensitive to corner depth errors.'}
(out/'offline-robot-move-audit.json').write_text(json.dumps(results,indent=2));print(json.dumps({k:v['summary'] for k,v in results.items() if k!='alignment'},indent=2));print(results['alignment'])
