import sys,json,cv2,numpy as np,time
from pathlib import Path
sys.path.insert(0,'/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/repo/g1');sys.path.insert(0,str(Path('outputs/g1-ball-session').resolve()))
from stereo_viser import Detector,detect_camera_tags,make_T,average_poses,select_ball_measurement
from ball_geometry import best_stereo_ball
from ball_center import refine_boxes
src=Path('outputs/ball-recorder/recordings/20261020-172639-275907');out=Path('outputs/ball-recorder/fallback-review');out.mkdir(exist_ok=True)
cal=json.loads((src/'calibration.json').read_text());world=json.loads(Path('outputs/g1-ball-session/session-calibration.json').read_text())['world'];det=Detector(families=world['tag_family'],quad_decimate=1,refine_edges=1)
frames=[json.loads(l) for l in (src/'frames.jsonl').read_text().splitlines()];ds={d['frame_id']:d for d in map(json.loads,(src/'detections.jsonl').read_text().splitlines())};rows=[];examples=[];video=None;last_example=-10
for index,fr in enumerate(frames):
 im=cv2.imread(str(src/fr['file']));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);w=im.shape[1]//2;d=ds.get(fr['left_id']);row={'frame_id':fr['left_id'],'time_s':fr['elapsed_s'],'status':'no_detector_message'};selected={};poses=[]
 if d:
  for side,key,T in [(0,'left_intrinsics',np.eye(4)),(1,'right_intrinsics',np.array(cal['T_left_right']))]:
   tags=detect_camera_tags(det,gray[:,side*w:(side+1)*w],cal[key],world['tag_size_m'],1)
   poses.extend(T@make_T(t.pose_R,t.pose_t) for t in tags if t.tag_id==5)
  row['status']='no_tag5' if not poses else 'lost'
  if poses:
   Tw=np.array(world['T_world_tag'])@np.linalg.inv(average_poses(poses))
   for mode in ['box','contour']:
    boxes=d['boxes'] if mode=='box' else [refine_boxes(gray[:,i*w:(i+1)*w],d['boxes'][i])[0] for i in (0,1)]
    opts=[]
    for l in boxes[0]:
     for r in boxes[1]:
      b=best_stereo_ball([l],[r],cal)
      if b is None:continue
      xyz=(Tw@np.r_[b['xyz_left'],1])[:3]
      if np.all(xyz>=world['play_volume_min_m']) and np.all(xyz<=world['play_volume_max_m']):opts.append((np.prod(b['confidence'])/(.5+b['reprojection_error_px']),b,xyz))
    if opts:
     _,b,xyz=max(opts,key=lambda x:x[0]);selected[mode]=(b,xyz);row[mode+'_xyz']=xyz.tolist()
   packet={'world_measurement':True,'ball':selected.get('box',(None,None))[0],'xyz_world':row.get('box_xyz'),'ball_refined':selected.get('contour',(None,None))[0],'xyz_world_refined':row.get('contour_xyz')}
   b,xyz,label=select_ball_measurement(packet,True)
   if b:
    row['status']='contour' if 'contour' in selected else 'fallback';row['auto_xyz']=list(xyz)
    for side,key in [(0,'left_center_px'),(1,'right_center_px')]:
     x,y=b[key];cv2.circle(im,(round(x)+side*w,round(y)),5,(0,255,0) if row['status']=='contour' else (0,165,255),1)
   if len(selected)==2:row['method_difference_mm']=float(np.linalg.norm(selected['box'][1]-selected['contour'][1])*1000)
  for side,boxes in enumerate(d['boxes']):
   for x1,y1,x2,y2,*_ in boxes:cv2.rectangle(im,(round(x1)+side*w,round(y1)),(round(x2)+side*w,round(y2)),(160,160,160),1)
 cv2.putText(im,f"{fr['elapsed_s']:.2f}s | {row['status'].upper()} | green=contour orange=YOLO fallback",(5,16),cv2.FONT_HERSHEY_SIMPLEX,.44,(0,255,255),1)
 if video is None:video=cv2.VideoWriter(str(out/'fallback-review.avi'),cv2.VideoWriter_fourcc(*'MJPG'),59.9108678,(im.shape[1],im.shape[0]))
 video.write(im);rows.append(row)
 if row['status']=='fallback' and fr['elapsed_s']-last_example>3 and len(examples)<6:examples.append(im.copy());last_example=fr['elapsed_s']
 if index%300==0:print('processed',index,flush=True)
video.release();counts={k:sum(r['status']==k for r in rows) for k in ['contour','fallback','lost','no_tag5','no_detector_message']};jumps=[]
for a,b in zip(rows,rows[1:]):
 if a.get('auto_xyz') is not None and b.get('auto_xyz') is not None and a['status']!=b['status']:jumps.append(float(np.linalg.norm(np.array(a['auto_xyz'])-b['auto_xyz'])*1000))
diffs=[r['method_difference_mm'] for r in rows if 'method_difference_mm' in r]
summary={'total_frames':len(rows),'counts':counts,'auto_valid':counts['contour']+counts['fallback'],'switch_count':len(jumps),'switch_displacement_median_mm':float(np.median(jumps)) if jumps else None,'switch_displacement_max_mm':max(jumps) if jumps else None,'same_frame_method_difference_median_mm':float(np.median(diffs)) if diffs else None,'notes':['Availability is not ground-truth accuracy; false detections may remain.','Switch displacements include real ball motion, not just method switching.','Replay uses current saved table calibration, tag5 same-frame pose and existing play-volume gates.']}
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'frames.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
if examples:cv2.imwrite(str(out/'fallback-examples.jpg'),np.vstack(examples))
print(json.dumps(summary,indent=2))
