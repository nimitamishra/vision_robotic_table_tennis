import sys,json,cv2,numpy as np
from pathlib import Path
sys.path.insert(0,'/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/repo/g1');sys.path.insert(0,str(Path('outputs/g1-ball-session').resolve()))
from ball_geometry import best_stereo_ball
from ball_center import refine_center
src=Path('outputs/ball-recorder/recordings/20261020-172639-275907');out=Path('outputs/ball-recorder/stereo-tracking-review');out.mkdir(exist_ok=True)
cal=json.loads((src/'calibration.json').read_text());frames=list(map(json.loads,(src/'frames.jsonl').read_text().splitlines()));ds={d['frame_id']:d for d in map(json.loads,(src/'detections.jsonl').read_text().splitlines())}
video=cv2.VideoWriter(str(out/'tracking-review.avi'),cv2.VideoWriter_fourcc(*'MJPG'),59.9108678,(960,270));rows=[];previous=[];track=None;velocity=np.zeros(3);last_t=None;last_pixel=None;hypotheses=[];shots=[]
for idx,fr in enumerate(frames):
 t=fr['elapsed_s'];im=cv2.imread(str(src/fr['file']));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);d=ds.get(fr['left_id']);candidates=[];rawpairs=0
 if d:
  for l in d['boxes'][0]:
   for r in d['boxes'][1]:
    rawpairs+=1;lc=np.array([(l[0]+l[2])/2,(l[1]+l[3])/2]);rc=np.array([(r[0]+r[2])/2,(r[1]+r[3])/2]);ls=max(l[2]-l[0],l[3]-l[1]);rs=max(r[2]-r[0],r[3]-r[1])
    if abs(lc[1]-rc[1])>1.5 or lc[0]-rc[0]<=.5 or not .6<ls/rs<1/.6:continue
    b=best_stereo_ball([l],[r],cal)
    if b is None:continue
    method='box';fits=[refine_center(gray[:,:480],l),refine_center(gray[:,480:],r)]
    if all(fits):
     adjusted=[]
     for box,fit in zip([l,r],fits):
      x,y=fit['center_px'];w=box[2]-box[0];h=box[3]-box[1];adjusted.append([x-w/2,y-h/2,x+w/2,y+h/2,*box[4:]])
     refined=best_stereo_ball([adjusted[0]],[adjusted[1]],cal)
     if refined and abs(refined['left_center_px'][1]-refined['right_center_px'][1])<=1.5:b=refined;method='contour'
    candidates.append({'q':np.array(b['xyz_left']),'pix':np.array(b['left_center_px']+b['right_center_px']),'confidence':float(l[4]*r[4]),'method':method})
 selected=None;state='unlocked';residual=None
 if track is not None and t-last_t>.15:track=None;velocity=np.zeros(3)
 if track is not None:
  dt=t-last_t;pred=track+velocity*dt;valid=[]
  for c in candidates:
   e=np.linalg.norm(c['q']-pred);speed=np.linalg.norm(c['q']-track)/max(dt,.001)
   # Permit bounce reversals; never output a predicted point as a measurement.
   if speed<=12 and e<=.06+4*dt:valid.append((e-.01*c['confidence'],c,e))
  if valid:
   _,selected,residual=min(valid,key=lambda x:x[0]);v=(selected['q']-track)/max(dt,.001);velocity=.5*velocity+.5*v;state='tracked'
  else:state='temporarily_lost'
 if track is None:
  # Require 3 observations of a moving candidate to initialize, not confidence alone.
  new=[]
  for c in candidates:
   matches=[h for h in hypotheses if 0<t-h['t']<.08 and np.linalg.norm(c['pix']-h['c']['pix'])<30 and np.linalg.norm(c['q']-h['c']['q'])<12*(t-h['t'])+.03]
   if matches:
    h=min(matches,key=lambda h:np.linalg.norm(c['pix']-h['c']['pix']));h={'c':c,'t':t,'n':h['n']+1,'start':h['start'],'first':h['first']}
   else:h={'c':c,'t':t,'n':1,'start':t,'first':c['pix']}
   new.append(h)
  hypotheses=new
  ready=[h for h in hypotheses if h['n']>=3 and np.linalg.norm(h['c']['pix']-h['first'])>=3]
  if ready:selected=max(ready,key=lambda h:h['c']['confidence'])['c'];state='acquired';velocity=np.zeros(3)
 if selected is not None:track=selected['q'];last_t=t;last_pixel=selected['pix']
 row={'frame_id':fr['left_id'],'time_s':t,'raw_pairs':rawpairs,'geometric_candidates':len(candidates),'state':state,'xyz_camera_m':selected['q'].tolist() if selected else None,'method':selected['method'] if selected else None,'prediction_residual_m':residual};rows.append(row)
 for c in candidates:
  for side in [0,1]:x,y=c['pix'][side*2:side*2+2];cv2.circle(im,(round(x)+480*side,round(y)),4,(130,130,130),1)
 if selected:
  for side in [0,1]:x,y=selected['pix'][side*2:side*2+2];cv2.circle(im,(round(x)+480*side,round(y)),8,(0,255,0),2)
 cv2.putText(im,f"{t:.2f}s {state} | pairs {len(candidates)} | "+(selected['method'] if selected else 'NO MEASUREMENT'),(4,16),cv2.FONT_HERSHEY_SIMPLEX,.47,(0,255,255),1);video.write(im)
 if idx in [300,600,900,1200,1500]:shots.append(im)
video.release();cv2.imwrite(str(out/'examples.jpg'),np.vstack(shots));(out/'frames.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n');summary={'frames':len(rows),'frames_with_geometric_candidates':sum(r['geometric_candidates']>0 for r in rows),'tracked_measurements':sum(r['xyz_camera_m'] is not None for r in rows),'states':{s:sum(r['state']==s for r in rows) for s in set(r['state'] for r in rows)},'limitations':['Experimental thresholds, no manually labeled ground truth.','Requires motion to initialize; stationary ball intentionally not acquired.','No tag5/world frame available. No table-volume gate.','Association can still switch objects; inspection required before deployment.']};(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
