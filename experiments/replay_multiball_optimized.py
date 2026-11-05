import sys,json,cv2,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path('outputs/ball-recorder/multiball-review').resolve()));sys.path.insert(0,str(Path('outputs/g1-ball-session').resolve()));sys.path.insert(0,'/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/repo/g1')
from tracker import MultiBallTracker
from ball_center import refine_center
from ball_geometry import best_stereo_ball
src=Path('outputs/ball-recorder/recordings/20261020-172639-275907');out=Path('outputs/ball-recorder/multiball-review');data=json.loads(Path('outputs/ball-recorder/ablation/candidates.json').read_text());ds={d['frame_id']:d for d in map(json.loads,(src/'detections.jsonl').read_text().splitlines())};cal=json.loads((src/'calibration.json').read_text());tracker=MultiBallTracker(**json.loads((out/'best-parameters-wide.json').read_text())['params']);v=cv2.VideoWriter(str(out/'optimized-multiball.avi'),cv2.VideoWriter_fourcc(*'MJPG'),59.9108678,(960,270));rows=[];shots=[];counts={'stereo_candidates':0,'assigned_measurements':0,'confirmed_measurements':0,'contour_measurements':0,'fallback_measurements':0,'frames_with_multiple_confirmed':0}
for idx,f in enumerate(data):
 im=cv2.imread(str(src/f['file']));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);cs=[];d=ds.get(f['frame_id'])
 for raw in f['cs']:
  c={k:raw[k] for k in ['q','pix','confidence']};c['box_pix']=c['pix'].copy();c['method']='box_fallback'
  if d:
   boxes=[min(d['boxes'][i],key=lambda b:np.linalg.norm(np.array([(b[0]+b[2])/2,(b[1]+b[3])/2])-c['box_pix'][i*2:i*2+2])) for i in (0,1)]
   fits=[refine_center(gray[:,i*480:(i+1)*480],boxes[i]) for i in (0,1)]
   if all(fits):
    adj=[]
    for b,fit in zip(boxes,fits):
     x,y=fit['center_px'];w=b[2]-b[0];h=b[3]-b[1];adj.append([x-w/2,y-h/2,x+w/2,y+h/2,*b[4:]])
    ball=best_stereo_ball([adj[0]],[adj[1]],cal)
    if ball and abs(ball['left_center_px'][1]-ball['right_center_px'][1])<=1.5:c.update(q=ball['xyz_left'],pix=ball['left_center_px']+ball['right_center_px'],method='contour')
  cs.append(c)
 result=tracker.update(cs,f['t']);confirmed=[r for r in result if r['confirmed']];counts['stereo_candidates']+=len(cs);counts['assigned_measurements']+=len(result);counts['confirmed_measurements']+=len(confirmed);counts['frames_with_multiple_confirmed']+=len(confirmed)>=2
 # Assertions explicitly check one-to-one assignment in both cameras.
 for side in (0,1):assert len({tuple(r['box_pix'][side*2:side*2+2]) for r in result})==len(result)
 for r in result:
  if r['confirmed']:counts['contour_measurements' if r['method']=='contour' else 'fallback_measurements']+=1
  color=(0,255,0) if r['confirmed'] else (150,150,150)
  for side in (0,1):
   x,y=r['pix'][side*2:side*2+2];pt=(round(x)+side*480,round(y));cv2.circle(im,pt,7,color,1);cv2.putText(im,f"{r['id']} {'C' if r['method']=='contour' else 'B'}",(pt[0]+7,pt[1]),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
 cv2.putText(im,f"{f['t']:.2f}s | confirmed {len(confirmed)} | C contour / B box fallback | gray tentative",(4,15),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,255,255),1);v.write(im);rows.append({'frame_id':f['frame_id'],'time_s':f['t'],'tracks':result})
 if idx in [180,600,900,1200,1500]:shots.append(im)
v.release();cv2.imwrite(str(out/'optimized-examples.jpg'),np.vstack(shots));counts.update(frames=len(data),created_ids=tracker.next_id-1,notes=['Counts describe algorithm output, not accuracy. No identity ground truth.','Motion association uses box centers consistently across contour/fallback switches.','Missing frames produce no measured point; internal prediction only supports association.','Camera coordinates only; no tag5 available.'])
(out/'optimized-summary.json').write_text(json.dumps(counts,indent=2));(out/'optimized-tracks.jsonl').write_text('\n'.join(json.dumps(r) for r in rows));print(json.dumps(counts,indent=2))
