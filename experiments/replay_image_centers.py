import sys,json,cv2,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path('outputs/g1-ball-session').resolve()))
from ball_center import refine_center
src=Path('outputs/ball-recorder/recordings/20261020-172639-275907');out=Path('outputs/ball-recorder/fallback-review');ds={d['frame_id']:d for d in map(json.loads,(src/'detections.jsonl').read_text().splitlines())};frames=list(map(json.loads,(src/'frames.jsonl').read_text().splitlines()));v=cv2.VideoWriter(str(out/'image-center-review.avi'),cv2.VideoWriter_fourcc(*'MJPG'),59.91,(960,270));counts={'boxes':0,'contour_pass':0,'contour_fail':0,'frames_both_yolo':0,'frames_both_contour':0};shots=[]
for n,f in enumerate(frames):
 im=cv2.imread(str(src/f['file']));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);d=ds.get(f['left_id']);passed=[0,0]
 if d:
  counts['frames_both_yolo']+=int(all(d['boxes']))
  for side,boxes in enumerate(d['boxes']):
   for box in boxes:
    fit=refine_center(gray[:,side*480:(side+1)*480],box);counts['boxes']+=1;counts['contour_pass' if fit else 'contour_fail']+=1;passed[side]+=fit is not None
    x1,y1,x2,y2=box[:4];color=(0,255,0) if fit else (0,165,255);x,y=fit['center_px'] if fit else [(x1+x2)/2,(y1+y2)/2]
    cv2.rectangle(im,(round(x1)+side*480,round(y1)),(round(x2)+side*480,round(y2)),color,1);cv2.circle(im,(round(x)+side*480,round(y)),2,color,1)
  counts['frames_both_contour']+=int(all(passed))
 cv2.putText(im,f"{f['elapsed_s']:.2f}s green=contour orange=box fallback; ALL candidates, not validated 3D",(3,14),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1);v.write(im)
 if n in [300,600,900,1200,1500]:shots.append(im)
v.release();cv2.imwrite(str(out/'image-examples.jpg'),np.vstack(shots));(out/'image-summary.json').write_text(json.dumps(counts,indent=2));print(counts)
