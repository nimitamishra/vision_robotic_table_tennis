import sys,json,cv2,numpy as np
from pathlib import Path
sys.path.insert(0,'/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/repo/g1');sys.path.insert(0,str(Path('outputs/g1-ball-session').resolve()))
from ball_geometry import best_stereo_ball
src=Path('outputs/ball-recorder/recordings/20261020-172639-275907');out=Path('outputs/ball-recorder/ablation');out.mkdir(exist_ok=True);cal=json.loads((src/'calibration.json').read_text());fs=list(map(json.loads,(src/'frames.jsonl').read_text().splitlines()));ds={d['frame_id']:d for d in map(json.loads,(src/'detections.jsonl').read_text().splitlines())};data=[];prev=None;train=[];tiles=[]
for idx,f in enumerate(fs):
 im=cv2.imread(str(src/f['file']),0);d=ds.get(f['left_id']);cs=[]
 if d:
  for l in d['boxes'][0]:
   for r in d['boxes'][1]:
    lc=np.array([(l[0]+l[2])/2,(l[1]+l[3])/2]);rc=np.array([(r[0]+r[2])/2,(r[1]+r[3])/2]);lw,lh=l[2]-l[0],l[3]-l[1];rw,rh=r[2]-r[0],r[3]-r[1]
    if abs(lc[1]-rc[1])>1.5 or lc[0]-rc[0]<=.5 or not .6<max(lw,lh)/max(rw,rh)<1/.6:continue
    b=best_stereo_ball([l],[r],cal)
    if not b:continue
    x,y=np.round(lc).astype(int);rad=max(5,int(max(lw,lh)*.7));x0,x1=max(0,x-rad),min(480,x+rad+1);y0,y1=max(0,y-rad),min(270,y+rad+1);crop=im[y0:y1,x0:x1];patch=cv2.resize(crop,(16,16));feat=(patch.astype(float)-patch.mean())/max(10,patch.std());motion=float(np.mean(cv2.absdiff(crop,prev[y0:y1,x0:x1]))) if prev is not None else 0
    shape=min(lw,lh)/max(lw,lh)*min(rw,rh)/max(rw,rh);c={'q':b['xyz_left'],'pix':list(lc)+list(rc),'confidence':float(l[4]*r[4]),'motion':motion,'shape':float(shape),'feat':feat.ravel().tolist()};cs.append(c)
    if idx%120==60 and f['elapsed_s']<15:
     tid=len(train);train.append({'id':tid,'frame':idx,'time':f['elapsed_s'],'candidate':len(cs)-1});tile=cv2.cvtColor(cv2.resize(crop,(90,90),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR);tile=cv2.copyMakeBorder(tile,22,0,0,0,cv2.BORDER_CONSTANT);cv2.putText(tile,f'{tid} {f["elapsed_s"]:.0f}s',(2,15),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1);tiles.append(tile)
 data.append({'frame_id':f['left_id'],'t':f['elapsed_s'],'file':f['file'],'cs':cs});prev=im
(out/'candidates.json').write_text(json.dumps(data));(out/'training-index.json').write_text(json.dumps(train));cols=8
while len(tiles)%cols:tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(out/'training-crops.jpg'),np.vstack([np.hstack(tiles[i:i+cols]) for i in range(0,len(tiles),cols)]));print('training crops',len(train))
