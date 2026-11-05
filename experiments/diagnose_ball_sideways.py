import sys,json,cv2,msgpack,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path('outputs/g1-ball-session').resolve()))
from ball_center import refine_center
out=Path('outputs/g1-ball-session');report={};tiles=[]
for label,file in [('far','ball-plane-far-source.msgpack'),('sideways','ball-plane-source.msgpack')]:
 rows=[]
 with (out/file).open('rb') as f:
  for p in msgpack.Unpacker(f,raw=False):
   im=cv2.imdecode(np.frombuffer(p['source_jpg'],np.uint8),0);centers=[];row=[]
   for side in (0,1):
    boxes=[b for b in p['boxes'][side] if 200<(b[0]+b[2])/2<350 and 55<(b[1]+b[3])/2<105]
    if not boxes:break
    b=max(boxes,key=lambda b:b[4]);center=np.array([(b[0]+b[2])/2,(b[1]+b[3])/2]);centers.append(center);trace={}
    def tracer(frame,event,arg):
     if frame.f_code.co_name=='refine_center' and event=='return':
      trace.update(line=frame.f_lineno,bg=frame.f_locals.get('bg'),bright=frame.f_locals.get('bright'),n_threshold_fits=len(frame.f_locals.get('fits',[])),spread=frame.f_locals.get('spread'))
     return tracer
    sys.settrace(tracer);fit=refine_center(im[:,side*480:(side+1)*480],b);sys.settrace(None)
    row.append({'box_center':center.tolist(),'refined':fit,'trace':trace})
    if not rows:
     x,y=np.round(center).astype(int);roi=im[y-17:y+18,side*480+x-22:side*480+x+23];tile=cv2.cvtColor(cv2.resize(roi,None,fx=8,fy=8,interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR);cv2.putText(tile,label+(' L' if side==0 else ' R'),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),1);tiles.append(tile)
   if len(row)==2:rows.append(row)
 ds=[np.array(r[0]['box_center'])-r[1]['box_center'] for r in rows]
 report[label]={'n':len(rows),'median_dx_dy_px':np.median(ds,axis=0).tolist(),'sides':[{'refined_valid':sum(r[i]['refined'] is not None for r in rows),'return_lines':{str(k):sum(r[i]['trace']['line']==k for r in rows) for k in set(r[i]['trace']['line'] for r in rows)},'example':rows[0][i]} for i in (0,1)]}
cv2.imwrite(str(out/'ball-center-sideways-crops.jpg'),np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])]))
(out/'ball-center-sideways-diagnosis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
