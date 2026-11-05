import sys,msgpack,cv2,numpy as np,json
sys.path[:0]=['outputs/g1-ball-session','/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/repo/g1']
from ball_center import refine_boxes
from ball_geometry import best_stereo_ball
results=[];old=[];fails=0
with open('outputs/g1-ball-session/frames-20261016-102335.msgpack','rb') as f:
 for r in msgpack.Unpacker(f,raw=False,max_buffer_size=10000000):
  if r.get('measurement_label')!='static_point_02_repeat' or r.get('xyz_world') is None: continue
  img=cv2.imdecode(np.frombuffer(r['source_jpg'],np.uint8),0);width=img.shape[1]//2
  boxes=[refine_boxes(img[:,s*width:(s+1)*width],r['boxes'][s])[0] for s in (0,1)]
  T=np.array(r['T_world_left']); candidates=[]
  for l in boxes[0]:
   for rr in boxes[1]:
    b=best_stereo_ball([l],[rr],r['calibration'])
    if b:
     xyz=(T@np.r_[b['xyz_left'],1])[:3]
     if np.all(xyz>[-1,-1,-.15]) and np.all(xyz<[1,3.4,1.5]):candidates.append((np.prod(b['confidence'])/(.5+b['reprojection_error_px']),xyz))
  if candidates: results.append(max(candidates,key=lambda c:c[0])[1]);old.append(r['xyz_world'])
  else:fails+=1
out={'accepted':len(results),'rejected':fails}
for name,data in [('box',old),('ellipse',results)]:
 a=np.array(data);out[name]={'median_cm':(np.median(a,axis=0)*100).tolist(),'std_mm':(a.std(0)*1000).tolist(),'y_error_cm':float(np.median(a[:,1])*100-53)}
print(json.dumps(out,indent=2));open('outputs/g1-ball-session/center-comparison.json','w').write(json.dumps(out,indent=2))
