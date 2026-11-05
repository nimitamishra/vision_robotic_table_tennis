import cv2,json,numpy as np
from pathlib import Path
s=Path('outputs/ball-recorder/recordings/20261020-172639-275907');o=Path('outputs/ball-recorder/multiball-review');data=json.loads(Path('outputs/ball-recorder/ablation/candidates.json').read_text());prev=None;refs=[];gap=6
for k,f in enumerate(data):
 im=cv2.imread(str(s/f['file']),0)
 if k>=gap and k%3==0 and data[k-gap]['cs'] and f['cs']:
  prev=cv2.imread(str(s/data[k-gap]['file']),0)
  cs=data[k-gap]['cs'];now=f['cs'];n=len(cs);points=np.array([c['pix'] for c in cs],np.float32).reshape(n,2,2);points[:,1,0]+=480;p=points.reshape(-1,1,2)
  q,st,_=cv2.calcOpticalFlowPyrLK(prev,im,p,None,winSize=(15,15),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
  back,st2,_=cv2.calcOpticalFlowPyrLK(im,prev,q,None,winSize=(15,15),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
  fb=np.linalg.norm(back-p,axis=2).reshape(n,2);q=q.reshape(n,2,2);q[:,1,0]-=480
  target=np.array([c['pix'] for c in now]).reshape(-1,2,2)
  for i in range(n):
   if not all(st.reshape(n,2)[i]) or not all(st2.reshape(n,2)[i]) or max(fb[i])>1:continue
   distance=np.linalg.norm(target-q[i],axis=2).max(axis=1);order=np.argsort(distance);j=int(order[0])
   if distance[j]>3 or (len(order)>1 and distance[order[1]]-distance[j]<2):continue
   # Forward/back flow validates local image correspondence, not semantic ball identity.
   refs.append({'a':k-gap,'b':k,'pix_a':cs[i]['pix'],'pix_b':now[j]['pix'],'flow_error_px':float(distance[j])})
 prev=im
(o/'flow-gap-correspondences.json').write_text(json.dumps(refs));print('independent local motion correspondences',len(refs))
