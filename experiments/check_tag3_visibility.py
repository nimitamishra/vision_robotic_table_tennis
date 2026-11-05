import sys,time,json,cv2,zmq,msgpack,numpy as np
from pathlib import Path
sys.path.insert(0,'/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/repo/g1')
from stereo_viser import Detector,detect_camera_tags
c=zmq.Context();s=c.socket(zmq.SUB);s.setsockopt(zmq.SUBSCRIBE,b'');s.setsockopt(zmq.CONFLATE,1);s.connect('tcp://192.168.123.164:5560');det=Detector(families='tag36h11',quad_decimate=1,refine_edges=1)
rows=[];start=time.monotonic();out=Path('outputs/g1-ball-session')
while len(rows)<30 and time.monotonic()-start<10:
 if not s.poll(500):continue
 p=msgpack.unpackb(s.recv(),raw=False);im=cv2.imdecode(np.frombuffer(p['jpg'],np.uint8),0);w=im.shape[1]//2;row=[]
 for side,key in [(0,'left_intrinsics'),(1,'right_intrinsics')]:
  tags=detect_camera_tags(det,im[:,side*w:(side+1)*w],p['calibration'][key],.0875,1)
  row.append([{'id':int(t.tag_id),'center':t.center.tolist(),'sides_px':np.linalg.norm(t.corners-np.roll(t.corners,1,axis=0),axis=1).tolist()} for t in tags])
 rows.append(row)
 if len(rows)==1:(out/'tag3-placement-live.jpg').write_bytes(p['jpg'])
s.close(0);c.term()
summary={'n':len(rows),'counts':[{str(t):sum(any(x['id']==t for x in r[side]) for r in rows) for t in range(6)} for side in range(2)],'first':rows[0] if rows else None}
(out/'tag3-visibility.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
