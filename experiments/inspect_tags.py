import sys,cv2,zmq,msgpack,numpy as np,json
sys.path.insert(0,'/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/repo/g1')
from stereo_viser import Detector,detect_camera_tags
c=zmq.Context();s=c.socket(zmq.SUB);s.setsockopt(zmq.SUBSCRIBE,b'');s.connect('tcp://192.168.123.164:5560')
if s.poll(2000):
 p=msgpack.unpackb(s.recv(),raw=False);im=cv2.imdecode(np.frombuffer(p['jpg'],np.uint8),0);w=im.shape[1]//2;det=Detector(families='tag36h11',quad_decimate=1,refine_edges=1)
 for side,key in [(0,'left_intrinsics'),(1,'right_intrinsics')]:
  tags=detect_camera_tags(det,im[:,side*w:(side+1)*w],p['calibration'][key],.0875,1)
  print(side,[(int(t.tag_id),np.round(t.center,1).tolist()) for t in tags])
s.close(0);c.term()
