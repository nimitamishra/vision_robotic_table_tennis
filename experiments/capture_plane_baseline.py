import zmq,msgpack,time,json
from pathlib import Path
out=Path('outputs/g1-ball-session');c=zmq.Context();s=c.socket(zmq.SUB);s.setsockopt(zmq.SUBSCRIBE,b'');s.setsockopt(zmq.CONFLATE,1);s.connect('tcp://192.168.123.164:5560')
start=time.monotonic();n=0
with (out/'plane-baseline-before-placement.msgpack').open('wb') as f:
 while n<30 and time.monotonic()-start<10:
  if not s.poll(500):continue
  data=s.recv();p=msgpack.unpackb(data,raw=False)
  if p['left_id']!=p['right_id']:continue
  f.write(msgpack.packb(p,use_bin_type=True));n+=1;time.sleep(.1)
s.close(0);c.term()
print(json.dumps({'synchronized_pairs':n,'purpose':'Baseline before moving raised tag3 onto tabletop; no calibration changes'}))
