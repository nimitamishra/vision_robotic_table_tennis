"""Receive one robot stereo feed; distribute latest packets locally, timestamp receipt."""
import argparse,signal,time
import msgpack,zmq
p=argparse.ArgumentParser();p.add_argument('--camera',required=True);p.add_argument('--publish',required=True);a=p.parse_args()
ctx=zmq.Context();source=ctx.socket(zmq.SUB);source.setsockopt(zmq.SUBSCRIBE,b'');source.setsockopt(zmq.CONFLATE,1);source.connect(a.camera)
pub=ctx.socket(zmq.PUB);pub.setsockopt(zmq.SNDHWM,1);pub.bind(a.publish);stop=False
def end(*_):
 global stop
 stop=True
signal.signal(signal.SIGTERM,end);signal.signal(signal.SIGINT,end)
count=0;start=time.monotonic()
try:
 while not stop:
  if not source.poll(100):continue
  packet=msgpack.unpackb(source.recv(),raw=False);packet['desktop_received_monotonic']=time.monotonic()
  pub.send(msgpack.packb(packet,use_bin_type=True));count+=1
  if time.monotonic()-start>5:print(f'Relay: {count/(time.monotonic()-start):.1f} FPS',flush=True);count=0;start=time.monotonic()
finally:source.close(0);pub.close(0);ctx.term()
