import zmq,msgpack,time,numpy as np,json
ctx=zmq.Context();poll=zmq.Poller();socks={};stats={n:[] for n in ['camera','worker','measure','trajectory']};frames={};model=[];gaps=[]
for name,addr in [('camera', 'tcp://127.0.0.1:53097'), ('worker', 'tcp://127.0.0.1:49729'), ('measure', 'tcp://127.0.0.1:43543'), ('trajectory', 'tcp://127.0.0.1:60255')]:
 s=ctx.socket(zmq.SUB);s.setsockopt(zmq.SUBSCRIBE,b'');s.setsockopt(zmq.CONFLATE,1);s.connect(addr);socks[s]=name;poll.register(s,zmq.POLLIN)
start=time.monotonic();paired={k:[] for k in stats};last_cam=0
while time.monotonic()-start<6:
 for s,_ in poll.poll(100):
  now=time.monotonic();d=msgpack.unpackb(s.recv(),raw=False);n=socks[s];fid=d.get('frame_id',d.get('left_id'));stats[n].append(now)
  if n=='camera':frames[fid]=now;last_cam=fid
  elif fid in frames:paired[n].append((now-frames[fid])*1000)
  if n=='worker':model.append(d['model_ms'])
  if n=='measure' and d.get('source_received_monotonic'):gaps.append((now-d['source_received_monotonic'])*1000)
def q(v):return dict(zip(['p50','p95','max'],map(float,np.percentile(v,[50,95,100])))) if v else None
print(json.dumps({'scope':'Desktop observations only; excludes camera exposure and network transport','rates_hz':{n:round(len(v)/6,1) for n,v in stats.items()},'same_frame_after_camera_ms':{n:q(v) for n,v in paired.items()},'worker_compute_ms':q(model),'desktop_to_measure_ms':q(gaps)},indent=2))
for s in socks:s.close(0)
ctx.term()
