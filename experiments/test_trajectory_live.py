import sys,time,json,subprocess,zmq,msgpack,socket
from pathlib import Path
root=Path('/home/jingxuan/Documents/Codex/2026-10-16/ni');ctx=zmq.Context();pub=ctx.socket(zmq.PUB);port=pub.bind_to_random_port('tcp://127.0.0.1')
with socket.socket() as s:s.bind(('127.0.0.1',0));op=s.getsockname()[1]
log=(root/'work/trajectory-live-test.log').open('w')
p=subprocess.Popen([sys.executable,str(root/'outputs/ball-trajectory/run.py'),'--no-viewer','--input',f'tcp://127.0.0.1:{port}','--publish',f'tcp://127.0.0.1:{op}','--output',str(root/'work/trajectory-live-test')],stdout=log,stderr=log)
try:
 time.sleep(.5)
 for i in range(30):
  t=i/60;pub.send(msgpack.packb({'frame_id':i,'capture_monotonic_ns':int((10+t)*1e9),'world_measurement':True,'ball':{},'xyz_world':[t,t,2+t-4.905*t*t]},use_bin_type=True));time.sleep(1/60)
 time.sleep(.4)
 folder=max((root/'work/trajectory-live-test').iterdir());latest=json.loads((folder/'latest.json').read_text());assert latest['status']=='stream_stale' and not latest['predictions']
 rows=[json.loads(x) for x in (folder/'trajectory.jsonl').read_text().splitlines()];assert any(r.get('status')=='tracking' and r.get('predictions') for r in rows)
 print('PASS: live ZMQ input, valid forecast, stale stream clears forecast, JSON output.')
finally:
 p.terminate();p.wait(timeout=5);log.close();pub.close(0);ctx.term()
