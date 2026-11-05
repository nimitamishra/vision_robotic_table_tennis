"""Offline checks: no robot connection and no motor control processes."""
import os,sys,time,json
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu')
from app import *
app=W.QApplication([]);window=Main();window.show()
assert not window.procs and window.stream is None
window.receive('camera',{'left_id':123,'jpg':b'bad image'})
window.receive('detection',{'frame_id':123,'boxes':[[],[]],'model_ms':2.,'inference_fps':30.})
assert '123' in window.metrics.text()
# Verify actual PTY transport used by SONIC, with a harmless echo process.
p=Process('test-pty',[sys.executable,'-u','-c','import tty,os;tty.setraw(0);print("READY",flush=True);print("RECEIVED="+os.read(0,1).decode(),flush=True)'],terminal=True)
for _ in range(100):
 p.pump()
 if b'READY' in p.path.read_bytes():break
 time.sleep(.02)
p.key('x')
for _ in range(100):
 p.pump()
 if not p.alive():break
 time.sleep(.02)
p.pump();assert b'RECEIVED=x' in p.path.read_bytes();p.close()
# Check subscription and msgpack path on local-only synthetic data.
ctx=zmq.Context();pub=ctx.socket(zmq.PUB);port=pub.bind_to_random_port('tcp://127.0.0.1')
stream=Streams('127.0.0.1',port);received=[];stream.packet.connect(lambda n,d:received.append((n,d)));stream.start()
for _ in range(100):
 pub.send(msgpack.packb({'frame_id':456,'boxes':[[],[]]},use_bin_type=True));app.processEvents();time.sleep(.02)
 if received:break
stream.requestInterruption();assert stream.wait(2000);pub.close(0);ctx.term()
assert any(n=='detection' and d['frame_id']==456 for n,d in received)
window.seen.clear();window.metrics.setText('Waiting for camera and detections');window.camera.setText('Waiting for data');window.tick()
for _ in range(10):app.processEvents();time.sleep(.05)
window.grab().save(str(ROOT/'preview.png'))
window.close();print('PASS: idle startup, UI packet updates, PTY keys, ZMQ subscription, clean shutdown.')
