import zmq,msgpack,json,subprocess,time,sys
from pathlib import Path
root=Path('/home/jingxuan/Documents/Codex/2026-10-16/ni')
src=root/'outputs/ball-recorder/recordings/20261020-172639-275907'
row=json.loads((src/'frames.jsonl').read_text().splitlines()[0]);jpg=(src/row['file']).read_bytes();cal=json.loads((src/'calibration.json').read_text())
ctx=zmq.Context();a=ctx.socket(zmq.PUB);b=ctx.socket(zmq.PUB);pa=a.bind_to_random_port('tcp://127.0.0.1');pb=b.bind_to_random_port('tcp://127.0.0.1')
p=subprocess.Popen([sys.executable,str(root/'outputs/G1-Studio/record.py'),'--seconds','1','--countdown','0','--camera',f'tcp://127.0.0.1:{pa}','--detector',f'tcp://127.0.0.1:{pb}','--output',str(root/'work/studio-record-test')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
for n in range(200):
 a.send(msgpack.packb({'left_id':n,'right_id':n,'jpg':jpg,'calibration':cal},use_bin_type=True));b.send(msgpack.packb({'frame_id':n,'boxes':[[],[]]},use_bin_type=True));time.sleep(.02)
 if p.poll() is not None:break
out=p.communicate(timeout=30)[0].decode();print(out);assert p.returncode==0
folder=max((root/'work/studio-record-test').iterdir());s=json.loads((folder/'summary.json').read_text());assert s['camera']>0 and s['detector']>0 and not s['errors'];assert (folder/'review.mp4').stat().st_size>0
subprocess.run([str(root/'work/ffmpeg'),'-v','error','-i',str(folder/'review.mp4'),'-f','null','-'],check=True)
a.close(0);b.close(0);ctx.term();print('PASS: stereo recording, detector messages, finalized files, decodable MP4.')
