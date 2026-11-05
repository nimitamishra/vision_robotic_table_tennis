"""Record stereo IR + detector messages without changing the running pipeline."""
import argparse,json,time,signal,threading,queue,datetime
from pathlib import Path
import cv2,msgpack,numpy as np,zmq


def main():
 p=argparse.ArgumentParser(description='Stereo ball recorder: Ctrl+C stops and finalizes recording.')
 p.add_argument('--seconds',type=float,default=30);p.add_argument('--countdown',type=int,default=5)
 p.add_argument('--camera',default='tcp://192.168.123.164:5560');p.add_argument('--detector',default='tcp://127.0.0.1:5561')
 p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'recordings');a=p.parse_args()
 if a.seconds<=0 or a.countdown<0:p.error('seconds must be positive; countdown must be nonnegative')
 stop=threading.Event();signal.signal(signal.SIGINT,lambda *_:stop.set())
 for n in range(a.countdown,0,-1):
  print(f'{n} 秒后开始，请准备弹球…',flush=True)
  if stop.wait(1):return
 folder=a.output/datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f');(folder/'frames').mkdir(parents=True)
 ctx=zmq.Context();poll=zmq.Poller();sockets={}
 for name,address in [('camera',a.camera),('detector',a.detector)]:
  s=ctx.socket(zmq.SUB);s.setsockopt(zmq.SUBSCRIBE,b'');s.setsockopt(zmq.RCVHWM,1000);s.connect(address);poll.register(s,zmq.POLLIN);sockets[s]=name
 work=queue.Queue(maxsize=1000);errors=[];counts={'camera':0,'detector':0,'queue_drops':0};gaps=0;lastid=None
 def writer():
  try:
   with (folder/'frames.jsonl').open('w') as frames,(folder/'detections.jsonl').open('w') as det:
    while True:
     item=work.get()
     if item is None:break
     kind,data,stamp,rel=item
     if kind=='camera':
      jpg=data.pop('jpg');fn=f"frames/{data['left_id']}_{data['right_id']}.jpg";(folder/fn).write_bytes(jpg)
      if not (folder/'calibration.json').exists():(folder/'calibration.json').write_text(json.dumps(data['calibration'],indent=2))
      frames.write(json.dumps({'file':fn,'left_id':data['left_id'],'right_id':data['right_id'],'capture_monotonic_ns':data.get('capture_monotonic_ns'),'received_unix':stamp,'elapsed_s':rel})+'\n')
     else:
      for key in ['source_jpg','preview_jpg','calibration']:data.pop(key,None)
      data.update(received_unix=stamp,elapsed_s=rel);det.write(json.dumps(data)+'\n')
  except Exception as e:errors.append(str(e));stop.set()
 thread=threading.Thread(target=writer,daemon=True);thread.start();start=time.monotonic();next_report=0
 print(f'开始录制 {a.seconds:g} 秒。请连续弹球，尽量让球经过容易漏检的位置。\n保存到：{folder}\n按 Ctrl+C 可提前结束。',flush=True)
 try:
  while not stop.is_set() and time.monotonic()-start<a.seconds:
   for s,_ in poll.poll(100):
    raw=s.recv();data=msgpack.unpackb(raw,raw=False);kind=sockets[s];rel=time.monotonic()-start
    if kind=='camera':
     fid=data['left_id']
     if lastid is not None and fid>lastid:gaps+=max(0,fid-lastid-1)
     lastid=fid
    try:work.put_nowait((kind,data,time.time(),rel));counts[kind]+=1
    except queue.Full:counts['queue_drops']+=1
   if time.monotonic()-start>=next_report:
    print(f"已录 {time.monotonic()-start:.0f}s | 双目图像 {counts['camera']} | 检测消息 {counts['detector']}",flush=True);next_report+=5
 finally:
  elapsed=time.monotonic()-start
  for s in sockets:s.close(0)
  ctx.term()
  while thread.is_alive():
   try:work.put(None,timeout=.2);break
   except queue.Full:continue
  thread.join()
 summary={'duration_s':elapsed,**counts,'camera_frame_id_gaps':gaps,'errors':errors,'camera_endpoint':a.camera,'detector_endpoint':a.detector,'notes':['Images are original received side-by-side JPEGs, including detector misses.','Join detector frame_id to camera left_id; absent messages are not equivalent to no detection.','capture_monotonic_ns is a robot receive timestamp, not exposure time.','Video uses estimated constant FPS for convenience; frames.jsonl preserves receipt timing.']}
 # Offline preview export keeps encoding outside acquisition.
 if counts['camera'] and not errors:
  records=[json.loads(x) for x in (folder/'frames.jsonl').read_text().splitlines()];detections={}
  for line in (folder/'detections.jsonl').read_text().splitlines():
   d=json.loads(line);detections[d['frame_id']]=d
  fps=(len(records)-1)/max(.001,records[-1]['elapsed_s']-records[0]['elapsed_s']) if len(records)>1 else 30
  first=cv2.imread(str(folder/records[0]['file']));h,w=first.shape[:2];video=cv2.VideoWriter(str(folder/'review.avi'),cv2.VideoWriter_fourcc(*'MJPG'),fps,(w,h));summary['video_created']=video.isOpened();summary['preview_fps']=fps
  matched=0
  if video.isOpened():
   for r in records:
    im=cv2.imread(str(folder/r['file']));d=detections.get(r['left_id']);text=f"{r['elapsed_s']:.2f}s frame {r['left_id']}"
    if d is None:text+=' | NO DETECTOR MESSAGE'
    else:
     matched+=1;text+=f" | L:{len(d['boxes'][0])} R:{len(d['boxes'][1])}"
     for side,boxes in enumerate(d['boxes']):
      for x1,y1,x2,y2,confidence,*_ in boxes:
       xoff=side*(w//2);cv2.rectangle(im,(round(x1)+xoff,round(y1)),(round(x2)+xoff,round(y2)),(0,255,0),1)
    cv2.putText(im,text,(5,15),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1);video.write(im)
   video.release()
  summary['frames_with_matching_detection_message']=matched
 (folder/'summary.json').write_text(json.dumps(summary,indent=2))
 print(f'录制结束：{folder}\n原始图像：frames/；带检测框视频：review.avi；统计：summary.json',flush=True)
 if not counts['camera']:print('警告：没有收到相机图像，请检查机器人相机服务。',flush=True)
 if not counts['detector']:print('警告：没有收到检测结果；请检查球检测服务。',flush=True)
 if errors:raise RuntimeError('; '.join(errors))

if __name__=='__main__':main()
