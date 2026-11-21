"""Render audited physics recordings; no browser WebGL dependency, no resimulation."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import mujoco,cv2,imageio.v2 as imageio
from mount import model
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--group',default='training_replay');p.add_argument('--count',type=int,default=1);a=p.parse_args()
 m=model();m.vis.global_.offheight=720;m.vis.global_.offwidth=1280;d=mujoco.MjData(m)
 renderer=mujoco.Renderer(m,height=720,width=1280);opt=mujoco.MjvOption();opt.sitegroup[:]=0
 cam=mujoco.MjvCamera();cam.lookat[:]=[1.1,0,.85];cam.distance=3.8;cam.azimuth=180;cam.elevation=-24
 movie=a.run/f'{a.group}.mp4'
 ffmpeg=HERE.parents[2]/'work/ffmpeg'
 proc=subprocess.Popen([str(ffmpeg),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','25','-i','-','-an','-c:v','libx264','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)],stdin=subprocess.PIPE)
 def line(p0,p1,color,width=.003):
  g=renderer.scene.geoms[renderer.scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color));mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,width,p0,p1);renderer.scene.ngeom+=1
 try:
  for ep in range(a.count):
   doc=json.loads((a.run/f'{a.group}_{ep:02d}.json').read_text());path=[];report=doc['report'];goal=np.array(report['target_world_m'])
   for k,f in enumerate(doc['frames']):
    d.qpos[:]=f['qpos'];mujoco.mj_forward(m,d);renderer.update_scene(d,cam,scene_option=opt)
    if f['ball_visible']:path.append(np.array(f['ball_m']))
    for p0,p1 in zip(path[:-1],path[1:]):line(p0,p1,[1,.7,.05,1],.002)
    w=.7625;z=.762
    for p0,p1 in [([.36,-w,z],[3.10,-w,z]),([.36,w,z],[3.10,w,z]),([.36,-w,z],[.36,w,z]),([3.10,-w,z],[3.10,w,z]),([.36,0,z],[3.10,0,z])]:line(p0,p1,[.96,.96,.96,1])
    for axis in np.eye(3):line(goal-axis*.025,goal+axis*.025,[.05,.9,1,1])
    normal=d.xmat[m.body('right_wrist_yaw_link').id].reshape(3,3)[:,1];face=np.array(f['face_m']);line(face,face+.12*normal,[.1,1,.3,1])
    im=renderer.render().copy();cv2.rectangle(im,(0,0),(1280,105),(17,25,35),-1)
    title='Interactive simulation' if a.group=='interactive' else 'Training case replay' if a.group=='training_replay' else f'Consecutive test {ep+1}/{a.count}'
    cv2.putText(im,f'HITTING PLANE > 64D SONIC TOKEN > PHYSICS | {title}',(18,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1,cv2.LINE_AA)
    goaltext=report['target_pelvis_m'];cv2.putText(im,f'XYZ = {goaltext} m | Hit time = {report["deadline_s"]:.2f} s | 0.5x playback',(18,62),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1,cv2.LINE_AA)
    cv2.putText(im,f't = {f["time_s"]:.2f} s | token norm = {np.linalg.norm(f["token"]):.3f} | online encoder calls: 0',(18,92),cv2.FONT_HERSHEY_SIMPLEX,.65,(110,235,245),1,cv2.LINE_AA)
    cv2.rectangle(im,(0,680),(1280,720),(17,25,35),-1)
    events=[v for v in doc['events'] if v['time_s']<=f['time_s']]
    label='INCOMING'
    if any(ev['type']=='racket_face' for ev in events):label='FACE CONTACT'
    if any(ev['type']=='net_crossing' for ev in events):label='NET CLEARED'
    if report['legal_return'] and any(ev['type']=='table' and ev['position'][0]>1.73 for ev in events):label='LEGAL RETURN'
    if k==len(doc['frames'])-1 and not report['legal_return']:label='MISS / '+str(report['failure'] or 'no legal landing')
    cv2.putText(im,f'{label} | Direct bracket model | provisional dynamics | simulation only',(18,707),cv2.FONT_HERSHEY_SIMPLEX,.68,(100,245,150),1,cv2.LINE_AA)
    proc.stdin.write(im.tobytes())
    if ep==0 and abs(f['remaining_s'])<.011:imageio.imwrite(a.run/f'{a.group}_impact.png',im)
   for _ in range(20):proc.stdin.write(im.tobytes())
 finally:
  proc.stdin.close();code=proc.wait();renderer.close()
 if code:raise RuntimeError('Video encoding failed')
 print(movie,flush=True)
if __name__=='__main__':main()
