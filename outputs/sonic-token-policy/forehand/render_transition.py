"""Render measured-start transition diagnostics at actual simulation speed."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import subprocess,cv2
from common import *
out=HERE/'runs/transition-20261120';doc=json.loads((out/'small_lift_supported.json').read_text())
m=model(supported=True);d=mujoco.MjData(m);m.vis.global_.offheight=720;m.vis.global_.offwidth=1280
renderer=mujoco.Renderer(m,height=720,width=1280);opt=mujoco.MjvOption();opt.sitegroup[:]=0
camera=mujoco.MjvCamera();camera.lookat[:]=[.35,0,.82];camera.distance=2.7;camera.azimuth=170;camera.elevation=-16
video=out/'small-lift-diagnostic.mp4'
p=subprocess.Popen([str(HERE.parents[2]/'work/ffmpeg'),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','25','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
try:
 for i,f in enumerate(doc['frames'][::2]):
    d.qpos[:]=f['qpos'];mujoco.mj_forward(m,d);renderer.update_scene(d,camera,scene_option=opt);im=renderer.render().copy()
    cv2.rectangle(im,(0,0),(1280,104),(17,25,35),-1)
    cv2.putText(im,'SMALL LIFT CHECK | measured start -> SONIC tokens -> MuJoCo',(18,30),cv2.FONT_HERSHEY_SIMPLEX,.72,(245,245,245),1,cv2.LINE_AA)
    cv2.putText(im,f'Target: raise 3 cm and return | t = {f["time_s"]:.2f} s | 1x speed',(18,63),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1,cv2.LINE_AA)
    cv2.putText(im,'NOT PASSED: overshoot and lower-body movement | NO ROBOT COMMANDS',(18,91),cv2.FONT_HERSHEY_SIMPLEX,.61,(255,150,90),1,cv2.LINE_AA)
    cv2.rectangle(im,(0,680),(1280,720),(17,25,35),-1)
    cv2.putText(im,'Rigid support approximation; uncalibrated racket and table pose. Diagnostic only.',(18,706),cv2.FONT_HERSHEY_SIMPLEX,.62,(245,245,245),1,cv2.LINE_AA)
    p.stdin.write(im.tobytes())
    if i==100:cv2.imwrite(str(out/'small-lift-diagnostic.png'),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
finally:
 p.stdin.close();code=p.wait();renderer.close()
if code:raise RuntimeError('Encoding failed')
print(video)
