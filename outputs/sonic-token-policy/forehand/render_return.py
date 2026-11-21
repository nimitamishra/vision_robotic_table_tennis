"""Six consecutive held-out physical serves, not a selected success montage."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,subprocess,cv2,imageio.v2 as imageio
from return_env import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--episode',type=int);a=p.parse_args();out=a.run
    m=model(ball_contacts=True);m.vis.global_.offheight=720;m.vis.global_.offwidth=1280
    d=mujoco.MjData(m);r=mujoco.Renderer(m,height=720,width=1280);opt=mujoco.MjvOption();opt.sitegroup[:]=0
    cam=mujoco.MjvCamera();cam.lookat[:]=[1.2,0,.8];cam.distance=3.7;cam.azimuth=180;cam.elevation=-24
    movie=out/('return_test.mp4' if a.episode is None else 'return_selected_example.mp4');proc=subprocess.Popen([str(HERE.parents[2]/'work/ffmpeg'),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','25','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)],stdin=subprocess.PIPE)
    summary=json.loads((out/'report.json').read_text())['test']['summary'];saved=False
    try:
        for ep in (range(6) if a.episode is None else [a.episode]):
            doc=json.loads((out/'test_rollouts'/f'episode_{ep:03d}.json').read_text());path=[]
            for f in doc['frames']:
                d.qpos[:]=f['qpos'];mujoco.mj_forward(m,d);r.update_scene(d,cam,scene_option=opt);path.append(np.array(f['ball_m']))
                recent=path[-150:]
                for p0,p1 in zip(recent[:-1],recent[1:]):
                    g=r.scene.geoms[r.scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([1,.7,.05,1.]));mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.0025,p0,p1);r.scene.ngeom+=1
                w=WIDTH/2;z=TOP+.002
                lines=[([NEAR,-w,z],[FAR,-w,z]),([NEAR,w,z],[FAR,w,z]),([NEAR,-w,z],[NEAR,w,z]),([FAR,-w,z],[FAR,w,z]),([NEAR,0,z],[FAR,0,z])]
                for p0,p1 in lines:
                    g=r.scene.geoms[r.scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.96,.96,.96,1.]));mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.003,p0,p1);r.scene.ngeom+=1
                im=r.render().copy();cv2.rectangle(im,(0,0),(1280,85),(17,25,35),-1)
                label=f'Consecutive test serve {ep+1}/6' if a.episode is None else f'SELECTED EXAMPLE: test serve #{ep+1}'
                cv2.putText(im,f'SONIC BALL RETURN | {label} | 0.5x',(20,31),cv2.FONT_HERSHEY_SIMPLEX,.75,(245,245,245),1,cv2.LINE_AA)
                state='LEGAL RETURN' if f['legal_return'] else 'NET CLEARED' if f['crossed_net'] else 'RACKET CONTACT' if f['hit'] else 'INCOMING'
                cv2.putText(im,f'{state} | Full held-out test: {summary["legal_returns"]}/{summary["episodes"]} legal returns',(20,65),cv2.FONT_HERSHEY_SIMPLEX,.7,(90,245,130),1,cv2.LINE_AA)
                cv2.rectangle(im,(0,680),(1280,720),(17,25,35),-1);cv2.putText(im,'MuJoCo physics | narrow front feeds | provisional racket/contact model | no hardware output',(20,706),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1,cv2.LINE_AA)
                proc.stdin.write(im.tobytes())
                if f['legal_return'] and not saved:imageio.imwrite(out/'return_success.png',im);saved=True
            for _ in range(12):proc.stdin.write(im.tobytes())
    finally:
        proc.stdin.close();status=proc.wait();r.close()
    if status:raise RuntimeError('Video encoding failed')
    print(movie,flush=True)
if __name__=='__main__':main()
