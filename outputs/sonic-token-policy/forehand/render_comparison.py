"""Render the supplied reference beside a saved, actual SONIC dynamics rollout."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import cv2,imageio.v2 as imageio,subprocess,contextlib
from common import *
from train_tracking import OUT

@contextlib.contextmanager
def video_writer(path):
    ffmpeg=HERE.parents[2]/'work/ffmpeg'
    process=subprocess.Popen([str(ffmpeg),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1280x640','-r','25','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    try:yield process.stdin
    finally:
        process.stdin.close()
        if process.wait()!=0:raise RuntimeError('Video encoding failed')

def main():
    name='table_tennis_forehand_y+000_z010';ref=next(m for m in motions() if m['name']==name)
    replay=json.loads((OUT/'rollouts'/f'{name}-tracked_prior.json').read_text())
    model_=model();model_.vis.global_.offheight=640;model_.vis.global_.offwidth=640
    d=mujoco.MjData(model_);renderer=mujoco.Renderer(model_,height=640,width=640)
    camera=mujoco.MjvCamera();camera.lookat[:]=[.25,0,.85];camera.distance=2.6;camera.azimuth=180;camera.elevation=-12
    joints=[model_.joint(str(n)).qposadr[0] for n in ref['joint_names']]
    video=OUT/'forehand_comparison.mp4'
    with video_writer(video) as writer:
        for i in range(126):
            images=[]
            for reference in (True,False):
                f=min(replay['frames'],key=lambda f:abs(f['phase_s']-i*.02))
                d.qpos[:]=f['qpos']
                if reference:
                    d.qpos[:3]=ref['root_pos'][i];d.qpos[3:7]=ref['root_wxyz'][i];d.qpos[joints]=ref['dof'][i]
                mujoco.mj_forward(model_,d);renderer.update_scene(d,camera);im=renderer.render().copy()
                cv2.rectangle(im,(0,0),(640,74),(18,25,35),-1)
                label='REFERENCE - kinematic poses' if reference else 'LEARNED TOKENS + PPO - SONIC physics'
                cv2.putText(im,label,(15,29),cv2.FONT_HERSHEY_SIMPLEX,.55,(245,245,245),1,cv2.LINE_AA)
                cv2.putText(im,f'Phase {i*.02:.2f}s | target height 96 cm',(15,57),cv2.FONT_HERSHEY_SIMPLEX,.55,(230,230,230),1,cv2.LINE_AA)
                images.append(im)
            combined=np.concatenate(images,axis=1)
            cv2.rectangle(combined,(0,603),(1280,640),(18,25,35),-1)
            cv2.putText(combined,'0.5x playback | Motion tracking only - no ball-return or hardware validation',(20,628),cv2.FONT_HERSHEY_SIMPLEX,.62,(245,245,245),1,cv2.LINE_AA)
            writer.write(combined.tobytes())
            if i==50:imageio.imwrite(OUT/'contact_comparison.png',combined)
    renderer.close();print(video,flush=True)
if __name__=='__main__':main()
