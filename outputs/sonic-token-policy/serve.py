"""Interactive local deployment: neural token policy -> SONIC -> MuJoCo."""
import argparse,json,threading,time
from pathlib import Path
import numpy as np
import torch
import viser
from scipy.spatial.transform import Rotation
from decoder_env import ROOT,TokenReachEnv
from runtime import load_policy,predict

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--port',type=int,default=8092);ap.add_argument('--autoplay',action='store_true');a=ap.parse_args()
    if a.checkpoint is None:a.checkpoint=Path(json.loads((ROOT/'deployment.json').read_text())['checkpoint'])
    torch.set_num_threads(1);env=TokenReachEnv();policy,metadata=load_policy(a.checkpoint,env)
    report=json.loads((a.checkpoint.parent/'report.json').read_text())
    if report['test_policy']['unsafe_episodes'] or report['test_policy']['mean_error_cm']>=report['test_constant_token']['mean_error_cm']:
        raise RuntimeError('Checkpoint did not pass simulation deployment criteria')
    server=viser.ViserServer(host='127.0.0.1',port=a.port)
    server.scene.set_up_direction('+z');server.scene.add_grid('/floor',plane='xy',width=3,height=3,cell_size=.25)
    server.gui.add_markdown('## SONIC Token Policy — Simulation\n'
        '**64D neural token → frozen SONIC decoder → MuJoCo**\n\n'
        'Stage 1: right-hand reaches within 1–3 cm. Full floating-base robot. '
        'Green: target; orange: SONIC hand tracking point.\n\n'
        '**No real-robot commands. No ball impact or calibrated racket model yet.**')
    result=report['test_policy'];baseline=report['test_constant_token']
    server.gui.add_markdown(f"Held-out mean error: **{result['mean_error_cm']:.2f} cm** · "
        f"constant-token baseline: {baseline['mean_error_cm']:.2f} cm.\n\n"
        f"Under 1 cm: {result['under_1cm_fraction']:.0%}. Lower-body motion is penalized, not locked.")
    goal_ui=server.gui.add_vector3('Target offset (cm)',initial_value=(0,-2,0),step=.1)
    start=server.gui.add_button('Run selected target');random=server.gui.add_button('Try random target')
    pause=server.gui.add_button('Pause / Resume');repeat=server.gui.add_checkbox('Repeat random targets',initial_value=a.autoplay)
    status=server.gui.add_html('<div style="height:150px">Ready.</div>')
    state={'request':np.array([0,-.02,0]) if a.autoplay else None,'paused':False,'active':False}
    lock=threading.Lock();rng=np.random.default_rng(517)
    def request(delta):
        if not np.isfinite(delta).all() or not .0099<=np.linalg.norm(delta)<=.0301:
            status.content='<div style="height:150px">Use a target offset with total distance between 1 and 3 cm.</div>';return
        with lock:state['request']=delta
    @start.on_click
    def chosen(_):request(np.asarray(goal_ui.value)/100)
    @random.on_click
    def randomize(_):
        d=rng.normal(size=3);d=d/np.linalg.norm(d)*rng.uniform(.01,.03);goal_ui.value=tuple(d*100);request(d)
    @pause.on_click
    def toggle(_):
        with lock:state['paused']=not state['paused']
    @server.on_client_connect
    def camera(client):client.camera.position=(1.6,-1.8,1.35);client.camera.look_at=(0,0,.7)
    # Render visual meshes from the exact MuJoCo model used for training.
    meshes=[];m=env.model
    for i in range(m.ngeom):
        if m.geom_type[i]!=7 or m.geom_contype[i]!=0:continue
        mesh=int(m.geom_dataid[i]);va=int(m.mesh_vertadr[mesh]);vn=int(m.mesh_vertnum[mesh]);fa=int(m.mesh_faceadr[mesh]);fn=int(m.mesh_facenum[mesh])
        handle=server.scene.add_mesh_simple(f'/robot/geom_{i}',vertices=m.mesh_vert[va:va+vn].copy(),faces=m.mesh_face[fa:fa+fn].copy(),color=tuple((m.geom_rgba[i,:3]*255).astype(int)))
        meshes.append((i,handle))
    hand=server.scene.add_icosphere('/hand',radius=.012,color=(255,155,35))
    target=server.scene.add_icosphere('/target',radius=.012,color=(40,220,100))
    obs=env.observation();info=None;finished_at=0.;counter=0;next_frame=time.monotonic()
    while True:
        with lock:
            requested=state['request'];state['request']=None;paused=state['paused']
        if requested is not None:
            obs=env.reset(requested);state['active']=True;info=None
        if state['active'] and not paused:
            obs,_,done,info=env.step(predict(policy,obs))
            if done:state['active']=False;finished_at=time.monotonic()
        elif repeat.value and not paused and time.monotonic()-finished_at>1.0:
            randomize(None)
        if counter%2==0:
            with server.atomic():
                for i,h in meshes:
                    h.position=env.data.geom_xpos[i]
                    q=Rotation.from_matrix(env.data.geom_xmat[i].reshape(3,3)).as_quat();h.wxyz=q[[3,0,1,2]]
                r=env.data.xmat[env.pelvis].reshape(3,3);p=env.data.xpos[env.pelvis]
                hand.position=p+r@env.point();target.position=p+r@env.goal
                err=100*np.linalg.norm(env.point()-env.goal)
                phase='Paused' if paused else ('Running' if state['active'] else 'Ready / finished')
                status.content=f'<div style="height:150px">{phase}<br>Simulation time: {env.k*env.dt:.2f} s<br>Hand error: {err:.2f} cm<br>Output: 64D token residual<br>SONIC decoder: frozen<br>Hardware output: disabled</div>'
            server.flush()
        counter+=1;next_frame+=env.dt;time.sleep(max(0,next_frame-time.monotonic()))
        if time.monotonic()-next_frame>.1:next_frame=time.monotonic()

if __name__=='__main__':main()
