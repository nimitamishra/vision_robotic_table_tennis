"""Live front-table simulation: learned lift, then SONIC-token interception."""
import argparse, hashlib, json, threading, time
from pathlib import Path
import mujoco, numpy as np, torch, trimesh, viser
from scipy.spatial.transform import Rotation
from decoder_env import ROOT, SEED, decoder_hash
from front_env import FrontEnv, CONFIG, config_hash, sample_cases
from front_model import TABLE
from train_front import ImpactPolicy, predict
from train_lift import LiftPolicy, observation as lift_observation


def load(path):
    cp=torch.load(path,map_location='cpu',weights_only=True)
    if not cp.get('simulation_only') or cp.get('hardware_approved'):raise ValueError('Expected simulation-only checkpoint')
    if cp['decoder_sha256']!=decoder_hash() or cp['environment_sha256']!=config_hash():raise ValueError('Checkpoint environment mismatch')
    if cp['residual_scale']!=CONFIG['residual_scale']:raise ValueError('Token scale mismatch')
    token=np.asarray(cp['base_token'])
    if token.shape!=(64,) or not np.isfinite(token).all():raise ValueError('Invalid base token')
    policy=ImpactPolicy();policy.load_state_dict(cp['state_dict']);return policy.eval(),cp


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8092)
    ap.add_argument('--checkpoint',type=Path);ap.add_argument('--autoplay',action='store_true');a=ap.parse_args()
    if a.checkpoint is None:
        manifest=json.loads((ROOT/'front-deployment.json').read_text());a.checkpoint=Path(manifest['checkpoint'])
        if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()!=manifest['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        if hashlib.sha256(Path(manifest['lift_checkpoint']).read_bytes()).hexdigest()!=manifest['lift_checkpoint_sha256']:raise ValueError('Lift checkpoint changed')
    torch.set_num_threads(1);policy,cp=load(a.checkpoint);env=FrontEnv()
    lift=LiftPolicy();lift.load_state_dict(torch.load(ROOT/'runs/front-lift-v1/policy.pt',weights_only=True)['state_dict']);lift.eval()
    ready_seed=env.base.seed;home_seed=dict(np.load(SEED));base_token=np.array(cp['base_token'],dtype=np.float32)
    report_path=a.checkpoint.parent/'report.json';report=json.loads(report_path.read_text()) if report_path.exists() else None
    server=viser.ViserServer(host='127.0.0.1',port=a.port);server.scene.set_up_direction('+z')
    server.scene.add_grid('/floor',plane='xy',width=7,height=5,cell_size=.25,position=(1.5,0,0))
    server.gui.add_markdown('## Front Table Tennis\n'
        '**Learned lift → front feed → SONIC-token policy**\n\n'
        'The ball comes over the net and bounces on the robot’s half. '
        'A legal return must touch the racket face, clear the net, and land on the opposite half.\n\n'
        'Table: 2.74 × 1.525 m · Height: 0.76 m · Net: 0.1525 m.')
    if report:
        r=report['test_policy'];n=r['episodes']
        server.gui.add_markdown(f"Held-out {n} balls: **{r['face_contacts']} face contacts · {r['net_crossings']} net crossings · {r['legal_returns']} legal returns**.\n\n"
            'Experimental simulation policy. Mount and ball-contact dynamics are provisional.')
    else:server.gui.add_markdown('**Training checkpoint — evaluation is still in progress.**')
    mode=server.gui.add_dropdown('Interception controller',options=('Trained policy','Hold raised racket'))
    new=server.gui.add_button('Lift and serve new ball');replay=server.gui.add_button('Replay same ball')
    pause=server.gui.add_checkbox('Paused',initial_value=False)
    repeat=server.gui.add_checkbox('Repeat balls',initial_value=a.autoplay)
    speed=server.gui.add_slider('Playback speed',min=.25,max=1.,step=.25,initial_value=1.)
    front=server.gui.add_button('Front camera');overview=server.gui.add_button('Overview camera')
    status=server.gui.add_html('<div style="height:230px">Ready.</div>')
    def set_camera(client,straight=False):
        client.camera.position=(4.7,0,1.9) if straight else (4.3,-2.8,2.3)
        client.camera.look_at=(1.45,0,.85)
    @server.on_client_connect
    def connected(client):set_camera(client,True)
    @front.on_click
    def front_camera(_):
        for client in server.get_clients().values():set_camera(client,True)
    @overview.on_click
    def overview_camera(_):
        for client in server.get_clients().values():set_camera(client)
    lock=threading.Lock();state={'request':'new' if a.autoplay else None}
    @new.on_click
    def new_ball(_):
        with lock:state['request']='new'
    @replay.on_click
    def same_ball(_):
        with lock:state['request']='same'
    handles=[];m=env.model
    for i in range(m.ngeom):
        name=m.geom(i).name or ''
        if m.geom_type[i]==7 and m.geom_contype[i]==0:
            mi=int(m.geom_dataid[i]);va=int(m.mesh_vertadr[mi]);vn=int(m.mesh_vertnum[mi]);fa=int(m.mesh_faceadr[mi]);fn=int(m.mesh_facenum[mi])
            vertices=m.mesh_vert[va:va+vn].copy();faces=m.mesh_face[fa:fa+fn].copy()
        elif name in ('training_face','training_handle'):
            mesh=trimesh.creation.cylinder(radius=m.geom_size[i,0],height=2*m.geom_size[i,1],sections=48);vertices,faces=mesh.vertices,mesh.faces
        elif name.startswith('table_'):
            mesh=trimesh.creation.box(extents=m.geom_size[i]*2);vertices,faces=mesh.vertices,mesh.faces
        else:continue
        h=server.scene.add_mesh_simple(f'/scene/geom_{i}',vertices=vertices,faces=faces,
            color=tuple((m.geom_rgba[i,:3]*255).astype(int)),opacity=float(m.geom_rgba[i,3]))
        handles.append((i,h))
    x0,x1=TABLE['near_x'],TABLE['far_x'];w=TABLE['width']/2;z=TABLE['height']+.001
    lines=np.array([[[x0,-w,z],[x1,-w,z]],[[x0,w,z],[x1,w,z]],[[x0,-w,z],[x0,w,z]],[[x1,-w,z],[x1,w,z]],[[x0,0,z],[x1,0,z]]])
    server.scene.add_line_segments('/table/lines',points=lines,colors=(245,245,245),line_width=3)
    ball=server.scene.add_icosphere('/ball',radius=.02,color=(255,220,40))
    target=server.scene.add_icosphere('/intercept',radius=.009,color=(30,220,120))
    trail=server.scene.add_line_segments('/ball_path',points=np.zeros((1,2,3)),colors=(255,180,25),line_width=3)
    rng=np.random.default_rng(4172);case=sample_cases(rng,1)[0];obs=env.reset(case)
    phase='Ready';k=0;last=home_seed['token'].copy();info=None;path=[];finished=time.monotonic();count=0;deadline=time.monotonic();selected=mode.value
    while True:
        with lock:request=state['request'];state['request']=None
        if request:
            if request=='new':case=sample_cases(rng,1)[0]
            env.base.on_substep=None;env.base.seed=home_seed;env.base.reset([0,0,0]);env.base.seed=ready_seed
            last=home_seed['token'].copy();phase='Lifting racket';k=0;info=None;path=[];selected=mode.value
            # Park the ball away from the robot until the lifting model is ready.
            env.data.qpos[env.ball_qa:env.ball_qa+3]=[6,0,3];env.data.qpos[env.ball_qa+3:env.ball_qa+7]=[1,0,0,0]
            env.data.qvel[env.ball_va:env.ball_va+6]=0;mujoco.mj_forward(m,env.data)
        if not pause.value and phase=='Lifting racket':
            with torch.no_grad():token=lift(torch.from_numpy(lift_observation(env.base,last,k*.02))).numpy().clip(-.6,.6)
            env.base.base_token=token;_,_,_,info=env.base.step(np.zeros(64));last=token;k+=1
            if info['unsafe']:phase='Posture limit';finished=time.monotonic()
            elif k>=250:
                if env.face()[2]<.88:phase='Lift failed';finished=time.monotonic()
                else:
                    env.base.base_token=base_token.copy();env.base.start_joints=env.data.qpos[env.base.bqa].copy();env.base.start_root=env.data.qpos[:3].copy()
                    env.base.start=env.base.point().copy();env.base.last_residual=np.zeros(64)
                    obs=env.reset(case,reset_robot=False);env.base.on_substep=env._contacts
                    phase='Front feed';info=None;path=[env.ball_position()]
        elif not pause.value and phase=='Front feed':
            obs,_,done,info=env.step(np.zeros(64) if selected=='Hold raised racket' else predict(policy,obs));path.append(env.ball_position())
            if done:
                phase='Legal return' if info['legal_return'] else 'Attempt finished';finished=time.monotonic()
        elif phase not in ('Lifting racket','Front feed') and repeat.value and not pause.value and time.monotonic()-finished>1.5:new_ball(None)
        if count%2==0:
            with server.atomic():
                for i,h in handles:
                    h.position=env.data.geom_xpos[i];q=Rotation.from_matrix(env.data.geom_xmat[i].reshape(3,3)).as_quat();h.wxyz=q[[3,0,1,2]]
                ball.visible=bool(path);trail.visible=bool(path);target.visible=phase=='Front feed'
                ball.position=env.ball_position();target.position=env.target
                if len(path)>1:trail.points=np.stack((path[:-1],path[1:]),axis=1)
                elif path:trail.points=np.array([[path[0],path[0]]])
                details=info or {}
                result_label={'return_wrong_side':'Return landed short or outside target half',
                    'ball_obstacle_':'Ball hit robot','ball_obstacle_training_handle':'Ball hit racket handle',
                    'ball_obstacle_floor':'Ball hit floor','racket_rim':'Ball hit racket rim',
                    'second_bounce':'Second bounce before interception','net_contact':'Ball hit net',
                    'posture_limit':'Robot posture limit','racket_obstacle_contact':'Racket hit an obstacle'}.get(details.get('failure'),details.get('failure') or '—')
                status.content=(f'<div style="height:230px"><b>{"Paused" if pause.value else phase}</b><br>'
                    f'Racket height: {env.face()[2]*100:.1f} cm<br>Table height: 76.0 cm<br>'
                    f'Own-side bounces: {details.get("own_bounces",0)}<br>Face contact: {details.get("face_contact",False)}<br>'
                    f'Over net: {details.get("crossed_net",False)}<br>Legal landing: {details.get("legal_return",False)}<br>'
                    f'Result: {result_label}<br>Hardware output: disabled</div>')
            server.flush()
        count+=1;deadline+=.02/speed.value;time.sleep(max(0,deadline-time.monotonic()))
        if time.monotonic()-deadline>.1:deadline=time.monotonic()

if __name__=='__main__':main()
