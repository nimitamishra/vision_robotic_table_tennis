"""Live local MuJoCo return policy viewer. No real-robot transports."""
import argparse,time
import trimesh,viser
from return_env import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=52175);p.add_argument('--run',type=Path);p.add_argument('--autoplay',action='store_true');args=p.parse_args()
    if args.run is None:
        manifest=json.loads((HERE.parent/'return-preview.json').read_text());args.run=Path(manifest['run'])
        if sha(args.run/'policy.json')!=manifest['policy_sha256']:raise ValueError('Return checkpoint changed')
    cp=json.loads((args.run/'policy.json').read_text());ref=args.run/'forward_reference.npz'
    if cp['hardware_approved'] or not cp['simulation_only'] or cp['decoder_sha256']!=sha(DECODER) or cp['encoder_sha256']!=sha(DECODER.parent/'model_encoder.onnx') or cp['reference_sha256']!=sha(ref):raise ValueError('Incompatible simulation checkpoint')
    params=np.array(cp['params']);env=ReturnEnv(ref);m,d=env.model,env.data
    server=viser.ViserServer(host='127.0.0.1',port=args.port);server.scene.set_up_direction('+z')
    server.gui.configure_theme(control_layout='floating',control_width='medium')
    server.gui.add_markdown('## Forehand Ball Return\n**SONIC tokens → live MuJoCo physics**\n\n'
        'Reward: hit **+20** · clear the net **+80** · land on the opponent’s table **+200**.\n\n'
        'The ball must bounce once on the robot’s half before the hit. Net contact, rim hits and out-of-bounds returns do not count.')
    rp=args.run/'report.json'
    if rp.exists():
        s=json.loads(rp.read_text())['test']['summary']
        server.gui.add_markdown(f"Held-out narrow feeds: **{s['legal_returns']} / {s['episodes']} legal returns**, {s['net_crossings']} net crossings.\n\nSimulation only; provisional contact and racket parameters.")
    mode=server.gui.add_dropdown('Controller',options=('Trained return policy','Before RL (forward seed)'))
    new=server.gui.add_button('Serve new ball');same=server.gui.add_button('Replay same serve')
    play=server.gui.add_checkbox('Play',initial_value=True);repeat=server.gui.add_checkbox('Repeat serves',initial_value=args.autoplay)
    speed=server.gui.add_slider('Playback speed',min=.25,max=1.,step=.25,initial_value=.5)
    front=server.gui.add_button('Front camera');overview=server.gui.add_button('Table overview')
    status=server.gui.add_html('<div style="height:235px">Ready</div>')
    server.gui.add_markdown('This first curriculum uses a narrow range of front feeds and ideal ball state. '
        'The learned stroke parameters pass through the SONIC encoder and frozen decoder. Hardware output is disabled.')
    def camera(c,wide=False):
        c.camera.fov=np.radians(50);c.camera.position=(4.2,-2.5,2.5) if wide else (3.7,0,2.1);c.camera.look_at=(1.3,0,.8) if wide else (1.15,0,.85)
    @server.on_client_connect
    def connect(c):camera(c)
    @front.on_click
    def fc(_):
        for c in server.get_clients().values():camera(c)
    @overview.on_click
    def oc(_):
        for c in server.get_clients().values():camera(c,True)
    server.scene.add_grid('/floor',plane='xy',width=7,height=5,cell_size=.25)
    handles=[]
    for i in range(m.ngeom):
        name=m.geom(i).name or ''
        if m.geom_type[i]==7 and m.geom_contype[i]==0:
            mi=int(m.geom_dataid[i]);va=int(m.mesh_vertadr[mi]);vn=int(m.mesh_vertnum[mi]);fa=int(m.mesh_faceadr[mi]);fn=int(m.mesh_facenum[mi]);mesh=trimesh.Trimesh(vertices=m.mesh_vert[va:va+vn],faces=m.mesh_face[fa:fa+fn],process=False)
        elif name=='training_face':mesh=trimesh.creation.cylinder(radius=m.geom_size[i,0],height=2*m.geom_size[i,1],sections=48)
        elif name=='training_handle':mesh=trimesh.creation.capsule(radius=m.geom_size[i,0],height=2*m.geom_size[i,1])
        elif name.startswith('table_'):mesh=trimesh.creation.box(extents=m.geom_size[i]*2)
        else:continue
        h=server.scene.add_mesh_simple(f'/scene/geom_{i}',vertices=mesh.vertices,faces=mesh.faces,color=tuple((m.geom_rgba[i,:3]*255).astype(int)),opacity=float(m.geom_rgba[i,3]));handles.append((i,h))
    w=WIDTH/2;zz=TOP+.001
    server.scene.add_line_segments('/table/lines',points=np.array([[[NEAR,-w,zz],[FAR,-w,zz]],[[NEAR,w,zz],[FAR,w,zz]],[[NEAR,-w,zz],[NEAR,w,zz]],[[FAR,-w,zz],[FAR,w,zz]],[[NEAR,0,zz],[FAR,0,zz]]]),colors=(245,245,245),line_width=3)
    ball=server.scene.add_icosphere('/ball',radius=.02,color=(255,220,40));trail=server.scene.add_line_segments('/ball_path',points=np.zeros((1,2,3)),colors=(255,185,20),line_width=3)
    normal=server.scene.add_line_segments('/face_normal',points=np.zeros((1,2,3)),colors=(30,230,140),line_width=4)
    rng=np.random.default_rng(8819);state={'request':'new'};case=NOMINAL;done=False;finished=0.;path=[];count=0;totals=[0,0]
    @new.on_click
    def new_ball(_):state['request']='new'
    @same.on_click
    def same_ball(_):state['request']='same'
    selected=mode.value
    while True:
        request=state['request'];state['request']=None
        if request:
            if request=='new':case=sample_cases(rng,1)[0]
            selected=mode.value;env.reset(params if selected=='Trained return policy' else np.zeros(9),case);done=False;path=[env.bp()]
        if play.value and not done:
            done=env.step();path.append(env.bp())
            if done:finished=time.monotonic();totals[1]+=1;totals[0]+=int(env.landed and not env.failure)
        elif done and repeat.value and play.value and time.monotonic()-finished>1.5:state['request']='new'
        with server.atomic():
            for gi,h in handles:
                h.position=d.geom_xpos[gi];q=Rotation.from_matrix(d.geom_xmat[gi].reshape(3,3)).as_quat();h.wxyz=q[[3,0,1,2]]
            ball.position=env.bp()
            if len(path)>1:trail.points=np.stack([path[:-1],path[1:]],axis=1)
            else:trail.points=np.array([[env.bp(),env.bp()]])
            normal.points=np.array([[env.point(),env.point()+.18*env.normal()]])
            state_label='LEGAL RETURN' if env.landed and not env.failure else 'Attempt finished' if done else 'Returning' if env.hit else 'Incoming ball'
            result=(env.failure or ('landed on opponent table' if env.landed else '—')).replace('_',' ')
            status.content=f'<div style="height:235px"><b>{state_label}</b><br>Time: {d.time:.2f} s<br>Own-side bounces: {env.bounces}<br>Face contact: {env.hit}<br>Net cleared: {env.crossed}<br>Legal landing: {env.landed and not env.failure}<br>Result: {result}<br>Live session: {totals[0]} / {totals[1]} returns<br>Green arrow: racket face normal<br>Hardware output: disabled</div>'
        server.flush();count+=1;time.sleep(.02/speed.value)
if __name__=='__main__':main()
