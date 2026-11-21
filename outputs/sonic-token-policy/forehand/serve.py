"""Reference and recorded SONIC dynamics comparison. Local preview, no robot IO."""
import argparse,time
import trimesh,viser
from viser.extras import ViserUrdf
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=52175);p.add_argument('--autoplay',action='store_true');args=p.parse_args()
    manifest=json.loads((HERE.parent/'forehand-preview.json').read_text())
    for key in ('checkpoint','tracking_checkpoint'):
        if sha(manifest[key])!=manifest[key+'_sha256']:raise ValueError('Preview checkpoint changed; re-evaluate before showing it')
    sys.path.insert(0,str(SOURCE/'scripts'))
    from build_table_tennis_forehand_bundle import load_robot
    ms=motions();by_offset={tuple(np.round(m['meta']['target_offset_world_m'][1:],2)):m for m in ms}
    server=viser.ViserServer(host='127.0.0.1',port=args.port);server.scene.set_up_direction('+z')
    server.gui.configure_theme(control_layout='floating',control_width='medium')
    server.gui.add_markdown('## Forehand Motion Training\n**Public forehand demonstrations → learned SONIC tokens**\n\n'
        'Backswing → forward swing → follow-through → recovery.\n\n'
        'The green marker is a contact target, **not a simulated ball**. This preview evaluates motion tracking, not successful returns.')
    options=['Learned tokens · MuJoCo replay','Reference · kinematic only','Encoded reference · MuJoCo replay']
    mode=server.gui.add_dropdown('View',options=options,initial_value=options[0])
    y=server.gui.add_slider('Target Y offset (m)',min=-.15,max=.05,step=.05,initial_value=0.)
    z=server.gui.add_slider('Target Z offset (m)',min=0,max=.15,step=.05,initial_value=.10)
    playing=server.gui.add_checkbox('Play',initial_value=args.autoplay)
    speed=server.gui.add_slider('Playback speed',min=.25,max=1.,step=.25,initial_value=.5)
    frame=server.gui.add_slider('Motion frame',min=0,max=125,step=1,initial_value=0)
    hit=server.gui.add_button('Show contact pose');front=server.gui.add_button('Front camera');overview=server.gui.add_button('Table overview')
    status=server.gui.add_html('<div style="height:225px">Loading...</div>')
    server.gui.add_markdown('**Simulation only.** MuJoCo replay uses recorded physics driven by SONIC. '
        'Reference view directly animates the supplied poses. Racket mass and mount are provisional; no hardware output.')
    floor=server.scene.add_grid('/floor',plane='xy',width=7,height=5,cell_size=.25)
    reference=server.scene.add_frame('/reference',show_axes=False)
    visual=ViserUrdf(server,load_robot(SOURCE/'assets/vicar/g1.urdf'),root_node_name='/reference')
    m=model();d=mujoco.MjData(m);physics=server.scene.add_frame('/physics',show_axes=False);handles=[]
    for i in range(m.ngeom):
        name=m.geom(i).name or ''
        if m.geom_type[i]==7 and m.geom_contype[i]==0:
            mi=int(m.geom_dataid[i]);va=int(m.mesh_vertadr[mi]);vn=int(m.mesh_vertnum[mi]);fa=int(m.mesh_faceadr[mi]);fn=int(m.mesh_facenum[mi]);mesh=trimesh.Trimesh(vertices=m.mesh_vert[va:va+vn],faces=m.mesh_face[fa:fa+fn],process=False)
        elif name=='training_face':mesh=trimesh.creation.cylinder(radius=m.geom_size[i,0],height=2*m.geom_size[i,1],sections=48)
        elif name=='training_handle':mesh=trimesh.creation.capsule(radius=m.geom_size[i,0],height=2*m.geom_size[i,1])
        elif name.startswith('table_'):mesh=trimesh.creation.box(extents=m.geom_size[i]*2)
        else:continue
        parent='/table' if name.startswith('table_') else '/physics'
        h=server.scene.add_mesh_simple(f'{parent}/geom_{i}',vertices=mesh.vertices,faces=mesh.faces,color=tuple((m.geom_rgba[i,:3]*255).astype(int)),opacity=float(m.geom_rgba[i,3]))
        handles.append((i,h))
    zz=.761;w=.7625
    server.scene.add_line_segments('/table/lines',points=np.array([[[.36,-w,zz],[3.1,-w,zz]],[[.36,w,zz],[3.1,w,zz]],[[.36,-w,zz],[.36,w,zz]],[[3.1,-w,zz],[3.1,w,zz]],[[.36,0,zz],[3.1,0,zz]]]),colors=(245,245,245),line_width=2)
    target=server.scene.add_icosphere('/contact_target',radius=.012,color=(40,220,100))
    reference_path=server.scene.add_line_segments('/reference_path',points=np.zeros((1,2,3)),colors=(40,220,100),line_width=2)
    actual_path=server.scene.add_line_segments('/actual_path',points=np.zeros((1,2,3)),colors=(255,160,30),line_width=3)
    def camera(c,wide=False):
        c.camera.fov=np.radians(45)
        c.camera.position=(4.1,-2,2.4) if wide else (2.6,0,1.7);c.camera.look_at=(1.2,0,.7) if wide else (.2,0,.85)
    @server.on_client_connect
    def connected(c):camera(c)
    @front.on_click
    def front_click(_):
        for c in server.get_clients().values():camera(c)
    @overview.on_click
    def overview_click(_):
        for c in server.get_clients().values():camera(c,True)
    @hit.on_click
    def hit_click(_):playing.value=False;frame.value=50
    last=None;rollout=None;trained_path=HERE/'runs/tracking-v1/report.json';count=0
    while True:
        # Refresh once after training finishes, preserving camera and playback.
        if count%50==0 and trained_path.exists() and len(mode.options)==3:
            mode.options=options+['PPO trial · MuJoCo replay'];mode.value='PPO trial · MuJoCo replay'
        selection=(round(y.value,2),round(z.value,2),mode.value)
        if selection!=last:
            selected=by_offset[selection[:2]];last=selection
            suffix='tracked_prior' if mode.value.startswith('PPO') else 'encoded_reference' if mode.value.startswith('Encoded') else 'learned_prior'
            directory=HERE/'runs/tracking-v1/rollouts' if suffix=='tracked_prior' else RUN/'rollouts'
            path=directory/f'{selected["name"]}-{suffix}.json'
            rollout=json.loads(path.read_text()) if path.exists() else None
            target.position=selected['meta']['target_hit_position_world']
            p=selected['actual_contact_pos'];reference_path.points=np.stack([p[:-1],p[1:]],axis=1)
            if rollout:
                pts=np.array([f['face_m'] for f in rollout['frames'] if f.get('time_s',0)>=0]);actual_path.points=np.stack([pts[:-1],pts[1:]],axis=1)
        kin=mode.value.startswith('Reference');i=int(frame.value);reference.visible=kin;physics.visible=not kin;actual_path.visible=not kin
        if kin:
            reference.position=selected['root_pos'][i];reference.wxyz=selected['root_wxyz'][i];visual.update_cfg(selected['cfg_urdf'][i])
            # Set the common static table once via model FK.
            mujoco.mj_forward(m,d);details='Kinematic reference: no physics or balance test.'
        elif rollout:
            f=min(rollout['frames'],key=lambda f:abs(f['phase_s']-i*.02)+(10 if f.get('time_s',0)<0 else 0))
            d.qpos[:]=f['qpos'];mujoco.mj_forward(m,d);r=rollout['report']
            details=f"Hit error (table frame): {r.get('hit_world_error_cm',float('nan')):.1f} cm<br>Arm tracking RMSE: {r['arm_rmse_rad']:.2f} rad<br>Racket obstacle frames: {r['obstacle_contact_frames']}<br>Posture limit triggered: {r['unsafe']}"
        else:details='Rollout not available yet.'
        with server.atomic():
            for gi,h in handles:
                h.position=d.geom_xpos[gi];q=Rotation.from_matrix(d.geom_xmat[gi].reshape(3,3)).as_quat();h.wxyz=q[[3,0,1,2]]
            phase='Backswing' if i<23 else 'Loaded' if i<35 else 'Forward swing' if i<50 else 'Follow-through' if i<65 else 'Settle' if i<75 else 'Recovery'
            status.content=f'<div style="height:225px"><b>{phase} · {i*.02:.2f} s</b><br>Contact at 1.00 s<br>Target height: {selected["meta"]["target_hit_position_world"][2]*100:.0f} cm<br>{details}<br>Green: reference path · Orange: measured path</div>'
        server.flush();time.sleep(.02/speed.value)
        if playing.value:frame.value=(i+1)%126
        count+=1
if __name__=='__main__':main()
