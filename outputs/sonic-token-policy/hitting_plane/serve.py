"""Local English UI: four inputs -> learned SONIC tokens -> live MuJoCo physics."""
import argparse,time,json
from pathlib import Path
import numpy as np
import viser,trimesh
from scipy.spatial.transform import Rotation
from environment import HittingPlaneEnv
from train_residual import ResidualPolicy,BASE
from export_policy import ExportedPolicy
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=52180);p.add_argument('--run',type=Path,default=HERE/'runs/bracket-v4');p.add_argument('--autoplay',action='store_true');a=p.parse_args()
 cp=json.loads((a.run/'policy.json').read_text());basis=np.load(a.run/'basis.npy')
 if not cp['simulation_only'] or cp['hardware_approved']:raise ValueError('Expected simulation-only checkpoint')
 env=HittingPlaneEnv(policy=ExportedPolicy(a.run/'policy.ts'));m,d=env.m,env.d
 server=viser.ViserServer(host='127.0.0.1',port=a.port);server.scene.set_up_direction('+z')
 server.gui.configure_theme(control_layout='floating',control_width='medium')
 server.gui.add_markdown('## Hitting Plane → SONIC\n**XYZ + time → 64D token → frozen SONIC → MuJoCo**\n\nDirect wrist bracket · simulation only')
 with server.gui.add_folder('Strike inputs',expand_by_default=True):
  server.gui.add_number('Plane X (m)',initial_value=.43,disabled=True)
  y=server.gui.add_slider('Target Y (m)',min=-.53,max=-.33,step=.005,initial_value=-.38)
  z=server.gui.add_slider('Target Z (m)',min=.078,max=.228,step=.005,initial_value=.178)
  deadline=server.gui.add_slider('Time to hit (s)',min=.8,max=1.,step=.02,initial_value=1.)
  server.gui.add_markdown('XYZ is relative to the pelvis **at episode start**. +X forward, +Y left, +Z up. Initial world pelvis height: 0.782 m.')
 run=server.gui.add_button('Run strike');pause=server.gui.add_checkbox('Play',initial_value=True)
 repeat=server.gui.add_checkbox('Repeat same inputs',initial_value=a.autoplay)
 speed=server.gui.add_slider('Playback speed',min=.25,max=1.,step=.25,initial_value=.5)
 front=server.gui.add_button('Front camera');overview=server.gui.add_button('Table overview')
 status=server.gui.add_html('<div style="height:300px">Initializing…</div>')
 server.gui.add_markdown('Yellow: ball · green: racket normal · cyan: requested hit point.\n\nThe neural prior plus learned token head uses no online motion encoder. The frozen SONIC decoder also receives joint state and IMU history.\n\nAttachment mass, contact response and ideal ball inputs are provisional. A single trained serve is not a general rally policy.')
 def camera(c,wide=False):
  c.camera.fov=np.radians(50);c.camera.position=(4.2,-2.5,2.5) if wide else (3.7,0,2.1);c.camera.look_at=(1.3,0,.8) if wide else (1.05,0,.9)
 @server.on_client_connect
 def connect(c):camera(c)
 @front.on_click
 def front_clicked(_):
  for c in server.get_clients().values():camera(c)
 @overview.on_click
 def overview_clicked(_):
  for c in server.get_clients().values():camera(c,True)
 server.scene.add_grid('/floor',plane='xy',width=7,height=5,cell_size=.25)
 handles=[]
 for i in range(m.ngeom):
  name=m.geom(i).name or '';typ=int(m.geom_type[i]);size=m.geom_size[i]
  if typ==7 and m.geom_contype[i]==0:
   mi=int(m.geom_dataid[i]);va=int(m.mesh_vertadr[mi]);vn=int(m.mesh_vertnum[mi]);fa=int(m.mesh_faceadr[mi]);fn=int(m.mesh_facenum[mi]);mesh=trimesh.Trimesh(vertices=m.mesh_vert[va:va+vn],faces=m.mesh_face[fa:fa+fn],process=False)
  elif name=='training_face':mesh=trimesh.creation.cylinder(radius=size[0],height=2*size[1],sections=48)
  elif name=='training_handle':mesh=trimesh.creation.capsule(radius=size[0],height=2*size[1])
  elif name.startswith('table_') or name=='training_bracket':mesh=trimesh.creation.box(extents=size*2)
  else:continue
  h=server.scene.add_mesh_simple(f'/scene/{i}',vertices=mesh.vertices,faces=mesh.faces,color=tuple((m.geom_rgba[i,:3]*255).astype(int)),opacity=float(m.geom_rgba[i,3]));handles.append((i,h))
 w=.7625;zz=.761
 server.scene.add_line_segments('/table_lines',points=np.array([[[.36,-w,zz],[3.10,-w,zz]],[[.36,w,zz],[3.10,w,zz]],[[.36,-w,zz],[.36,w,zz]],[[3.10,-w,zz],[3.10,w,zz]],[[.36,0,zz],[3.10,0,zz]]]),colors=(245,245,245),line_width=3)
 server.scene.add_line_segments('/hit_plane',points=np.array([[[.43,-.6,.80],[.43,-.3,.80]],[[.43,-.3,.80],[.43,-.3,1.08]],[[.43,-.3,1.08],[.43,-.6,1.08]],[[.43,-.6,1.08],[.43,-.6,.80]]]),colors=(30,190,230),line_width=2)
 ball=server.scene.add_icosphere('/ball',radius=.02,color=(255,220,40));target=server.scene.add_icosphere('/target',radius=.012,color=(20,220,240))
 trail=server.scene.add_line_segments('/ball_path',points=np.zeros((1,2,3)),colors=(255,185,20),line_width=3)
 normal=server.scene.add_line_segments('/face_normal',points=np.zeros((1,2,3)),colors=(30,230,140),line_width=4)
 state={'request':True};done=True;finished=0.;path=[];error=None
 @run.on_click
 def run_clicked(_):state['request']=True
 while True:
  if state['request']:
   state['request']=False
   try:env.reset(y.value,z.value,deadline.value);done=False;error=None;path=[]
   except ValueError as exc:error=str(exc);done=True
  start=time.monotonic()
  if pause.value and not done:
   done=env.step()
   if env.launched:path.append(env.ball())
   if done:finished=time.monotonic()
  elif done and repeat.value and pause.value and not error and time.monotonic()-finished>1.5:state['request']=True
  if error:
   status.content=f'<div style="height:300px">Cannot start: {error}</div>';time.sleep(.05);continue
  r=env.report()
  with server.atomic():
   for gi,h in handles:h.position=d.geom_xpos[gi];h.wxyz=Rotation.from_matrix(d.geom_xmat[gi].reshape(3,3)).as_quat()[[3,0,1,2]]
   ball.position=env.ball();ball.visible=env.launched;target.position=env.world_goal
   trail.points=np.stack([path[:-1],path[1:]],axis=1) if len(path)>1 else np.zeros((1,2,3))
   normal.points=np.array([[env.point(),env.point()+.18*env.normal()]])
   label='LEGAL RETURN' if r['legal_return'] else 'Episode finished' if done else 'Running physics'
   err='—' if r['face_error_at_deadline_cm'] is None else f"{r['face_error_at_deadline_cm']:.1f} cm"
   status.content=f'<div style="height:300px"><b>{label}</b><br>Time: {d.time:.2f} s · remaining: {env.deadline-d.time:.2f} s<br>Input XYZ: {env.goal.round(3).tolist()}<br>Output: 64D SONIC token · norm {np.linalg.norm(env.token):.3f}<br>Own-side bounce: {env.own_bounces}<br>Face contact: {env.hit}<br>Net cleared: {env.crossed}<br>Opponent landing: {env.landed}<br>Racket error at deadline: {err}<br>Result: {env.failure or "—"}<br>Online encoder calls: 0<br>Hardware output: disabled</div>'
  server.flush();time.sleep(max(.001,.02/speed.value-(time.monotonic()-start)))
if __name__=='__main__':main()
