"""Direct-wrist racket; source geometry and explicit provisional attachment dynamics."""
import sys, json, hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from impact_env import make_spec
from decoder_env import REPO
URDF=REPO/'decoupled_wbc/control/robot_model/model_data/g1/g1_29dof.urdf'
CONFIG={'mount':'direct wrist bracket','dex3_removed':True,'wrist_to_blade_center_m':[.2365,0,0],
 'contact_offset_m':[.2365,.0062,0],'face_normal_wrist':[0,1,0],
 'blade_mass_kg':.14,'handle_mass_kg':.025,'bracket_mass_kg':.035,
 'attachment_dynamics_measured':False,'hardware_approved':False,
 'geometry_source':'https://github.com/dvij542/table-tennis-motion-kit',
 'source_commit':'c6c4c061f8fcb24714fbe118a31e66c69adeb4bf',
 'bare_wrist_inertia_source':str(URDF),'bare_wrist_inertia_sha256':hashlib.sha256(URDF.read_bytes()).hexdigest(),
 'left_terminal':'bare wrist; no ball holder in this experiment',
 'collision_geometry':'blade cylinder, handle capsule, bracket box; approximations',
 'physics':'MuJoCo contact; no drag or spin model; contact parameters provisional'}

def model(dt=.001,mass_scale=1.):
 s=make_spec();s.option.timestep=dt
 for a in list(s.actuators):
  if '_hand_' in a.name:s.delete(a)
 for side in ('left','right'):
  for digit in ('thumb','middle','index'):s.delete(s.body(f'{side}_hand_{digit}_0_link'))
  wrist=s.body(f'{side}_wrist_yaw_link')
  for g in list(wrist.geoms):
   if '_hand_' in g.meshname:s.delete(g)
  el=ET.parse(URDF).getroot().find(f"link[@name='{side}_wrist_yaw_link']/inertial")
  wrist.mass=float(el.find('mass').get('value'));wrist.ipos=np.fromstring(el.find('origin').get('xyz'),sep=' ')
  inertia=el.find('inertia');ixx,iyy,izz,ixy,ixz,iyz=[float(inertia.get(k)) for k in ('ixx','iyy','izz','ixy','ixz','iyz')]
  values,vectors=np.linalg.eigh([[ixx,ixy,ixz],[ixy,iyy,iyz],[ixz,iyz,izz]])
  if np.linalg.det(vectors)<0:vectors[:,0]*=-1
  wrist.inertia=values;wrist.iquat=Rotation.from_matrix(vectors).as_quat()[[3,0,1,2]];wrist.explicitinertial=True
 f=s.geom('training_face');f.pos=[.2365,0,0];f.size=[.075,.0062,0];f.mass=CONFIG['blade_mass_kg']*mass_scale
 h=s.geom('training_handle');h.fromto=[.09,0,0,.17,0,0];h.mass=CONFIG['handle_mass_kg']*mass_scale
 s.body('training_racket').add_geom(name='training_bracket',type=mujoco.mjtGeom.mjGEOM_BOX,
  pos=[.07,0,0],size=[.025,.018,.018],mass=CONFIG['bracket_mass_kg']*mass_scale,rgba=[.25,.25,.28,1])
 s.pair('ball_face_contact').solref=[.016,.1]
 for name,pos,size,color in [('table_top',[1.73,0,.7475],[1.37,.7625,.0125],[.07,.24,.42,1]),('table_net',[1.73,0,.83625],[.004,.8,.07625],[.8,.85,.9,.6])]:
  s.worldbody.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_BOX,pos=pos,size=size,rgba=color)
 for i,x in enumerate((.66,2.8)):
  for j,y in enumerate((-.60,.60)):
   s.worldbody.add_geom(name=f'table_leg_{i}_{j}',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[x,y,.36],size=[.025,.025,.36],rgba=[.2,.2,.2,1])
 s.add_pair(name='ball_table_contact',geomname1='table_top',geomname2='training_ball_geom',condim=3,solref=[.016,.1],solimp=[.95,.99,.001,.5,2],friction=[.1,.1,.001,.0001,.0001])
 m=s.compile()
 assert m.nu==29 and not any('_hand_' in m.body(i).name for i in range(m.nbody))
 return m
