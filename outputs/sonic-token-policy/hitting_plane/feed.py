"""Plan a front serve in isolated ball/table physics, without a racket or controller."""
import numpy as np
import mujoco
from scipy.optimize import brentq
class FeedPlanner:
 def __init__(self,dt=.001):
  s=mujoco.MjSpec();s.option.timestep=dt
  s.worldbody.add_geom(name='table',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[1.73,0,.7475],size=[1.37,.7625,.0125])
  s.worldbody.add_geom(name='net',type=mujoco.mjtGeom.mjGEOM_BOX,pos=[1.73,0,.83625],size=[.004,.8,.07625])
  b=s.worldbody.add_body(name='ball');b.add_freejoint(name='ball_joint')
  b.add_geom(name='ball_geom',type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.02,0,0],mass=.0027,solref=[.008,.3],solimp=[.95,.99,.001,.5,2])
  s.add_pair(name='ball_table',geomname1='ball_geom',geomname2='table',condim=3,solref=[.016,.1],solimp=[.95,.99,.001,.5,2],friction=[.1,.1,.001,.0001,.0001])
  self.m=s.compile();self.d=mujoco.MjData(self.m);self.cache={}
 def crossing(self,vz,y,plane_x):
  m,d=self.m,self.d;mujoco.mj_resetData(m,d);d.qpos[:]=[2.05,y,1.15,1,0,0,0];d.qvel[:]=[-2.5,0,vz,0,0,0];mujoco.mj_forward(m,d)
  bounces=0;last=-1;net=False
  for _ in range(round(.95/m.opt.timestep)):
   prev=d.qpos[:3].copy();t=d.time;mujoco.mj_step(m,d)
   for i,c in enumerate(d.contact):
    names={m.geom(int(g)).name for g in (c.geom1,c.geom2)}
    f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
    if f[0]<1e-5:continue
    if 'net' in names:net=True
    if 'table' in names and d.time-last>.08:bounces+=1;last=d.time
   if prev[0]>=plane_x>d.qpos[0]:
    alpha=(prev[0]-plane_x)/(prev[0]-d.qpos[0]);p=prev+alpha*(d.qpos[:3]-prev)
    return {'position':p.tolist(),'flight_s':float(t+alpha*m.opt.timestep),'bounces':bounces,'net_contact':net,'vz':float(vz),'velocity':d.qvel[:3].tolist()}
  return None
 def plan(self,target_world):
  x,y,z=map(float,target_world);key=tuple(np.round([x,y,z],7))
  if key in self.cache:return dict(self.cache[key])
  samples=[]
  for v in np.linspace(1.2,2.9,35):
   c=self.crossing(v,y,x)
   if c and c['bounces']==1 and not c['net_contact']:samples.append((v,c['position'][2]-z))
  solutions=[]
  for (v0,e0),(v1,e1) in zip(samples[:-1],samples[1:]):
   if v1-v0>.051 or e0*e1>0:continue
   def f(v):
    c=self.crossing(v,y,x)
    if c is None or c['bounces']!=1 or c['net_contact']:raise ValueError('Discontinuous serve branch')
    return c['position'][2]-z
   try:v=brentq(f,v0,v1,xtol=1e-7)
   except ValueError:continue
   c=self.crossing(v,y,x)
   if abs(c['position'][2]-z)<.001:solutions.append(c)
  if not solutions:raise ValueError('No verified front serve for this target in the current feed range')
  result=min(solutions,key=lambda c:abs(c['vz']-2.086));result['initial_position']=[2.05,y,1.15];result['initial_velocity']=[-2.5,0,result['vz']]
  self.cache[key]=result;return dict(result)
