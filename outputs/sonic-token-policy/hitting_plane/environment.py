"""Learned XYZ/time -> token -> frozen SONIC -> dynamic robot/ball. No robot IO."""
import sys,json
from pathlib import Path
import numpy as np
import mujoco
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'forehand'))
import common as fh
from policy import HittingPlanePolicy
from feed import FeedPlanner
from mount import model as bracket_model

class HittingPlaneEnv:
 def __init__(self,checkpoint=None,dt=.001,mass_scale=1.,policy=None):
  self.policy=policy if policy is not None else HittingPlanePolicy(checkpoint);self.reference=fh.motions()[0]
  self.base=fh.TokenReachEnv(model=bracket_model(dt,mass_scale),steps=1000,point_offset=fh.CONTACT,goal_radius=1)
  self.m,self.d=self.base.model,self.base.data;self.planner=FeedPlanner(dt)
  self.qa=int(self.m.joint('training_ball_joint').qposadr[0]);self.va=int(self.m.joint('training_ball_joint').dofadr[0])
  self.ball_geom=self.m.geom('training_ball_geom').id;self.face_geom=self.m.geom('training_face').id;self.table_geom=self.m.geom('table_top').id;self.net_geom=self.m.geom('table_net').id
  self.racket={self.face_geom,self.m.geom('training_handle').id,self.m.geom('training_bracket').id}
 def point(self):return fh.face(self.base)
 def normal(self):return self.d.xmat[self.base.wrist].reshape(3,3)[:,1].copy()
 def ball(self):return self.d.qpos[self.qa:self.qa+3].copy()
 def reset(self,y_m=-.38,z_m=.178,deadline_s=1.,ball=True):
  self.goal=np.array([.43,float(y_m),float(z_m)]);self.deadline=float(deadline_s);self.with_ball=bool(ball)
  if not .8<=self.deadline<=1.:raise ValueError('Initial time to hit must be 0.80–1.00 s for this first curriculum')
  self.policy.predict(self.goal,self.deadline)
  e=self.base;e.on_substep=None;e.before_substep=None;fh.make_env(self.reference,existing=e)
  self.anchor=self.reference['root_pos'][0].copy();self.world_goal=self.anchor+self.goal
  self.feed=self.planner.plan(self.world_goal) if ball else None
  self.launch_time=self.deadline-self.feed['flight_s'] if ball else None
  if ball and self.launch_time<0:raise ValueError('Time to hit is shorter than the planned serve flight')
  # Startup settles under the same learned four-input policy, not an encoder or reference trajectory.
  for _ in range(25):
   e.base_token=self.policy.predict(self.goal,1.);e.step(np.zeros(64))
  self.d.time=0.;self.actual_launch_time=None;self.k=0;self.launched=False;self.done=False;self.failure=None
  self.frames=[];self.events=[];self.hit=False;self.rim=False;self.crossed=False;self.landed=False;self.own_bounces=0;self.last_bounce=-1.;self.hit_time=None;self.outgoing=None;self.landing=None;self.net_position=None
  self.hit_error_cm=None;self.hit_normal=None;self.peak_drift=0.;self.peak_leg=0.;self.obstacle_contacts=0;self.minimum_distance=10.;self.token=self.policy.predict(self.goal,self.deadline)
  self.previous_ball=self.ball();self.previous_face=self.point();e.on_substep=self.contacts;e.before_substep=self.launch
  return self.report()
 def contacts(self):
  d,m=self.d,self.m;p=self.ball()
  if self.launched:
   self.minimum_distance=min(self.minimum_distance,float(np.linalg.norm(p-self.point())))
  for i,c in enumerate(d.contact):
   pair={int(c.geom1),int(c.geom2)}
   if not pair.intersection(self.racket|{self.ball_geom}):continue
   force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
   if force[0]<1e-5:continue
   if self.ball_geom not in pair:
    if pair.intersection(self.racket) and force[0]>1:
     self.obstacle_contacts+=1;self.failure=self.failure or 'racket_obstacle'
    continue
   if not self.launched:continue
   other=next(g for g in pair if g!=self.ball_geom)
   if other==self.table_geom:
    if d.time-self.last_bounce<=.08:continue
    self.last_bounce=d.time
    if not self.hit:
     if .36<p[0]<1.73:self.own_bounces+=1
     else:self.failure=self.failure or 'invalid_feed'
     if self.own_bounces>1:self.failure=self.failure or 'second_bounce'
    else:
     if self.landing is None:self.landing=p.tolist()
     if self.crossed and 1.75<p[0]<3.08 and abs(p[1])<.7425:self.landed=True
     elif not self.landed:self.failure=self.failure or 'wrong_landing'
    self.events.append({'type':'table','time_s':float(d.time),'position':p.tolist()})
   elif other==self.face_geom:
    if abs(float(c.frame[:3]@self.normal()))<.8:self.rim=True
    elif self.own_bounces!=1:self.failure=self.failure or 'hit_before_bounce'
    elif not self.hit:
     self.hit=True;self.hit_time=float(d.time);self.events.append({'type':'racket_face','time_s':self.hit_time,'position':p.tolist()})
   elif other==self.net_geom:self.events.append({'type':'net','time_s':float(d.time)});self.failure=self.failure or 'net_contact'
   else:
    if not self.landed:self.failure=self.failure or 'ball_obstacle'
  if self.hit and self.outgoing is None and d.time>self.hit_time+.035:
   self.outgoing={'position':p.tolist(),'velocity':d.qvel[self.va:self.va+3].tolist()}
  if self.hit and self.previous_ball[0]<1.73<=p[0] and self.d.qvel[self.va]>0:
   f=(1.73-self.previous_ball[0])/(p[0]-self.previous_ball[0]);at=self.previous_ball+f*(p-self.previous_ball)
   self.net_position=at.tolist()
   if at[2]>.76+.1525+.02 and abs(at[1])<.7425:
    self.crossed=True;self.events.append({'type':'net_crossing','time_s':float(d.time),'position':at.tolist()})
  self.previous_ball=p
 def launch(self):
  if self.with_ball and not self.launched and self.d.time+1e-9>=self.launch_time:
   # Initial ball assignment at launch only. Flight/bounce/impact then use physics.
   self.d.qpos[self.qa:self.qa+7]=[*self.feed['initial_position'],1,0,0,0]
   self.d.qvel[self.va:self.va+6]=[*self.feed['initial_velocity'],0,0,0];mujoco.mj_forward(self.m,self.d)
   self.launched=True;self.previous_ball=self.ball()
   self.actual_launch_time=float(self.d.time)
 def step(self,record=False):
  if self.done:return True
  remaining=self.deadline-self.d.time
  self.token=self.policy.predict(self.goal,remaining);self.base.base_token=self.token
  self.base.step(np.zeros(64));self.k+=1
  legs=float(np.max(abs(self.d.qpos[self.base.bqa[:12]]-self.reference['dof'][0,:12])))
  drift=float(np.linalg.norm(self.d.qpos[:2]-self.anchor[:2]));self.peak_drift=max(self.peak_drift,drift);self.peak_leg=max(self.peak_leg,legs)
  if self.d.qpos[2]<.55 or self.d.xmat[self.base.pelvis].reshape(3,3)[2,2]<.9 or drift>.10 or legs>.30 or not np.isfinite(self.d.qpos).all():self.failure=self.failure or 'posture_limit'
  if self.hit_error_cm is None and self.d.time+1e-8>=self.deadline:
   self.hit_error_cm=float(100*np.linalg.norm(self.point()-self.world_goal));self.hit_normal=self.normal().tolist()
  self.done=bool(self.d.time>=self.deadline+1.3-1e-8 or self.failure in ('posture_limit','racket_obstacle'))
  if record:self.frames.append({'time_s':float(self.d.time),'remaining_s':float(self.deadline-self.d.time),'input':[ *self.goal.tolist(),float(remaining)],'token':self.token.tolist(),'qpos':self.d.qpos.tolist(),'ball_m':self.ball().tolist(),'face_m':self.point().tolist(),'ball_visible':self.launched,'face_contact':self.hit})
  return self.done
 def report(self):
  return {'outgoing':self.outgoing,'landing_m':self.landing,'net_position_m':self.net_position,'target_pelvis_m':self.goal.tolist(),'target_world_m':self.world_goal.tolist(),'deadline_s':self.deadline,'checkpoint_sha256':self.policy.checkpoint_sha256,'simulation_only':True,'online_encoder_calls':0,'face_error_at_deadline_cm':self.hit_error_cm,'face_normal_at_deadline':self.hit_normal,'face_contact':self.hit,'rim_contact':self.rim,'hit_time_s':self.hit_time,'hit_time_error_ms':None if self.hit_time is None else 1000*(self.hit_time-self.deadline),'net_crossed':self.crossed,'opponent_landing':self.landed,'legal_return':bool(self.hit and self.crossed and self.landed and self.own_bounces==1 and not self.rim and self.failure is None),'own_bounces':self.own_bounces,'failure':self.failure,'racket_obstacle_contacts':self.obstacle_contacts,'peak_root_drift_cm':100*self.peak_drift,'peak_leg_change_rad':self.peak_leg,'minimum_ball_face_distance_cm':100*self.minimum_distance if self.launched else None,'completed':self.done and self.d.time>=self.deadline+1.3-1e-8,'actual_launch_time_s':getattr(self,'actual_launch_time',None),'planned_launch_time_s':self.launch_time}
 def run(self,**kwargs):
  self.reset(**kwargs)
  for _ in range(150):
   if self.step(record=True):break
  return self.report()
