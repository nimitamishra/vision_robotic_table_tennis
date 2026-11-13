"""Single-ball position/velocity filter. Coordinates: table frame, z upward, SI units."""
from collections import deque
import numpy as np

G=np.array([0.,0.,-9.81])

class Tracker:
    def __init__(self, max_gap=.20, contour_sigma=.012, box_sigma=.025, table_z=0., radius=.02, horizon=.5,
                 table_bounds=None, restitution=.8, tangent_retention=.9):
        self.horizon=horizon;self.max_gap=max_gap; self.sigmas={'contour':contour_sigma,'box':box_sigma}
        self.contact_z=table_z+radius;self.x=None;self.P=None;self.t=None;self.last_seen=None
        self.table_bounds=table_bounds;self.restitution=restitution;self.tangent_retention=tangent_retention
        self.impact_enabled=True
        self.seed=deque(maxlen=6);self.rejects=deque(maxlen=3);self.mode='waiting';self.event=None
        self.stationary_anchor=None;self.previous=None;self.epoch=0;self.seed_min=5
    def reset(self,reason):
        self.x=None;self.P=None;self.seed.clear();self.rejects.clear();self.mode='initializing';self.event=reason
        self.stationary_anchor=None;self.previous=None;self.epoch+=1;self.seed_min=5
    def begin(self,t,p,sigma):
        self.seed.append((t,p.copy()))
        if len(self.seed)<self.seed_min or t-self.seed[0][0]<.05:return
        times=np.array([q[0]-t for q in self.seed]);points=np.array([q[1] for q in self.seed])
        observed_velocity=np.linalg.lstsq(np.column_stack([np.ones(len(times)),times]),points,rcond=None)[0][1]
        if np.max(np.linalg.norm(points-points.mean(axis=0),axis=1))<.018 and np.linalg.norm(observed_velocity)<.15:
            self.x=np.r_[points.mean(axis=0),np.zeros(3)];self.stationary_anchor=self.x[:3].copy();self.mode='stationary'
        else:
            A=np.column_stack([np.ones(len(times)),times])
            coeff=np.linalg.lstsq(A,points-.5*times[:,None]**2*G,rcond=None)[0]
            self.x=np.r_[coeff[0],coeff[1]];self.mode='tracking'
        self.P=np.diag([sigma*sigma]*3+[.4**2]*3);self.last_seen=t;self.seed.clear()
    def propagate(self,dt):
        F=np.eye(6);F[:3,3:]=np.eye(3)*dt
        B=np.vstack([np.eye(3)*(.5*dt*dt),np.eye(3)*dt])
        self.x=F@self.x+B@G;self.P=F@self.P@F.T+B@B.T*4.
    def predict(self,dt):
        """Coast through at most one plausible table impact; never call it a measurement."""
        if not self.impact_enabled or self.table_bounds is None or dt<=0:
            self.propagate(dt);return False
        z=self.x[2]-self.contact_z;vz=self.x[5]
        disc=vz*vz+2*9.81*z
        hit=(vz+np.sqrt(disc))/9.81 if disc>=0 else None
        if z<=0 and vz<0:hit=0.
        if hit is None or not 0<=hit<=dt or vz-9.81*hit>=-.1:
            self.propagate(dt);return False
        x_hit=self.x[0]+self.x[3]*hit;y_hit=self.x[1]+self.x[4]*hit
        xmin,xmax,ymin,ymax=self.table_bounds
        if not xmin+.02<x_hit<xmax-.02 or not ymin+.02<y_hit<ymax-.02:
            self.propagate(dt);return False
        self.propagate(hit)
        self.x[2]=self.contact_z
        self.x[3:5]*=self.tangent_retention
        self.x[5]=-self.x[5]*self.restitution
        J=np.eye(6);J[3,3]=self.tangent_retention;J[4,4]=self.tangent_retention;J[5,5]=-self.restitution
        # Bounce timing and restitution are approximate; inflate uncertainty.
        self.P=J@self.P@J.T+np.diag([.01**2,.01**2,.015**2,.3**2,.3**2,.5**2])
        self.propagate(dt-hit)
        return True
    def step(self,t,p=None,source='box',range_sigma_m=None):
        self.event=None
        if not np.isfinite(t):return {'status':'invalid_timestamp'}
        if self.t is not None and t<=self.t:return {'status':'out_of_order','timestamp_s':t}
        dt=0 if self.t is None else t-self.t
        if self.t is not None and dt>self.max_gap:self.reset('time_gap')
        self.t=t
        if p is not None:
            p=np.asarray(p,dtype=float)
            if p.shape!=(3,) or not np.all(np.isfinite(p)):p=None
        if self.last_seen is not None and t-self.last_seen>self.max_gap and (self.x is not None or self.seed):self.reset('track_lost')
        sigma=self.sigmas.get(source,self.sigmas['box'])
        accepted=False
        if self.x is None:
            if p is not None:self.begin(t,p,sigma);self.last_seen=t;accepted=True
        elif self.mode=='stationary':
            if p is not None:
                if np.linalg.norm(p-self.stationary_anchor)>.035:
                    self.reset('motion_started');self.begin(t,p,sigma)
                else:self.x[:3]=.8*self.x[:3]+.2*p
                self.last_seen=t;accepted=True
        else:
            if self.predict(dt):self.event='predicted_bounce'
            if p is not None:
                residual=p-self.x[:3];R=np.eye(3)*sigma**2
                # Stereo range error grows rapidly when the two ball centers have
                # only a few pixels of disparity. In the table frame that error
                # mostly affects distance along the table, with a smaller vertical
                # component because the head camera looks downward.
                if range_sigma_m is not None and np.isfinite(range_sigma_m) and range_sigma_m>0:
                    R[1,1]=max(2*sigma,range_sigma_m)**2
                    R[2,2]=max(sigma,0.4*range_sigma_m)**2
                else:
                    R[1,1]*=4
                S=self.P[:3,:3]+R;distance=float(residual@np.linalg.solve(S,residual))
                bounce=(self.previous is not None and self.previous[1][2]<self.contact_z+.07
                        and self.x[5]<-.25 and p[2]-self.previous[1][2]>.012)
                if bounce:
                    # Do not fit one parabola across the velocity discontinuity at impact.
                    self.reset('possible_bounce');self.seed_min=3;self.begin(t,p,sigma)
                    self.last_seen=t;accepted=True
                elif distance<16.27: # 3D chi-square gate; defaults must be tuned using real noise.
                    K=np.linalg.solve(S,self.P[:3,:]).T
                    self.x+=K@residual;H=np.zeros((3,6));H[:,:3]=np.eye(3);I=np.eye(6)-K@H
                    self.P=I@self.P@I.T+K@R@K.T;self.last_seen=t;self.rejects.clear();accepted=True
                else:
                    self.rejects.append((t,p.copy()));self.event='measurement_rejected'
                    if len(self.rejects)==3:
                        recent=list(self.rejects)
                        # Require plausible consecutive velocities, rather than adopting one outlier.
                        v1=(recent[1][1]-recent[0][1])/(recent[1][0]-recent[0][0])
                        v2=(recent[2][1]-recent[1][1])/(recent[2][0]-recent[1][0])
                        if max(np.linalg.norm(v1),np.linalg.norm(v2))<25 and np.linalg.norm(v2-v1)<4:
                            self.reset('motion_change');self.seed.extend(recent);self.last_seen=t
        if accepted and p is not None:self.previous=(t,p.copy())
        return self.result(t,accepted,source)
    def result(self,t,accepted=False,source=None):
        result={'timestamp_s':t,'status':self.mode if self.x is not None else 'initializing',
                'event':self.event,'measurement_accepted':accepted,'source':source,'track_epoch':self.epoch,
                'acceleration_source':'gravity model (not measured)','predictions':[]}
        if self.x is None:
            result['status']='initializing' if self.seed else 'waiting_for_3d'
            result['initialization_samples']=len(self.seed)
            return result
        age=max(0.,t-self.last_seen)
        if age>self.max_gap:result['status']='lost';return result
        if age>0:result['status']='coasting'
        result.update(position_m=self.x[:3].tolist(),velocity_m_s=self.x[3:].tolist(),
                      speed_m_s=float(np.linalg.norm(self.x[3:])),measurement_age_s=age,
                      position_std_m=np.sqrt(np.maximum(0,np.diag(self.P)[:3])).tolist(),
                      uncertainty_note='Model covariance; not empirically calibrated accuracy.')
        if self.mode=='stationary':
            result.update(acceleration_m_s2=[0.,0.,0.],acceleration_source='stationary constraint (not measured)');return result
        result['acceleration_m_s2']=G.tolist()
        # Stop the model at the next descending crossing of the contact-height plane.
        z=self.x[2]-self.contact_z;vz=self.x[5]
        disc=vz*vz+2*9.81*z
        hit=(vz+np.sqrt(disc))/9.81 if disc>=0 else None
        if z<=0:hit=0.
        if hit is not None and hit>=0:
            landing=self.x[:3]+self.x[3:]*hit+.5*G*hit**2
            result['contact_plane_crossing']={'time_s':float(hit),'position_m':landing.tolist(),
                 'note':'Height-plane crossing only; check table bounds. No bounce forecast.'}
        result['forecast_path']=[]
        end=self.horizon if hit is None else min(self.horizon,max(0.,hit))
        result['forecast_duration_s']=float(end)
        result['forecast_limit']='contact_plane' if hit is not None and hit<=self.horizon else 'time_horizon'
        if end>0:
            for h in np.linspace(0,end,max(2,int(np.ceil(end/.01))+1)):
                result['forecast_path'].append((self.x[:3]+self.x[3:]*h+.5*G*h*h).tolist())
        for h in [.05,.1,.2,.3,.4,.5,.6,.8,1.]:
            if h>self.horizon:continue
            if hit is not None and h>=hit:continue
            F=np.eye(6);F[:3,3:]=np.eye(3)*h;B=np.vstack([np.eye(3)*(.5*h*h),np.eye(3)*h])
            cov=F@self.P@F.T+4*B@B.T
            result['predictions'].append({'horizon_s':h,'position_m':(self.x[:3]+self.x[3:]*h+.5*G*h*h).tolist(),
                        'position_std_m':np.sqrt(np.maximum(0,np.diag(cov)[:3])).tolist()})
        return result


def measurement(packet):
    """Never blend a contour point from one eye with a box point from the other."""
    for pose,key,source in [('ball_refined','xyz_world_refined','contour'),('ball','xyz_world','box')]:
        p=packet.get(key)
        if packet.get('world_measurement') and packet.get(pose) is not None and p is not None:
            a=np.asarray(p,dtype=float)
            if a.shape==(3,) and np.all(np.isfinite(a)):return a,source
    return None,'unavailable'


def stereo_range_sigma(packet,source):
    """Approximate range noise from optical depth and stereo disparity.

    A small disparity makes a subpixel center error a large metric range error.
    The pixel-error priors below are conservative for low-contrast IR ball boxes
    and contours; this is a filter weight, not a calibrated accuracy claim.
    """
    pose=packet.get('ball_refined' if source=='contour' else 'ball')
    if not isinstance(pose,dict):return None
    try:
        depth=float(pose['xyz_left'][2])
        left=float(pose['left_center_px'][0]);right=float(pose['right_center_px'][0])
    except (KeyError,IndexError,TypeError,ValueError):return None
    disparity=abs(left-right)
    if not np.isfinite([depth,disparity]).all() or depth<=0 or disparity<1:return None
    pixel_sigma=.25 if source=='contour' else .35
    base=.012 if source=='contour' else .025
    return float(np.clip(np.sqrt(2)*pixel_sigma*depth/disparity,2*base,.4))
