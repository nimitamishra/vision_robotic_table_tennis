"""Ballistic finite-table forecast. Net/edge contacts are conservative risks.
No spin, drag, racket, robot or net-deformation model; parameters are uncalibrated.
"""
import numpy as np
G=np.array([0.,0.,-9.81])

def descending_time(z,vz,level):
    if z<level-1e-7:return None
    disc=vz*vz+19.62*(z-level)
    if disc<0:return None
    t=(vz+np.sqrt(disc))/9.81
    return float(t) if t>1e-6 else None

def at(p,v,t):return p+v*t+.5*G*t*t

def forecast(position,velocity,spec,horizon=.5,table_z=0.,radius=.02,restitution=.8,tangent_retention=.9):
    if not 0<=restitution<=1 or not 0<=tangent_retention<=1:raise ValueError('Impact parameters must be between 0 and 1')
    p=np.asarray(position,float);v=np.asarray(velocity,float)
    if p.shape!=(3,) or v.shape!=(3,) or not np.all(np.isfinite(np.r_[p,v])):raise ValueError('Finite 3D position and velocity required')
    near=-float(spec['tag_center_inset_m']);far=near+float(spec['length_m']);half=float(spec['width_m'])/2;net_y=(near+far)/2
    contact=table_z+radius;floor=table_z-float(spec['height_m'])+radius
    elapsed=0.;events=[];segments=[];bounces=0;limit='time_horizon';contact_adjustment=0.
    # Small filter penetration is a contact-boundary uncertainty, not a path
    # through the table. Never change the measured/filter state itself.
    over_table=near-radius<=p[1]<=far+radius and abs(p[0])<=half+radius
    if over_table and contact-.005<=p[2]<contact:
        contact_adjustment=float(contact-p[2]);p=p.copy();p[2]=contact
    while elapsed<horizon-1e-7:
        remaining=horizon-elapsed;candidates=[]
        if near-radius<=p[1]<=far+radius and abs(p[0])<=half+radius and p[2]<contact-.005:
            limit='below_table_surface';break
        t=0. if p[2]==contact and v[2]<=0 else descending_time(p[2],v[2],contact)
        if t is not None and t<=remaining:
            q=at(p,v,t)
            if abs(q[0])<=half+radius and near-radius<=q[1]<=far+radius:
                safe=abs(q[0])<half-radius and near+radius<q[1]<far-radius
                candidates.append((t,'table' if safe else 'table_edge_risk',q))
        # Conservative sphere-vs-net test: expanded rectangle/slab, both directions.
        # Includes vertical entry into the slab; stops rather than inventing a net bounce.
        if abs(v[1])>1e-9:
            enter,leave=sorted(((net_y-radius-p[1])/v[1],(net_y+radius-p[1])/v[1]))
            lo=max(0.,enter);hi=min(remaining,leave)
        else:lo,hi=(0.,remaining) if abs(p[1]-net_y)<=radius else (1.,0.)
        if hi>=lo:
            times=[lo,hi]
            for level in [table_z-radius,table_z+float(spec['net_height_m'])+radius]:
                disc=v[2]**2+19.62*(p[2]-level)
                if disc>=0:
                    times.extend(t for t in [(v[2]-np.sqrt(disc))/9.81,(v[2]+np.sqrt(disc))/9.81] if lo<=t<=hi)
            if abs(v[0])>1e-9:
                times.extend(t for t in [(-half-.1-radius-p[0])/v[0],(half+.1+radius-p[0])/v[0]] if lo<=t<=hi)
            for nt in sorted(times):
                q=at(p,v,nt)
                if abs(q[0])<=half+.1+radius+1e-8 and table_z-radius-1e-8<=q[2]<=table_z+float(spec['net_height_m'])+radius+1e-8:
                    candidates.append((float(nt),'net_risk',q));break
        ft=descending_time(p[2],v[2],floor)
        if ft is not None and ft<=remaining:candidates.append((ft,'floor',at(p,v,ft)))
        event=min(candidates,key=lambda x:x[0]) if candidates else None
        duration=event[0] if event else remaining
        times=np.linspace(0,duration,max(2,int(np.ceil(duration/.01))+1))
        segments.append({'phase':'after_bounce' if bounces else 'free_flight','start_time_s':elapsed,
                         'end_time_s':elapsed+duration,'points':[at(p,v,t).tolist() for t in times]})
        elapsed+=duration
        if event is None:break
        _,kind,q=event;incoming=v+G*duration
        item={'type':kind,'time_s':float(elapsed),'ball_center_m':q.tolist(),'incoming_velocity_m_s':incoming.tolist(),
              'basis':'uncalibrated ballistic model'}
        if kind=='table':item['surface_point_m']=[float(q[0]),float(q[1]),float(table_z)]
        events.append(item);limit=kind
        if kind!='table' or bounces>=1:break
        outgoing=np.array([incoming[0]*tangent_retention,incoming[1]*tangent_retention,-incoming[2]*restitution])
        item['outgoing_velocity_m_s']=outgoing.tolist();item['restitution']=restitution;item['tangent_retention']=tangent_retention
        if outgoing[2]<.1:limit='low_energy_contact';break
        p=q;v=outgoing;bounces+=1;limit='time_horizon'
    return {'collision_events':events,'collision_segments':segments,'collision_forecast_duration_s':float(elapsed),
            'collision_forecast_limit':limit,'collision_model':{'restitution':restitution,'tangent_retention':tangent_retention,
            'contact_adjustment_m':contact_adjustment,'contact_tolerance_m':.005,
            'calibrated':False,'net':'conservative expanded rectangle; no rebound','max_table_rebounds':1}}
