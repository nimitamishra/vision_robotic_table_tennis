"""Causal virtual-plane evaluation. Reference crossings are observations, not ground truth."""
import numpy as np

def predicted_crossing(result,plane_y):
    segments=result.get('collision_segments')
    if segments is None:
        points=result.get('forecast_path',[])
        segments=[dict(points=points,start_time_s=0.,end_time_s=result.get('forecast_duration_s',0.))]
    for segment in segments:
        pts=segment['points']
        if len(pts)<2:continue
        times=np.linspace(segment['start_time_s'],segment['end_time_s'],len(pts))
        for i,(a,b) in enumerate(zip(pts,pts[1:])):
            if a[1]>plane_y>=b[1]:
                fraction=(a[1]-plane_y)/(a[1]-b[1]);p=np.array(a)+fraction*(np.array(b)-a)
                return {'time_s':float(times[i]+fraction*(times[i+1]-times[i])),'position_m':p.tolist()}
    return None

class Trial:
    def __init__(self,number,plane_y,width,table_z=0.,max_gap=.1):
        self.number=number;self.plane_y=plane_y;self.width=width;self.table_z=table_z;self.max_gap=max_gap
        self.previous=None;self.last_time=None;self.forecasts=[];self.samples=0;self.packets=0;self.max_observed_gap=0.;self.done=False
    def observe(self,t,raw_position,result):
        if self.done or (self.last_time is not None and t<=self.last_time):return None
        self.last_time=t;self.packets+=1
        prediction=predicted_crossing(result,self.plane_y)
        if prediction is not None:
            self.forecasts.append({'issued_t':t,'arrival_t':t+prediction['time_s'],'position_m':prediction['position_m'],
                                   'desktop_pipeline_age_s':result.get('desktop_pipeline_age_s')})
        if raw_position is None:return None
        p=np.asarray(raw_position,float)
        if p.shape!=(3,) or not np.all(np.isfinite(p)):return None
        self.samples+=1
        previous=self.previous;self.previous=(t,p)
        if previous is None:return None
        t0,p0=previous;dt=t-t0;self.max_observed_gap=max(self.max_observed_gap,dt)
        if not p0[1]>self.plane_y>=p[1]:return None
        if dt>self.max_gap or np.linalg.norm(p-p0)/dt>25:return None
        fraction=(p0[1]-self.plane_y)/(p0[1]-p[1]);cross_t=t0+fraction*dt;point=p0+fraction*(p-p0)
        status='observed_crossing' if abs(point[0])<=self.width/2 and self.table_z+.02<=point[2]<=self.table_z+1.2 else 'outside_test_window'
        record=self.finish(status);record.update(reference_time_s=cross_t,reference_position_m=point.tolist(),reference_bracket_s=dt)
        for lead in [.1,.2,.3]:
            # No post-cutoff measurements can influence this selected forecast.
            eligible=[f for f in self.forecasts if cross_t-lead-.05<=f['issued_t']<=cross_t-lead]
            metric={'requested_lead_s':lead,'status':'no_prediction_in_window'}
            if eligible:
                f=max(eligible,key=lambda x:x['issued_t'])
                metric.update(status='evaluated',actual_lead_s=cross_t-f['issued_t'],
                    position_error_m=float(np.linalg.norm(np.array(f['position_m'])-point)),
                    time_error_s=f['arrival_t']-cross_t,predicted_position_m=f['position_m'],
                    desktop_pipeline_age_s=f['desktop_pipeline_age_s'])
            record['lead_metrics'].append(metric)
        return record
    def finish(self,status=None):
        self.done=True
        if status is None:status='no_packets' if not self.packets else 'no_valid_3d' if not self.samples else 'no_observed_crossing'
        return {'trial':self.number,'status':status,'plane_y_m':self.plane_y,'observed_samples':self.samples,
                'input_packets':self.packets,'max_observed_gap_s':self.max_observed_gap,'lead_metrics':[],
                'reference':'interpolated raw stereo observations; not independent ground truth'}
