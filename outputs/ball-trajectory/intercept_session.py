"""Timed trials count even entirely missed throws, without guessing from detections."""
import json,time
from pathlib import Path
from intercept import Trial,predicted_crossing

class Session:
    def __init__(self,output):
        self.output=Path(output);self.active=False;self.trial=None;self.results=[];self.phase='idle';self.current_prediction=None
    def start(self,now,plane_y,width,table_z=0.,count=20):
        if self.active:return
        self.folder=self.output/('intercept-'+time.strftime('%Y%m%d-%H%M%S')+'-'+str(time.time_ns()%1000000))
        self.folder.mkdir(parents=True);self.file=(self.folder/'trials.jsonl').open('w')
        self.plane_y=plane_y;self.width=width;self.table_z=table_z;self.total=count;self.results=[];self.active=True;self.number=1
        self.phase='countdown';self.deadline=now+3;self.trial=None;self.current_prediction=None
        (self.folder/'config.json').write_text(json.dumps({'plane_y_m':plane_y,'width_m':width,'table_z_m':table_z,'trials':count,'countdown_s':3,'throw_window_s':5,'lead_tolerance_s':.05},indent=2))
    def save(self,record):
        self.results.append(record);self.file.write(json.dumps(record)+'\n');self.file.flush()
        summary={'attempts_completed':len(self.results),'planned_attempts':self.total,'statuses':{},'reference':'camera consistency, not independent accuracy'}
        for r in self.results:summary['statuses'][r['status']]=summary['statuses'].get(r['status'],0)+1
        summary['lead_metrics']={}
        for lead in [.1,.2,.3]:
            m=[m for r in self.results if r['status']=='observed_crossing' for m in r['lead_metrics'] if m['requested_lead_s']==lead and m['status']=='evaluated']
            summary['lead_metrics'][str(lead)]={'evaluated':len(m),'attempts_completed':len(self.results),
                'mean_position_error_m':sum(x['position_error_m'] for x in m)/len(m) if m else None,
                'mean_absolute_time_error_s':sum(abs(x['time_error_s']) for x in m)/len(m) if m else None}
        tmp=self.folder/'summary.tmp';tmp.write_text(json.dumps(summary,indent=2));tmp.replace(self.folder/'summary.json')
    def tick(self,now):
        if not self.active:return
        if now<self.deadline:return
        if self.phase=='countdown':
            self.phase='throw';self.deadline=now+5;self.trial=Trial(self.number,self.plane_y,self.width,self.table_z)
        else:
            if self.trial is not None and not self.trial.done:self.save(self.trial.finish())
            self.current_prediction=None
            if self.number>=self.total:self.stop();self.phase='complete'
            else:self.number+=1;self.phase='countdown';self.deadline=now+3;self.trial=None
    def observe(self,now,t,p,result):
        self.tick(now)
        if not self.active or self.phase!='throw' or self.trial.done:return
        self.current_prediction=predicted_crossing(result,self.plane_y)
        record=self.trial.observe(t,p,result)
        if record:self.save(record);self.current_prediction=None
    def stop(self):
        if self.active:
            if self.trial is not None and not self.trial.done:self.save(self.trial.finish('cancelled'))
            self.file.close()
        self.active=False;self.phase='stopped';self.current_prediction=None
    def label(self,now):
        if not self.active:return f'Intercept Test: {self.phase}. Completed: {len(self.results)} trials.'
        action='GET READY' if self.phase=='countdown' else 'RECORDED — wait' if self.trial.done else 'THROW ONE BALL'
        return f'**{action}** · Trial {self.number}/{self.total} · {max(0,self.deadline-now):.1f}s\n\nOne throw per THROW window. Missed throws count as failures.\n\nCompleted: {len(self.results)}'
