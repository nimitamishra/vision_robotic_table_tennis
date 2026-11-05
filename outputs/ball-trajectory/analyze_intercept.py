"""Offline causal comparison against frozen camera crossings; no robot connections."""
import argparse,json,collections
from pathlib import Path
import numpy as np
from tracker import Tracker,measurement,G
from collision import forecast
from intercept import predicted_crossing

def evaluate(results,trials,plane):
    rows=[]
    for trial in trials:
        if trial['status']!='observed_crossing':continue
        t=trial['reference_time_s'];reference=np.array(trial['reference_position_m'])
        for lead in [.1,.2,.3]:
            candidates=[]
            for r in results:
                if t-lead-.05<=r.get('timestamp_s',-1)<=t-lead:
                    pred=predicted_crossing(r,plane)
                    if pred:candidates.append((r,pred))
            row={'trial':trial['trial'],'lead_s':lead,'status':'missing'}
            if candidates:
                r,p=max(candidates,key=lambda q:q[0]['timestamp_s']);delta=np.array(p['position_m'])-reference
                row.update(status='evaluated',position_error_cm=float(np.linalg.norm(delta)*100),dz_cm=float(delta[2]*100),
                           time_error_ms=(r['timestamp_s']+p['time_s']-t)*1000,source=r.get('source'),event=r.get('event'))
            rows.append(row)
    return rows

def summarize(rows):
    summary={}
    for group,select in [('all',lambda t:True),('development_first10',lambda t:t<=10),('validation_last10',lambda t:t>10)]:
        summary[group]={}
        for lead in [.1,.2,.3]:
            r=[r for r in rows if r['lead_s']==lead and select(r['trial']) and r['status']=='evaluated']
            summary[group][str(lead)]={'evaluated':len(r),**{k:float(np.mean([abs(x[k]) if k=='time_error_ms' else x[k] for x in r])) if r else None for k in ['position_error_cm','dz_cm','time_error_ms']}}
    return summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('session',type=Path);ap.add_argument('test',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    packets=[json.loads(x) for x in (a.session/'measurements.jsonl').read_text().splitlines()]
    original=[json.loads(x) for x in (a.session/'trajectory.jsonl').read_text().splitlines()]
    trials=[json.loads(x) for x in (a.test/'trials.jsonl').read_text().splitlines()]
    cfg=json.loads((a.test/'config.json').read_text());spec=json.loads((a.session/'scene-calibration.json').read_text())['world']['table']
    allresults={'recorded':original}
    for name,e,ret in [('post_impact_restart',.8,.9),('restart_and_session_fit',.9,.6)]:
        tracker=Tracker();results=[]
        for packet in packets:
            pos,source=measurement(packet)
            r=tracker.step(packet['capture_monotonic_ns']*1e-9,pos,source)
            if r.get('velocity_m_s') is not None and tracker.mode!='stationary':
                r.update(forecast(r['position_m'],r['velocity_m_s'],spec,restitution=e,tangent_retention=ret))
            results.append(r)
        allresults[name]=results
    for window in [.15,.2,.25]:
      for accel in ['gravity','fitted']:
        history=collections.deque();results=[]
        for packet in packets:
            t=packet['capture_monotonic_ns']*1e-9;p,source=measurement(packet)
            if p is None:continue
            if history and t-history[-1][0]>.1:history.clear()
            history.append((t,p))
            while history and t-history[0][0]>window:history.popleft()
            if len(history)<5 or t-history[0][0]<.08:continue
            dt=np.array([x[0]-t for x in history]);points=np.array([x[1] for x in history]);A=np.column_stack([dt*0+1,dt])
            if accel=='gravity':
                coeff=np.linalg.lstsq(A,points-.5*dt[:,None]**2*G,rcond=None)[0];pos,vel=coeff;acc=G
            else:
                A=np.column_stack([A,.5*dt**2]);coeff=np.linalg.lstsq(A,points,rcond=None)[0];pos,vel,acc=coeff
            if np.linalg.norm(vel)>25:continue
            r={'timestamp_s':t,'source':source}
            if accel=='gravity':r.update(forecast(pos,vel,spec))
            else:
                # Diagnostic only: short-window acceleration is noisy; no bounce extrapolation.
                times=np.linspace(0,.5,51);pts=pos+times[:,None]*vel+.5*times[:,None]**2*acc
                below=np.flatnonzero(pts[:,2]<.02)
                if len(below):pts=pts[:below[0]];times=times[:below[0]]
                r.update(forecast_path=pts.tolist(),forecast_duration_s=float(times[-1]) if len(times) else 0)
            results.append(r)
        allresults[f'window_{window}_{accel}']=results
    report={}
    for name,results in allresults.items():
        rows=evaluate(results,trials,cfg['plane_y_m']);report[name]={'summary':summarize(rows),'rows':rows}
    a.output.write_text(json.dumps(report,indent=2))
    for name,r in report.items():print(name,json.dumps(r['summary']))
if __name__=='__main__':main()
