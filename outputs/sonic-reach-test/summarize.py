"""Summarize actual simulated tracking, never infer success from target motion."""
import json,argparse
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
rows=[json.loads(x) for x in (a.run/'samples.jsonl').read_text().splitlines()]
reach=[r for r in rows if r['phase']=='reach'];complete=[r for r in rows if r['phase']=='complete']
report={'simulation_only':True,'requested_displacement_m':rows[-1].get('requested_displacement_m',[.05,0,0]),'ramp_duration_s':2,'reference':'SONIC offset hand point relative to pelvis; not racket center','completed':bool(complete),'virtual_support':any(r.get('virtual_support',False) for r in rows if r['phase'] in ['hold','reach','complete'])}
hold=[r for r in rows if r['phase']=='hold']
if hold:
 tail=[r for r in hold if r['time']>=hold[-1]['time']-5]
 report['final_hold_mean_errors_cm']=dict(zip(['left','right','head'],np.mean([np.linalg.norm(np.array(r['actual_local_m'])-r['target_local_m'],axis=1)*100 for r in tail],axis=0).tolist()))
 report['hold_ready_observed']=any(r.get('ready_for_reach',False) for r in hold)
 report['position_feedback']=hold[-1].get('position_feedback',False)
 report['hold_min_root_height_m']=min(r['root_height_m'] for r in hold)
if (a.run/'finished.json').exists():
 report['exit_phase']=json.loads((a.run/'finished.json').read_text())['phase']
 report['completed']=report['completed'] or report['exit_phase']=='complete'
if reach:
 t0=reach[0]['time'];start=np.array(reach[0]['actual_local_m'][1]);steady=[r for r in reach+complete if 6<=r.get('trial_elapsed_s',r['time']-t0)<=7]
 report['starting_error_cm']=reach[0].get('right_error_cm')
 if steady:
  report['mean_final_error_cm']=float(np.mean([r['right_error_cm'] for r in steady]));report['max_final_error_cm']=max(r['right_error_cm'] for r in steady)
  report['measured_displacement_m']=(np.mean([r['actual_local_m'][1] for r in steady],axis=0)-start).tolist()
 report['min_root_height_m']=min(r['root_height_m'] for r in reach)
 report['max_root_height_m']=max(r['root_height_m'] for r in reach)
 if all('feet_in_contact' in r for r in reach):report['both_feet_contact_fraction']=float(np.mean([r['feet_in_contact']==2 for r in reach]))
 if all('correction_m' in r for r in reach):report['right_correction_at_limit_fraction']=float(np.mean([np.linalg.norm(r['correction_m'][1])>=.0399 for r in reach]))
 report['final_window_samples']=len(steady)
 report['final_window_span_s']=steady[-1]['time']-steady[0]['time'] if steady else 0.
 report['within_2cm_for_final_second']=report['completed'] and len(steady)>=15 and report['final_window_span_s']>=.8 and all(r['right_error_cm']<=2 for r in steady)
else:report['reason']='No reach phase was recorded.'
(a.run/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
