"""Recompute only collision forecasts from recorded states; preserve reference trials."""
import argparse,json
from pathlib import Path
from collision import forecast
from analyze_intercept import evaluate,summarize

def main():
    ap=argparse.ArgumentParser();ap.add_argument('session',type=Path);ap.add_argument('test',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    original=[json.loads(x) for x in (a.session/'trajectory.jsonl').read_text().splitlines()]
    trials=[json.loads(x) for x in (a.test/'trials.jsonl').read_text().splitlines()]
    spec=json.loads((a.session/'scene-calibration.json').read_text())['world']['table']
    cfg=json.loads((a.session/'config.json').read_text());plane=json.loads((a.test/'config.json').read_text())['plane_y_m']
    updated=[]
    for r in original:
        new=dict(r);settings=r.get('impact_settings',{'enabled':True,'restitution':cfg['restitution'],'retention':cfg['tangent_retention']})
        if r.get('collision_segments') is not None and settings['enabled']:
            new.update(forecast(r['position_m'],r['velocity_m_s'],spec,horizon=r.get('requested_horizon_s',cfg['horizon']),
                table_z=cfg['table_z'],restitution=settings['restitution'],tangent_retention=settings['retention']))
        updated.append(new)
    report={'reference':'Frozen camera observations; not independent ground truth. Same recorded filter states.', 'results':{}}
    for label,results in [('recorded',original),('contact_fix',updated)]:
        rows=evaluate(results,trials,plane);report['results'][label]={'summary':summarize(rows),'rows':rows}
    old={(r['trial'],r['lead_s']):r for r in report['results']['recorded']['rows']};changes=[]
    for row in report['results']['contact_fix']['rows']:
        before=old[(row['trial'],row['lead_s'])]
        if row!=before:changes.append({'before':before,'after':row})
    report['changed_evaluations']=changes
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v['summary']['all'] for k,v in report['results'].items()},indent=2))
    print('Changed evaluations:',len(changes))
    for c in changes:print(json.dumps(c))
if __name__=='__main__':main()
