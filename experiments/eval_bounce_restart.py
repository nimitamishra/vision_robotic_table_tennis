exec(open('work/fit_bounce.py').read().split("fits=[]")[0])
from tracker import Tracker
tracker=Tracker();results=[]
for q in packets:
 pos,source=measurement(q);r=tracker.step(q['capture_monotonic_ns']*1e-9,pos,source)
 if r.get('velocity_m_s') is not None and tracker.mode!='stationary':r.update(forecast(r['position_m'],r['velocity_m_s'],spec))
 results.append(r)
rows=evaluate(results,trials,plane)
print(json.dumps(summarize(rows),indent=2))
Path('outputs/ball-trajectory/analysis/bounce-restart.json').write_text(json.dumps({'summary':summarize(rows),'rows':rows},indent=2))
