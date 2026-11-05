import sys,json,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path('outputs/ball-trajectory').resolve()))
from tracker import Tracker,measurement
spec=importlib.util.spec_from_file_location('old_tracker','work/tracker-before-forecast-display.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
a=old.Tracker();b=Tracker();ca=cb=n=0;changed=0
for line in Path('outputs/ball-trajectory/runs/20261022-183002/measurements.jsonl').open():
 d=json.loads(line);p,s=measurement(d);t=d['capture_monotonic_ns']*1e-9
 x=a.step(t,p,s);y=b.step(t,p,s);n+=1;ca+=bool(x['predictions']);cb+=bool(y.get('forecast_path'))
 if x.get('velocity_m_s')!=y.get('velocity_m_s'):changed+=1
assert changed==0
print(json.dumps({'messages':n,'old_visible_forecast_frames':ca,'new_visible_forecast_frames':cb,'changed_velocity_estimates':changed,'note':'Display coverage only; not an accuracy or detection recall measurement.'},indent=2))
# A descending ball with <50ms until contact must still have a curve to the plane.
b.x=__import__('numpy').array([0.,0.,.04,0.,0.,-1.]);b.P=__import__('numpy').eye(6)*.001;b.mode='tracking';b.last_seen=0
r=b.result(0);assert 0<r['forecast_duration_s']<.05 and len(r['forecast_path'])>=2
assert abs(r['forecast_path'][-1][2]-.02)<1e-8
print('PASS: final sub-50ms flight segment ends exactly at contact height.')
