import sys,json,numpy as np
from pathlib import Path
sys.path.insert(0,'outputs/ball-trajectory')
from tracker import measurement
from collision import forecast
from analyze_intercept import evaluate,summarize
p=Path('outputs/ball-trajectory/runs/20261030-155041');test=p/'intercept-20261030-155117-215234'
packets=[json.loads(x) for x in (p/'measurements.jsonl').read_text().splitlines()]
original=[json.loads(x) for x in (p/'trajectory.jsonl').read_text().splitlines()]
trials=[json.loads(x) for x in (test/'trials.jsonl').read_text().splitlines()]
spec=json.loads((p/'scene-calibration.json').read_text())['world']['table'];plane=.1735
obs=[(q['capture_monotonic_ns']*1e-9,measurement(q)[0]) for q in packets if measurement(q)[0] is not None]
fits=[]
for tr in trials:
 if 'reference_time_s' not in tr:continue
 tc=tr['reference_time_s'];near=[(t,p) for t,p in obs if tc-.35<t<tc+.12]
 if len(near)<8:continue
 tmin,minimum=min(near,key=lambda q:q[1][2])
 if minimum[2]>.09:continue
 data=[(t,p) for t,p in near if abs(t-tmin)<.14]
 best=None
 for tb in np.linspace(tmin-.035,tmin+.035,141):
  dt=np.array([t-tb for t,p in data]);pts=np.array([p for t,p in data]);pre=dt<0
  if sum(pre)<3 or sum(~pre)<3:continue
  A=np.column_stack([dt*pre,dt*(~pre)])
  vz=np.linalg.lstsq(A,pts[:,2]-.02+.5*9.81*dt**2,rcond=None)[0]
  residual=float(np.mean((A@vz-(pts[:,2]-.02+.5*9.81*dt**2))**2))
  if vz[0]>=-1 or vz[1]<=1:continue
  xy=np.linalg.lstsq(np.column_stack([dt*0+1,A]),pts[:,:2],rcond=None)[0]
  if best is None or residual<best[0]:best=(residual,tb,vz,xy)
 if best:
  error,tb,vz,xy=best
  rec=dict(trial=tr['trial'],restitution=float(-vz[1]/vz[0]),retention=float(xy[2,1]/xy[1,1]),z_fit_rmse_cm=error**.5*100,bounce_before_crossing_s=tc-tb)
  fits.append(rec)
print('fits',json.dumps(fits,indent=2))
dev=[q for q in fits if q['trial']<=10 and q['z_fit_rmse_cm']<2 and 0<q['retention']<1 and 0<q['restitution']<1.2]
e=round(float(np.median([q['restitution'] for q in dev])),2);ret=round(float(np.median([q['retention'] for q in dev])),2);print('dev medians',e,ret)
report={'fits':fits,'development_parameters':{'restitution':e,'retention':ret},'comparisons':{}}
for name,rest,tan in [('baseline',.8,.9),('dev_fitted',min(e,1),ret)]:
 results=[]
 for r in original:
  r=r.copy()
  if r.get('velocity_m_s') is not None and r.get('acceleration_source')!='stationary constraint (not measured)':
   r.update(forecast(r['position_m'],r['velocity_m_s'],spec,restitution=rest,tangent_retention=tan))
  results.append(r)
 rows=evaluate(results,trials,plane);report['comparisons'][name]={'summary':summarize(rows),'rows':rows}
 print(name,json.dumps(summarize(rows)))
Path('outputs/ball-trajectory/analysis/bounce-fit.json').write_text(json.dumps(report,indent=2))
