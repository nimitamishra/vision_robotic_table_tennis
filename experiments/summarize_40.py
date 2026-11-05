import csv,glob,json,time
from pathlib import Path
import numpy as np
out=Path('outputs/g1-ball-session');p=out/'control.json';t=p.with_suffix('.tmp');t.write_text(json.dumps({'label':'setup','truth_m':None}));t.replace(p)
ref=json.loads((out/'static-center-40cm-reference.json').read_text())
rows=[r for f in glob.glob(str(out/'measurements-*.csv')) for r in csv.DictReader(open(f)) if r['label']=='static_center_40cm' and ref['start_unix']<=float(r['unix_time'])<=ref['start_unix']+12]
result={'reference':ref,'samples':len(rows),'window':'first 12 seconds after reference start; excludes later interruption time'}
if rows:
 result['duration_s']=float(rows[-1]['unix_time'])-float(rows[0]['unix_time'])
 for name,prefix,status_key in [('box','','status'),('contour','refined_','refined_status')]:
  valid=[r for r in rows if r.get(status_key)=='valid']
  a=np.array([[float(r[prefix+k+'_m']) for k in 'xyz'] for r in valid])
  result[name]={'valid':len(valid),'valid_fraction':len(valid)/len(rows)}
  if len(a):
   med=np.median(a,axis=0);result[name].update(median_cm=(med*100).tolist(),std_mm=(a.std(0)*1000).tolist(),p05_cm=(np.percentile(a,5,axis=0)*100).tolist(),p95_cm=(np.percentile(a,95,axis=0)*100).tolist(),xy_error_cm=(100*(med[:2]-[0,.4])).tolist())
(out/'static-center-40cm-summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
