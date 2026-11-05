import csv,json,glob
from pathlib import Path
from collections import Counter
import numpy as np
out=Path('outputs/g1-ball-session');p=out/'control.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps({'label':'setup','truth_m':None}));tmp.replace(p)
rows=[r for f in glob.glob(str(out/'measurements-*.csv')) for r in csv.DictReader(open(f)) if r['label']=='static_point_02']
good=[r for r in rows if r['status']=='valid']
xyz=np.array([[float(r[k]) for k in ('x_m','y_m','z_m')] for r in good])
summary={'label':'static_point_02','samples':len(rows),'statuses':dict(Counter(r['status'] for r in rows)),'duration_s':float(rows[-1]['unix_time'])-float(rows[0]['unix_time']),'valid_fraction':len(good)/len(rows),'median_xyz_m':np.median(xyz,axis=0).tolist(),'std_xyz_mm':(np.std(xyz,axis=0)*1000).tolist(),'p05_xyz_m':np.percentile(xyz,5,axis=0).tolist(),'p95_xyz_m':np.percentile(xyz,95,axis=0).tolist(),'local_pipeline_ms_p50_p95':np.percentile([float(r['local_pipeline_ms']) for r in rows],[50,95]).tolist(),'reprojection_px_p50_p95':np.percentile([float(r['reprojection_error_px']) for r in good],[50,95]).tolist(),'truth':'not yet measured; stability only'}
(out/'static-point-02-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
