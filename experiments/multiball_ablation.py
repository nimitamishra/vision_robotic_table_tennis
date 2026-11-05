import sys,json,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path('outputs/ball-recorder/multiball-review').resolve()))
from tracker import MultiBallTracker
out=Path('outputs/ball-recorder/multiball-review');data=json.loads(Path('outputs/ball-recorder/ablation/candidates.json').read_text());results={}
for mode in ['stereo_only','motion','shape','trajectory','motion_shape','motion_shape_trajectory']:
 tr=MultiBallTracker();n=0;frames=0;multi=0;confirmed=0
 for f in data:
  cs=[]
  for c in f['cs']:
   if 'motion' in mode and c['motion']<5:continue
   if 'shape' in mode and c['shape']<.5:continue
   cs.append({k:c[k] for k in ['pix','q','confidence']}|{'box_pix':c['pix']})
  if 'trajectory' not in mode:tr.tracks=[]
  r=tr.update(cs,f['t']);n+=len(r);frames+=bool(r);multi+=len(r)>=2;confirmed+=sum(x['confirmed'] for x in r)
 results[mode]={'paired_detections':n,'frames_with_pairs':frames,'frames_with_multiple_pairs':multi,'confirmed_track_observations':confirmed if 'trajectory' in mode else None,'created_ids':tr.next_id-1 if 'trajectory' in mode else None}
(out/'ablation.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
# Validate two independent tracks through a short missed observation.
t=MultiBallTracker()
def c(x):return {'box_pix':[x,100,x-10,100],'pix':[x,100,x-10,100],'q':[0,0,1],'confidence':.9}
r=t.update([c(100),c(200)],0);ids=[x['id'] for x in r]
for k in range(1,4):r=t.update([c(100+k),c(200-k)],k/60)
assert [x['id'] for x in r]==ids and all(x['confirmed'] for x in r)
t.update([],4/60);r=t.update([c(105),c(195)],5/60);assert [x['id'] for x in r]==ids
print('PASS: independent IDs, confirmation and short-gap recovery')
