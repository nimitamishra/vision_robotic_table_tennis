import json,numpy as np
from pathlib import Path
r=[json.loads(x) for x in Path('outputs/ball-recorder/multiball-review/tracks.jsonl').read_text().splitlines()];ages={};births=[];prev=[]
for f in r:
 for x in f['tracks']:
  ages[x['id']]=ages.get(x['id'],0)+1
  if ages[x['id']]==1 and prev:
   e=min(np.linalg.norm(np.array(x['box_pix'])-y['box_pix']) for y in prev);births.append(e)
 prev=f['tracks']
print('track length quantiles',np.percentile(list(ages.values()),[0,25,50,75,90,99,100]));print('new birth nearest previous',np.percentile(births,[0,25,50,75,90,99,100]));print('gaps candidate counts')
d=json.loads(Path('outputs/ball-recorder/ablation/candidates.json').read_text());print('dt quantiles',np.percentile(np.diff([f['t'] for f in d]),[0,50,99,100]));print('last frame',d[-1]['t'])
