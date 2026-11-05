import json,cv2,numpy as np
from pathlib import Path
o=Path('outputs/ball-recorder/multiball-review');src=Path('outputs/ball-recorder/recordings/20261020-172639-275907');data=json.loads(Path('outputs/ball-recorder/ablation/candidates.json').read_text());rs=[json.loads(l) for l in (o/'best-wide-box-tracks.jsonl').read_text().splitlines()];refs=json.loads((o/'flow-gap-correspondences.json').read_text());tiles=[]
for r in refs:
 a=next((x for x in rs[r['a']]['tracks'] if x['box_pix']==r['pix_a']),None);b=next((x for x in rs[r['b']]['tracks'] if x['box_pix']==r['pix_b']),None)
 if not a or not b or a['id']==b['id']:continue
 pair=[]
 for frame,c in [(r['a'],a),(r['b'],b)]:
  im=cv2.imread(str(src/data[frame]['file']))[:,:480];x,y=c['box_pix'][:2];cv2.circle(im,(round(x),round(y)),10,(0,0,255),2);cv2.putText(im,f"{data[frame]['t']:.2f}s ID {c['id']}",(4,16),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1);pair.append(im)
 tiles.append(np.hstack(pair))
if tiles:cv2.imwrite(str(o/'remaining-gap-examples.jpg'),np.vstack(tiles))
print('remaining gap cases',len(tiles))
