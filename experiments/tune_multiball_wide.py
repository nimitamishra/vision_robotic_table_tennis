import sys,json,itertools,numpy as np,time
from pathlib import Path
sys.path.insert(0,str(Path('outputs/ball-recorder/multiball-review').resolve()))
from tracker import MultiBallTracker
from tracker_initial import MultiBallTracker as Original
out=Path('outputs/ball-recorder/multiball-review');data=json.loads(Path('outputs/ball-recorder/ablation/candidates.json').read_text());refs=json.loads((out/'flow-correspondences.json').read_text());allcs=[[{'q':c['q'],'pix':c['pix'],'box_pix':c['pix'],'confidence':c['confidence']} for c in f['cs']] for f in data]
def evaluate(tr):
 rows=[];ids={};confirmed=0
 for f,cs in zip(data,allcs):
  r=tr.update(cs,f['t']);rows.append(r)
  for c in r:ids[c['id']]=ids.get(c['id'],0)+1;confirmed+=c['confirmed']
 def section(lo,hi):
  res={'checks':0,'same_id':0,'broken':0,'conflicting_link':0,'not_paired':0}
  for ref in refs:
   if not lo<=data[ref['b']]['t']<hi:continue
   a=next((r for r in rows[ref['a']] if r['box_pix']==ref['pix_a']),None);b=next((r for r in rows[ref['b']] if r['box_pix']==ref['pix_b']),None)
   if a is None or b is None:res['not_paired']+=1;continue
   res['checks']+=1
   if a['id']==b['id']:res['same_id']+=1
   elif any(r['id']==a['id'] for r in rows[ref['b']]):res['conflicting_link']+=1
   else:res['broken']+=1
  return res
 return {'created_ids':tr.next_id-1,'confirmed_observations':confirmed,'median_observations_per_id':float(np.median(list(ids.values()))),'train':section(0,20),'heldout':section(20,31)},rows
results=[];stats,_=evaluate(Original());results.append({'name':'original','params':{},**stats});print('original',stats,flush=True)
for init,gate,gap,weight in itertools.product([35,50],[25,35,50],[.25,.4],[.6,1.]):
 params={'initial_gate':init,'residual_gate':gate,'max_gap':gap,'velocity_weight':weight};stats,_=evaluate(MultiBallTracker(**params));results.append({'name':'candidate','params':params,**stats})
# Choose using first 20 seconds ONLY; score penalizes contradicted links more than broken links.
def score(r):
 t=r['train'];return (t['same_id']-4*t['conflicting_link']-t['not_paired'])/max(1,t['checks']+t['not_paired'])
best=max(results[1:],key=score);(out/'parameter-sweep-wide.json').write_text(json.dumps(results,indent=2));(out/'best-parameters-wide.json').write_text(json.dumps(best,indent=2));stats,rows=evaluate(MultiBallTracker(**best['params']));(out/'best-wide-box-tracks.jsonl').write_text('\n'.join(json.dumps({'t':f['t'],'tracks':r}) for f,r in zip(data,rows)));print('BEST',json.dumps(best),flush=True)
