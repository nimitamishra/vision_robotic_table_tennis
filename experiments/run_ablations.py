import json,cv2,numpy as np
from pathlib import Path
out=Path('outputs/ball-recorder/ablation');src=Path('outputs/ball-recorder/recordings/20261020-172639-275907');data=json.loads((out/'candidates.json').read_text());modes=['stereo_only','motion_only','shape_only','trajectory_only','motion_shape','motion_shape_trajectory'];states={m:{'q':None,'t':None,'v':np.zeros(3),'n':0,'switches':0,'run':0,'runs':[],'rows':[]} for m in modes};v=cv2.VideoWriter(str(out/'method-comparison.avi'),cv2.VideoWriter_fourcc(*'MJPG'),59.9108678,(1440,540));photos=[]
for idx,f in enumerate(data):
 base=cv2.imread(str(src/f['file']))[:,:480];tiles=[]
 for m in modes:
  s=states[m];cs=f['cs'];t=f['t'];selected=None
  if 'motion' in m:cs=[c for c in cs if c['motion']>=5]
  if 'shape' in m:cs=[c for c in cs if c['shape']>=.5]
  if 'trajectory' in m and s['q'] is not None and t-s['t']<=.15:
   dt=t-s['t'];pred=s['q']+s['v']*dt;valid=[c for c in cs if np.linalg.norm(np.array(c['q'])-pred)<=.06+4*dt and np.linalg.norm(np.array(c['q'])-s['q'])<=12*dt]
   if valid:selected=min(valid,key=lambda c:np.linalg.norm(np.array(c['q'])-pred))
  else:
   if cs:selected=max(cs,key=lambda c:c['confidence'])
  if selected:
   q=np.array(selected['q'])
   if s['q'] is not None and t-s['t']<.15:s['v']=.5*s['v']+.5*(q-s['q'])/max(.001,t-s['t'])
   else:s['v']=np.zeros(3);s['switches']+=1
   s['q']=q;s['t']=t;s['n']+=1;s['run']+=1
  elif s['run']:s['runs'].append(s['run']);s['run']=0
  s['rows'].append({'frame':f['frame_id'],'t':t,'xyz':selected['q'] if selected else None,'pixel':selected['pix'] if selected else None})
  im=base.copy()
  if selected:
   x,y=selected['pix'][:2];cv2.circle(im,(round(x),round(y)),9,(0,255,0),2)
  cv2.putText(im,f"{m} {t:.2f}s",(3,15),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1);tiles.append(im)
 grid=np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]);v.write(grid)
 if idx in [600,1200]:photos.append(grid)
v.release();summary={}
for m,s in states.items():
 if s['run']:s['runs'].append(s['run'])
 summary[m]={'output_frames':s['n'],'coverage_percent':round(100*s['n']/len(data),1),'acquisitions':s['switches'],'median_uninterrupted_frames':float(np.median(s['runs'])) if s['runs'] else 0}
 (out/(m+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in s['rows']))
(out/'metrics.json').write_text(json.dumps({'results':summary,'warning':'Coverage is NOT accuracy. No ground-truth ball identities; multiple balls may be present. Classifier not trained pending label clarification.','thresholds':{'motion_mean_abs_pixel_difference':5,'shape_product_of_box_aspect_ratios':.5,'max_speed_m_s':12,'max_gap_s':.15}},indent=2));cv2.imwrite(str(out/'comparison-examples.jpg'),np.vstack(photos));print(json.dumps(summary,indent=2))
