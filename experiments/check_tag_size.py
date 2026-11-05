import json,msgpack,numpy as np
from pathlib import Path
out=Path('outputs/g1-ball-session');cfg=json.loads((out/'session-calibration.json').read_text());old=cfg['world']['tag_size_m'];ratio=.0875/old
labels=['static_height_sweep_20','static_height_sweep_40','static_height_sweep_60','static_height_sweep_80','static_height_sweep_20_return'];rows={k:[] for k in labels}
for name in ('frames-20261016-103533.msgpack','frames-20261016-110554.msgpack'):
 with (out/name).open('rb') as f:
  for p in msgpack.Unpacker(f,raw=False,max_buffer_size=10000000):
   label=p.get('measurement_label')
   if label not in rows or not p.get('ball_refined') or p.get('T_world_left') is None:continue
   T=np.array(p['T_world_left']);xyz=np.array(p['ball_refined']['xyz_left']);before=T[:3,:3]@xyz+T[:3,3];after=T[:3,:3]@xyz+ratio*T[:3,3]
   rows[label].append([before,after])
result={'old_tag_size_m':old,'new_tag_size_m':.0875,'ratio':ratio,'method':'Tag5 PnP translation scales linearly with tag size; rotation and stereo ball coordinates unchanged. World tag origin is zero.','points':{}}
for k,v in rows.items():
 if v:
  a=np.array(v);result['points'][k]={'n':len(a),'before_cm':(np.median(a[:,0],axis=0)*100).tolist(),'after_cm':(np.median(a[:,1],axis=0)*100).tolist()}
(out/'tag-size-replay.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
