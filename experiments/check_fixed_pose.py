import json,msgpack,numpy as np
from pathlib import Path
out=Path('outputs/g1-ball-session');labels=['static_height_sweep_20','static_height_sweep_60','static_height_sweep_80','static_height_sweep_20_return'];packets={k:[] for k in labels}
for name in ('frames-20261016-103533.msgpack','frames-20261016-110554.msgpack'):
 with (out/name).open('rb') as f:
  for p in msgpack.Unpacker(f,raw=False,max_buffer_size=10000000):
   if p.get('measurement_label') in packets and p.get('ball_refined') and p.get('T_world_left'):
    packets[p['measurement_label']].append(p)
T=np.array(packets[labels[0]][len(packets[labels[0]])//2]['T_world_left']);results={}
for label,ps in packets.items():
 zs=[];fixed=[];required=[];angles=[]
 for p in ps:
  point=np.array(p['ball_refined']['xyz_left']);tw=np.array(p['T_world_left']);cal=p['calibration'];tlr=np.array(cal['T_left_right']);intr=cal['right_intrinsics']
  zs.append((tw@np.r_[point,1])[2]);fixed.append((T@np.r_[point,1])[2]);ray=point/point[2]
  denom=tw[2,:3]@ray
  depth=(.275-tw[2,3])/denom
  target=ray*depth;right=tlr[:3,:3].T@(target-tlr[:3,3]);u=intr['fx']*right[0]/right[2]+intr['cx'];required.append(u-p['ball_refined']['right_center_px'][0])
  r=tw[:3,:3]@T[:3,:3].T;angles.append(np.degrees(np.arccos(np.clip((np.trace(r)-1)/2,-1,1))))
 results[label]={'n':len(ps),'median_z_cm':float(np.median(zs)*100),'fixed_pose_median_z_cm':float(np.median(fixed)*100),'median_right_u_correction_px_for_27_5cm_assuming_left_ray_and_pose_correct':float(np.median(required)),'pose_angle_difference_deg':float(np.median(angles))}
(out/'fixed-pose-diagnosis.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
