import msgpack,numpy as np,json
p='outputs/g1-ball-session/frames-20261016-102335.msgpack'
rows=[]
with open(p,'rb') as f:
 for r in msgpack.Unpacker(f,raw=False,max_buffer_size=10000000):
  if r.get('measurement_label')=='static_point_02_repeat' and r.get('xyz_world') is not None:rows.append(r)
T=np.array([r['T_world_left'] for r in rows]);c=np.array([r['ball']['xyz_left'] for r in rows]);w=np.array([r['xyz_world'] for r in rows]);l=np.array([r['ball']['left_center_px'] for r in rows]);rr=np.array([r['ball']['right_center_px'] for r in rows]);ref=len(rows)//2
fixed_pose=np.array([(T[ref]@np.r_[p,1])[:3] for p in c]);fixed_ball=np.array([(t@np.r_[np.median(c,axis=0),1])[:3] for t in T])
out={'n':len(rows),'camera_point_std_mm':(c.std(0)*1000).tolist(),'world_point_std_mm':(w.std(0)*1000).tolist(),'world_std_if_pose_fixed_mm':(fixed_pose.std(0)*1000).tolist(),'world_std_if_ball_fixed_mm':(fixed_ball.std(0)*1000).tolist(),'left_pixel_p05_p50_p95':np.percentile(l,[5,50,95],axis=0).tolist(),'right_pixel_p05_p50_p95':np.percentile(rr,[5,50,95],axis=0).tolist(),'disparity_p05_p50_p95':np.percentile(l[:,0]-rr[:,0],[5,50,95]).tolist()}
for name,mask in [('near',w[:,1]<.52),('far',w[:,1]>=.52)]:
 out[name]={'n':int(mask.sum()),'left_center':np.median(l[mask],axis=0).tolist(),'right_center':np.median(rr[mask],axis=0).tolist(),'camera_xyz':np.median(c[mask],axis=0).tolist(),'world_xyz':np.median(w[mask],axis=0).tolist()}
print(json.dumps(out,indent=2));open('outputs/g1-ball-session/jump-analysis.json','w').write(json.dumps(out,indent=2))
