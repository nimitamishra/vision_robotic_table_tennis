import sys
import numpy as np
sys.path.insert(0,'/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/repo/g1')
from ball_geometry import best_stereo_ball, project_point
intr=dict(fx=420.,fy=420.,cx=240.,cy=135.,distortion=[.01,-.001,0,0,0])
T=np.eye(4);T[0,3]=.05
cal=dict(left_intrinsics=intr,right_intrinsics=intr,T_left_right=T.tolist())
def box(p): return [p[0]-4,p[1]-4,p[0]+4,p[1]+4,.95,0]
for xyz in ([.1,.1,.6],[-.2,.15,1.5],[.3,-.1,3.]):
 xyz=np.array(xyz);l=box(project_point(xyz,intr));r=box(project_point(xyz-T[:3,3],intr))
 result=best_stereo_ball([l],[r],cal)
 assert result and np.linalg.norm(np.array(result['xyz_left'])-xyz)<1e-6
 bad=r.copy();bad[1]+=15;bad[3]+=15
 assert best_stereo_ball([l],[bad],cal) is None
assert best_stereo_ball([],[],cal) is None
assert best_stereo_ball([box([240,135])],[box([240,135])],cal) is None
assert best_stereo_ball([box([230,135])],[box([250,135])],cal) is None
print('PASS: known 3D points with distortion; mismatched, missing, parallel and behind-camera cases')
