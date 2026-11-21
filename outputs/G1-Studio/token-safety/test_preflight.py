import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from swing_preflight import check_pose
import numpy as np
z=np.zeros(29)
assert check_pose(z,z,z)['start_pose_matches']
q=z.copy();q[25]=1.6
r=check_pose(q,z,z)
assert not r['start_pose_matches'] and r['worst_joint_index']==25
q=z.copy();q[0]=.2
assert not check_pose(q,z,z)['start_pose_matches'] # lower body also matters
assert not check_pose(z,np.ones(29),z)['start_pose_matches']
for x in ([0]*7,[float('nan')]*29):
    try:check_pose(x,z,z)
    except ValueError:pass
    else:raise AssertionError('Bad state accepted')
print('PASS: start pose, lower-body mismatch, motion, malformed state')
