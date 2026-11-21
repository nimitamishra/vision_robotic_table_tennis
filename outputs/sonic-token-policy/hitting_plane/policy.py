"""Simulation-only four-input neural SONIC token policy. No reference/encoder at inference."""
from pathlib import Path
import sys,json
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'forehand'))
from train_prior import load
from common import sha
DEFAULT_CHECKPOINT=HERE.parent/'forehand/runs/forward-v1/policy.pt'
class HittingPlanePolicy:
 frame='pelvis_at_stroke_start'
 plane_x_m=.43
 def __init__(self,checkpoint=DEFAULT_CHECKPOINT):
  torch.set_num_threads(1);self.checkpoint=Path(checkpoint);self.predictor=load(self.checkpoint)
  self.checkpoint_sha256=sha(self.checkpoint)
 def predict(self,target_xyz_m,time_to_hit_s):
  target=np.asarray(target_xyz_m,dtype=float)
  if target.shape!=(3,):raise ValueError('Expected target XYZ in meters relative to pelvis at stroke start')
  features=np.r_[target,float(time_to_hit_s)]
  if features.shape!=(4,) or not np.isfinite(features).all():raise ValueError('Expected finite XYZ and seconds')
  token=np.asarray(self.predictor(features),dtype=np.float32)
  if token.shape!=(64,) or not np.isfinite(token).all():raise ValueError('Invalid SONIC token')
  return token
 def predict_uv(self,y_m,z_m,time_to_hit_s):
  return self.predict([self.plane_x_m,y_m,z_m],time_to_hit_s)
