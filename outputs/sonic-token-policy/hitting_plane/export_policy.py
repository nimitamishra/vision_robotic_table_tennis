"""Package the neural prior and RL token head into ONE four-input TorchScript model."""
import sys,json,argparse,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from train_residual import ResidualPolicy
from policy import HittingPlanePolicy
from train_prior import Prior

class CombinedPolicy(nn.Module):
 def __init__(self,cp,basis,params):
  super().__init__();self.prior=Prior();self.prior.load_state_dict(cp['state_dict'])
  for k in ('x_mean','x_scale','y_mean','y_scale'):self.register_buffer(k,torch.tensor(cp[k],dtype=torch.float64))
  self.register_buffer('basis',torch.tensor(basis,dtype=torch.float64))
  self.register_buffer('params',torch.tensor(params,dtype=torch.float64))
 def forward(self,x:torch.Tensor)->torch.Tensor:
  torch._assert(x.dim()==2 and x.size(1)==4,'Expected batch of [X,Y,Z,time]')
  torch._assert(bool(torch.isfinite(x).all()),'Nonfinite input')
  torch._assert(bool((torch.abs(x[:,0]-.43)<=.002).all()),'X outside fixed plane')
  torch._assert(bool(((x[:,1]>=-.531)&(x[:,1]<=-.329)).all()),'Y outside training domain')
  torch._assert(bool(((x[:,2]>=.077)&(x[:,2]<=.229)).all()),'Z outside training domain')
  torch._assert(bool(((x[:,3]>=-1.501)&(x[:,3]<=1.001)).all()),'Time outside training domain')
  x=x.to(torch.float64);t=x[:,3]
  features=torch.cat((x[:,:3],torch.clamp(t-self.params[-1]*.18,-1.5,1.).unsqueeze(1)),dim=1)
  features=features.to(torch.float32).to(torch.float64)
  core=self.prior(((features-self.x_mean)/self.x_scale).to(torch.float32)).to(torch.float64)*self.y_scale+self.y_mean
  core=core.to(torch.float32).to(torch.float64)
  gate=torch.exp(-.5*((t-.20)/.40)**4).unsqueeze(1)
  return (core+gate*(self.params[:-1]@self.basis)).to(torch.float32)

class ExportedPolicy:
 def __init__(self,path):
  torch.set_num_threads(1);self.model=torch.jit.load(str(path));self.model.eval()
  self.checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
 def predict(self,target,t):
  features=np.r_[target,float(t)]
  with torch.no_grad():return self.model(torch.tensor(features[None],dtype=torch.float64))[0].numpy()

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();cfg=json.loads((a.run/'policy.json').read_text())
 cp=torch.load(cfg['base_checkpoint'],map_location='cpu',weights_only=True);basis=np.load(a.run/'basis.npy')
 model=torch.jit.script(CombinedPolicy(cp,basis,cfg['params']).eval());model.save(str(a.run/'policy.ts'))
 python=ResidualPolicy(HittingPlanePolicy(cfg['base_checkpoint']),basis,cfg['params']);exported=ExportedPolicy(a.run/'policy.ts')
 rng=np.random.default_rng(313);errors=[]
 for _ in range(200):
  goal=[.43,rng.uniform(-.53,-.33),rng.uniform(.078,.228)];t=rng.uniform(-1.5,1.)
  errors.append(float(np.max(abs(python.predict(goal,t)-exported.predict(goal,t)))))
 if max(errors)>1e-6:raise AssertionError(f'Export mismatch {max(errors)}')
 report={'input':['target_x_pelvis_at_start_m','target_y_pelvis_at_start_m','target_z_pelvis_at_start_m','time_before_hit_s'],
  'output':'64D SONIC token','artifact':'policy.ts','sha256':exported.checkpoint_sha256,'max_python_export_difference':max(errors),'checks':len(errors),
  'online_encoder_required':False,'architecture':'4-128-128-128-64 tanh prior + learned low-rank token head + bounded phase lead',
  'training':'Supervised motion-token prior; CEM reinforcement learning of 13 token-head parameters on direct-bracket dynamics',
  'simulation_only':True,'hardware_approved':False,'mount':json.loads((a.run/'mount.json').read_text()),
  'base_sha256':hashlib.sha256(Path(cfg['base_checkpoint']).read_bytes()).hexdigest(),
  'config_sha256':hashlib.sha256((a.run/'policy.json').read_bytes()).hexdigest(),'basis_sha256':hashlib.sha256((a.run/'basis.npy').read_bytes()).hexdigest()}
 (a.run/'model.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
