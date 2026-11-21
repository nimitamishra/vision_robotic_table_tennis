"""Checkpoint adapter for the offline simulator, deliberately no robot transport."""
import numpy as np
import torch
from decoder_env import decoder_hash
from train import TokenPolicy

def load_policy(path,env):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('simulation_only') is not True or checkpoint.get('hardware_approved') is not False:
        raise ValueError('Expected an explicitly simulation-only checkpoint')
    if checkpoint.get('decoder_sha256') != decoder_hash():raise ValueError('SONIC decoder changed; revalidate the model')
    if not np.allclose(checkpoint.get('base_token'),env.base_token,atol=1e-7):raise ValueError('Nominal token / seed mismatch')
    if checkpoint.get('residual_scale') != env.residual_scale:raise ValueError('Token scaling mismatch')
    policy=TokenPolicy();policy.load_state_dict(checkpoint['state_dict'],strict=True);policy.eval()
    return policy,checkpoint

def predict(policy,obs):
    obs=np.asarray(obs,dtype=np.float32)
    if obs.shape!=(131,) or not np.isfinite(obs).all():raise ValueError('Invalid policy observation')
    with torch.no_grad():action=policy(torch.from_numpy(obs))[0].numpy()
    if action.shape!=(64,) or not np.isfinite(action).all():raise ValueError('Invalid token residual')
    return np.clip(action,-1,1)
