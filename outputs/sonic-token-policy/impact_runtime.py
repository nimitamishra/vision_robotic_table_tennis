"""Validated local simulation loading; no hardware publisher."""
import numpy as np
import torch
from decoder_env import decoder_hash
from impact_env import config_hash
from train_impact import ImpactPolicy, predict

def load_policy(path, env):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('simulation_only') is not True or checkpoint.get('hardware_approved') is not False:
        raise ValueError('Expected a simulation-only checkpoint')
    if checkpoint.get('decoder_sha256') != decoder_hash():raise ValueError('Decoder mismatch')
    if checkpoint.get('environment_sha256') != config_hash():raise ValueError('Impact environment mismatch')
    if not np.allclose(checkpoint['base_token'], env.base.base_token, atol=1e-7):raise ValueError('Seed mismatch')
    if checkpoint['residual_scale'] != env.base.residual_scale:raise ValueError('Token scale mismatch')
    policy = ImpactPolicy(); policy.load_state_dict(checkpoint['state_dict'], strict=True); policy.eval()
    return policy, checkpoint
