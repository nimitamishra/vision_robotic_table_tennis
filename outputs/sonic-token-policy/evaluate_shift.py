"""Additional simulation check using a separately collected initial state."""
import argparse,json
import numpy as np
import torch
from decoder_env import *
from runtime import load_policy
from train import evaluate,sample_goals

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args()
    torch.set_num_threads(1);env=TokenReachEnv();policy,_=load_policy(a.run/'policy.pt',env)
    alternate=dict(np.load(ROOT/'runs/decoder-validation/seed.npz'))
    # Keep deployed nominal token fixed; change only robot state and history.
    for key in ('qpos','qvel','history'):env.seed[key]=alternate[key]
    goals=sample_goals(np.random.default_rng(28171),40)
    result,_=evaluate(env,policy,goals)
    baseline,_=evaluate(env,policy,goals,mode='zero')
    report={'initial_state':'Separate C++ SONIC standing simulation recording',
            'constant_deployed_token':True,'model':result,'baseline':baseline,
            'note':'Small initial-posture variation only; not a hardware or domain-randomization test.'}
    (a.run/'shifted_state_eval.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({name:{k:v for k,v in report[name].items() if k!='trials'} for name in ('model','baseline')}))
if __name__=='__main__':main()
