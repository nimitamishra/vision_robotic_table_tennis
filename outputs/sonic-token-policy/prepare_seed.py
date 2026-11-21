"""Build decoder histories from a recorded loopback-only SONIC run."""
import argparse,json
import numpy as np
from scipy.spatial.transform import Rotation
from decoder_env import *

def debug_frame(d):
    q=np.asarray(d['base_quat'])
    gravity=Rotation.from_quat(q[[1,2,3,0]]).inv().apply([0,0,-1])
    return np.r_[d['base_ang_vel'],(np.asarray(d['body_q'])-DEFAULT)[TO_ISAAC],
                 np.asarray(d['body_dq'])[TO_ISAAC],
                 ((np.asarray(d['last_action'])-DEFAULT)/SCALE)[TO_ISAAC],gravity]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args()
    tokens=[json.loads(x) for x in (a.run/'tokens.jsonl').read_text().splitlines()]
    samples=[json.loads(x) for x in (a.run/'samples.jsonl').read_text().splitlines()]
    i=max(i for i,x in enumerate(tokens) if x['phase']=='hold')
    if not all(x['simulation_only'] for x in tokens):raise ValueError('Not simulation data')
    history=np.array([debug_frame(x['debug']) for x in tokens[i-9:i+1]])
    sample=next(x for x in reversed(samples) if x['phase']=='hold')
    np.savez(a.run/'seed.npz',qpos=sample['full_qpos'],qvel=sample['full_qvel'],history=history,token=tokens[i]['token_state'])
    # Independently compare ONNX outputs to the next C++ debug action.
    decoder=Decoder();errors=[]
    for j in range(20,len(tokens)-1,10):
        window=tokens[j-9:j+2]
        if any(window[k+1]['controller_index']-window[k]['controller_index']!=1 for k in range(10)):continue
        obs=pack_observation(tokens[j]['token_state'],[debug_frame(x['debug']) for x in window[:10]])
        predicted=decoder(obs)
        actual=debug_frame(tokens[j+1]['debug'])[61:90]
        errors.append(float(np.max(np.abs((predicted-actual)[TO_MJ]*SCALE))))
    report={'samples':len(errors),'max_joint_target_difference_rad':max(errors) if errors else None,
            'median_joint_target_difference_rad':float(np.median(errors)) if errors else None,
            'note':'Debug-packet history reconstruction; controller time resampling may duplicate adjacent frames.'}
    (a.run/'decoder_parity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
