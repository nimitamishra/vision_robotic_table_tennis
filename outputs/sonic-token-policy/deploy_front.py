"""Select an evaluated experimental model for the local front-table viewer only."""
import argparse,hashlib,json
from pathlib import Path
from decoder_env import ROOT
from serve_front import load

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args()
    checkpoint=(a.run/'policy.pt').resolve();_,cp=load(checkpoint)
    report=json.loads((a.run/'report.json').read_text());audit=json.loads((a.run/'audit.json').read_text())
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if audit['checkpoint_sha256']!=digest:raise ValueError('Audit/checkpoint mismatch')
    if report['test_policy']['episodes']<100:raise ValueError('Insufficient independent evaluation')
    if report['test_policy']['unsafe_episodes'] or any(audit[k]['unsafe_episodes'] for k in ('nominal','half_timestep','full_lift_pipeline')):
        raise ValueError('Posture-limit failures: do not promote this checkpoint')
    lift=ROOT/'runs/front-lift-v1/policy.pt'
    manifest={'checkpoint':str(checkpoint),'checkpoint_sha256':digest,'lift_checkpoint':str(lift),
        'lift_checkpoint_sha256':hashlib.sha256(lift.read_bytes()).hexdigest(),
        'status':'experimental_front_table','simulation_only':True,'hardware_approved':False,
        'legal_returns':report['test_policy']['legal_returns'],'test_episodes':report['test_policy']['episodes'],
        'note':'Local visualization only; this is not approval for physical deployment or evidence of reliable rallies.'}
    (ROOT/'front-deployment.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
