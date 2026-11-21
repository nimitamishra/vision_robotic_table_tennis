"""Promote an evaluated checkpoint to the local simulation application only."""
import argparse,json,hashlib
from pathlib import Path
from decoder_env import ROOT,TokenReachEnv
from runtime import load_policy

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();run=a.run.resolve()
    report=json.loads((run/'report.json').read_text())
    policy=report['test_policy'];baseline=report['test_constant_token']
    if (policy['unsafe_episodes'] or policy['mean_error_cm']>=1
            or policy['mean_error_cm']>=baseline['mean_error_cm']):
        raise SystemExit('Simulation promotion criteria failed')
    load_policy(run/'policy.pt',TokenReachEnv())
    manifest={'checkpoint':str(run/'policy.pt'),'report':str(run/'report.json'),
              'checkpoint_sha256':hashlib.sha256((run/'policy.pt').read_bytes()).hexdigest(),
              'target':'local MuJoCo simulation only','hardware_approved':False,
              'mean_error_cm':policy['mean_error_cm'],'best_update':report['best_update']}
    tmp=ROOT/'deployment.tmp';tmp.write_text(json.dumps(manifest,indent=2));tmp.replace(ROOT/'deployment.json')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
