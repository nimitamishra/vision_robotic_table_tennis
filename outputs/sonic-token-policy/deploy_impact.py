"""Promote an evaluated contact policy only to the local simulation."""
import argparse
import hashlib
import json
import torch
from pathlib import Path
from decoder_env import ROOT
from impact_env import ImpactEnv, config_hash
from impact_runtime import load_policy

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run', type=Path); a = ap.parse_args()
    run = a.run.resolve(); report = json.loads((run/'report.json').read_text())
    audit = json.loads((run/'sensitivity.json').read_text())
    checkpoint_hash = hashlib.sha256((run/'policy.pt').read_bytes()).hexdigest()
    if audit.get('checkpoint_sha256') != checkpoint_hash:
        raise SystemExit('Contact audit does not match this checkpoint')
    if any(audit[k]['unsafe_episodes'] or audit[k]['racket_body_contact_episodes']
           for k in ('nominal', 'physics_1ms', 'observations_delayed_20ms', 'alternate_initial_state')):
        raise SystemExit('Contact audit failed; do not promote to the application')
    result = report['test_policy']
    if (report['environment_sha256'] != config_hash() or result['unsafe_episodes']
            or result['return_fraction'] <= report['test_constant_token']['return_fraction']
            or result['return_fraction'] <= report['test_initial_policy']['return_fraction']):
        raise SystemExit('Simulation promotion criteria failed; keep previous deployment')
    for key in ('test_previous_reaching_policy', 'test_previous_contact_policy'):
        if key in report and result['return_fraction'] <= report[key]['return_fraction']:
            raise SystemExit('Policy did not improve on the unchanged earlier checkpoint')
    checkpoint = torch.load(run/'policy.pt', map_location='cpu', weights_only=True)
    load_policy(run/'policy.pt', ImpactEnv(residual_scale=checkpoint['residual_scale']))
    manifest = {'checkpoint': str(run/'policy.pt'), 'report': str(run/'report.json'),
        'checkpoint_sha256': checkpoint_hash,
        'environment_sha256': config_hash(), 'return_fraction': result['return_fraction'],
        'target': 'local MuJoCo contact simulation only', 'hardware_approved': False}
    tmp = ROOT/'impact-deployment.tmp'; tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(ROOT/'impact-deployment.json'); print(json.dumps(manifest, indent=2))

if __name__ == '__main__':main()
