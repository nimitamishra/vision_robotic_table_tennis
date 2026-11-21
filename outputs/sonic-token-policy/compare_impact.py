"""Evaluate unchanged earlier policies with their original token scale."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from decoder_env import ROOT
from impact_env import ImpactEnv, sample_cases
from train_impact import initialize, evaluate, compact

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run', type=Path); a = ap.parse_args()
    torch.set_num_threads(1); report = json.loads((a.run/'report.json').read_text())
    cases = sample_cases(np.random.default_rng(881337), 120)
    comparisons = [('test_previous_reaching_policy', ROOT/'runs/reach-v2/policy.pt'),
                   ('test_previous_contact_policy', ROOT/'runs/impact-v2/policy.pt')]
    for key, path in comparisons:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        env = ImpactEnv(residual_scale=checkpoint['residual_scale'])
        policy = initialize(path).eval()
        report[key], _ = evaluate(env, policy, cases)
        print(key, json.dumps(compact(report[key])), flush=True)
    report['test_initial_policy_note'] = 'The adapted starting actor at the new action scale; unchanged earlier actors are evaluated separately with their original scales.'
    (a.run/'report.json').write_text(json.dumps(report, indent=2))

if __name__ == '__main__':main()
