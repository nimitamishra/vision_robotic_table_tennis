"""Re-evaluate successful training strokes on validation feeds only."""
import argparse
from train_return import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run
    rows=json.loads((out/'training_trials.json').read_text());candidates=[r for r in rows if r['legal_return']]
    env=ReturnEnv(out/'forward_reference.npz');cases=sample_cases(np.random.default_rng(811),8)
    best=None;score=-float('inf');seen=set()
    for i,row in enumerate(candidates):
        key=tuple(np.round(row['params'],8))
        if key in seen:continue
        seen.add(key);report=evaluate(env,np.array(row['params']),cases);s=select_score(report)
        print(i,report['summary'],flush=True)
        if s>score:
            score=s;best={'params':row['params'],'validation':report,'selection':'successful training candidates, validation RNG 811 only'}
            (out/'selected_candidate.json').write_text(json.dumps(best,indent=2))
    print('BEST',json.dumps(best['validation']['summary']) if best else None,flush=True)
if __name__=='__main__':main()
