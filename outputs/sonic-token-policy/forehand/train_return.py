"""Episodic policy-search RL: maximize actual legal returns, through SONIC tokens.

CEM learns a bounded distribution over nine forehand motion parameters. It is
not PPO, and is deliberately a narrow-feed first-return curriculum. No learned
parameter is sent directly to a robot joint. The encoder turns the resulting
reference into tokens and the frozen decoder drives physical dynamics.
"""
import argparse,time
from return_env import *

def init_worker(ref):
    global worker_env
    worker_env=ReturnEnv(Path(ref))

def rollout_worker(job):
    params,cases=job
    return [worker_env.run(params,case) for case in cases]

def compact(rows):
    return {'episodes':len(rows),'face_contacts':sum(r['face_contact'] for r in rows),'net_crossings':sum(r['crossed_net'] for r in rows),
        'legal_returns':sum(r['legal_return'] for r in rows),'posture_failures':sum(r['failure']=='posture_limit' for r in rows),
        'racket_obstacle_failures':sum(r['failure']=='racket_obstacle' for r in rows),'mean_reward':float(np.mean([r['reward'] for r in rows]))}

def evaluate(env,params,cases,record_dir=None):
    rows=[]
    for i,case in enumerate(cases):
        r=env.run(params,case,record=record_dir is not None);rows.append(r)
        if record_dir:
            record_dir.mkdir(parents=True,exist_ok=True)
            (record_dir/f'episode_{i:03d}.json').write_text(json.dumps({'report':r,'events':env.events,'frames':env.frames}))
    return {'summary':compact(rows),'trials':rows}

def select_score(report):
    s=report['summary'];return 1000*s['legal_returns']+100*s['net_crossings']+5*s['face_contacts']-1000*s['posture_failures']-50*s['racket_obstacle_failures']+.01*s['mean_reward']

def main():
    p=argparse.ArgumentParser();p.add_argument('--iterations',type=int,default=45);p.add_argument('--population',type=int,default=32);p.add_argument('--output',type=Path,default=RETURN_DIR);p.add_argument('--resume',type=Path);p.add_argument('--fitness-cases',type=int,default=1);p.add_argument('--seed',type=int,default=771);p.add_argument('--local',action='store_true');p.add_argument('--workers',type=int,default=1);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'policy.json').exists():raise RuntimeError('Choose a new output directory')
    ref=out/'forward_reference.npz'
    if not ref.exists():corrected_reference(out)
    env=ReturnEnv(ref);rng=np.random.default_rng(a.seed)
    mean=np.zeros(9);std=np.full(9,.30)
    if a.resume:mean=np.array(json.loads(a.resume.read_text())['params']);std[:]=.12
    if a.local:std[:]=.05
    val=sample_cases(np.random.default_rng(811),8);baseline=evaluate(env,mean,val);best=mean.copy();best_report=baseline;best_score=select_score(baseline)
    print('BASELINE',json.dumps(baseline['summary']),flush=True);history=[];started=time.monotonic();all_rows=[]
    pool=None
    if a.workers>1:
        import multiprocessing
        pool=multiprocessing.get_context('spawn').Pool(a.workers,initializer=init_worker,initargs=(str(ref),))
    for iteration in range(a.iterations):
        population=np.clip(rng.normal(mean,std,size=(a.population,9)),-1,1);population[0]=mean;population[1]=best
        population[2:8]=np.clip(rng.normal(best,.04 if a.local else .22,size=(6,9)),-1,1)
        population[-3:]=np.clip(rng.normal(best,.15,size=(3,9)),-1,1) if a.local else rng.uniform(-1,1,size=(3,9))
        cases=sample_cases(rng,a.fitness_cases) if a.fitness_cases>1 or iteration>5 else [NOMINAL]
        scores=[];reports=[]
        batches=pool.map(rollout_worker,[(params,cases) for params in population]) if pool else [[env.run(params,case) for case in cases] for params in population]
        for trials in batches:
            reports.extend(trials);scores.append(float(np.mean([r['reward'] for r in trials])));all_rows.extend(trials)
        elite=population[np.argsort(scores)[-max(5,a.population//5):]]
        mean=.2*mean+.6*elite.mean(0)+.2*best;std=np.clip(.2*std+.8*elite.std(0),.015 if a.local else .07,.08 if a.local else .4)
        row={'iteration':iteration+1,'episodes':len(all_rows),'seconds':time.monotonic()-started,'population':compact(reports),'search_std_mean':float(std.mean())}
        if (iteration+1)%3==0 or iteration==a.iterations-1:
            candidates=[mean,population[np.argmax(scores)]]
            for candidate in candidates:
                validation=evaluate(env,candidate,val)
                if select_score(validation)>best_score:
                    best=candidate.copy();best_score=select_score(validation);best_report=validation
            row['best_validation']=best_report['summary']
            (out/'candidate.json').write_text(json.dumps({'params':best.tolist(),'validation':best_report,'iteration':iteration+1},indent=2))
        history.append(row);(out/'progress.json').write_text(json.dumps({'history':history,'mean':mean.tolist(),'std':std.tolist(),'best':best.tolist()}))
        (out/'training_trials.json').write_text(json.dumps(all_rows))
        print(json.dumps(row),flush=True)
    if pool:pool.close();pool.join()
    cp={'algorithm':'Cross-entropy episodic policy search','action':'9 bounded reference parameters -> SONIC encoder -> 64D token -> frozen decoder',
        'params':best.tolist(),'parameter_names':PARAM_NAMES,'parameter_scales':PARAM_SCALE.tolist(),'simulation_only':True,'hardware_approved':False,
        'reference_sha256':sha(ref),'decoder_sha256':sha(DECODER),'encoder_sha256':sha(DECODER.parent/'model_encoder.onnx'),'seed':a.seed,'fitness_cases':a.fitness_cases,'local_refinement':a.local,'resume_from':str(a.resume) if a.resume else None,
        'reward':{'face_contact':20,'net_crossing':80,'legal_landing':200,'ball_failure':-8,'posture_or_racket_obstacle':-80,'predicted_landing_shaping_max':20},'iterations':a.iterations,'training_episodes':len(all_rows)}
    (out/'policy.json').write_text(json.dumps(cp,indent=2))
    test=sample_cases(np.random.default_rng(9917),80)
    result=evaluate(env,best,test,out/'test_rollouts');zero=evaluate(env,np.zeros(9),test)
    wider=evaluate(env,best,sample_cases(np.random.default_rng(9918),40,wide=True),out/'wide_rollouts')
    report={'simulation_only':True,'hardware_approved':False,'training_episodes':len(all_rows),'validation':best_report,'test':result,'baseline_test':zero,'wider_test':wider,'history':history,
        'limitations':['Single forehand reference and narrow front-feed distribution; not general table tennis.',
        'Open-loop stroke-parameter policy chosen at serve start; no visual latency, spin or actuator randomization.',
        'Physics uses estimated racket mass/mount/contact parameters; not calibrated hardware.',
        'Warm-up starts from an initialized ready pose. No real-robot output.']}
    (out/'report.json').write_text(json.dumps(report,indent=2));(out/'training_trials.json').write_text(json.dumps(all_rows))
    print('FINAL',json.dumps({k:report[k]['summary'] for k in ('test','baseline_test','wider_test')}),flush=True)
if __name__=='__main__':main()
