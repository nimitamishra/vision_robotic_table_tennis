"""Local-response warm start followed by PPO through the frozen SONIC decoder."""
import argparse,copy,json,time
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from decoder_env import *

class TokenPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(131,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,64))
        self.critic=nn.Sequential(nn.Linear(131,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh(),nn.Linear(64,1))
        self.log_std=nn.Parameter(torch.full((64,),-2.8))
    def forward(self,x):return self.actor(x),self.critic(x).squeeze(-1)

def sample_goals(rng,n):
    # A deliberately narrow initial curriculum: 1-3 cm from the settled hand.
    directions=rng.normal(size=(n,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
    return directions*rng.uniform(.01,.03,size=(n,1))

def teacher(goals,response):
    targets=np.concatenate([np.asarray(goals),np.zeros((len(goals),22))],axis=1)-response['zero']
    return np.clip(targets@response['inverse'].T,-.9,.9).astype(np.float32)

def evaluate(env,policy,goals,mode='model',response=None,record=False):
    records=[];episodes=[]
    for delta in goals:
        obs=env.reset(delta);peak_lower=0;peak_drift=0;unsafe=False;frames=[env.frame()] if record else []
        for _ in range(env.steps):
            if mode=='zero':action=np.zeros(64)
            elif mode=='teacher':action=teacher([delta],response)[0]
            else:
                with torch.no_grad():action=policy(torch.from_numpy(obs))[0].numpy()
            obs,_,done,info=env.step(action)
            peak_lower=max(peak_lower,info['lower_body_max_rad']);peak_drift=max(peak_drift,info['root_drift_cm'])
            unsafe|=info['unsafe']
            if record:frames.append(env.frame())
            if done:break
        records.append({'goal_delta_m':delta.tolist(),'error_cm':info['error_cm'],
                        'lower_body_peak_rad':peak_lower,'root_drift_peak_cm':peak_drift,'unsafe':unsafe})
        if record:episodes.append({'metrics':records[-1],'frames':frames})
    errors=np.array([x['error_cm'] for x in records])
    report={'episodes':len(records),'mean_error_cm':float(errors.mean()),'median_error_cm':float(np.median(errors)),
            'p95_error_cm':float(np.quantile(errors,.95)),'under_1cm_fraction':float(np.mean(errors<1)),
            'unsafe_episodes':sum(x['unsafe'] for x in records),
            'max_lower_body_drift_rad':max(x['lower_body_peak_rad'] for x in records),
            'max_root_drift_cm':max(x['root_drift_peak_cm'] for x in records),'trials':records}
    return report,episodes

def save(out,policy,env,update):
    torch.save({'state_dict':policy.state_dict(),'architecture':'131-128-128-64 actor; separate critic',
                'action':'64D bounded residual added to nominal SONIC token',
                'base_token':env.base_token.tolist(),'residual_scale':env.residual_scale,
                'decoder_sha256':decoder_hash(),'training_stage':'short_reach_1_to_3_cm',
                'simulation_only':True,'hardware_approved':False,'update':update},out/'policy.pt')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--updates',type=int,default=60)
    ap.add_argument('--episodes',type=int,default=12);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--init',type=Path);ap.add_argument('--terminal-weight',type=float,default=0.)
    ap.add_argument('--actor-lr',type=float,default=1e-4);ap.add_argument('--separate-grad-clip',action='store_true');a=ap.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'policy.pt').exists():raise SystemExit('Refusing to overwrite an existing training run')
    torch.set_num_threads(1);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    env=TokenReachEnv();response=dict(np.load(ROOT/'runs/probe/response.npz'))
    policy=TokenPolicy();opt=torch.optim.Adam([
        {'params':list(policy.actor.parameters())+[policy.log_std],'lr':a.actor_lr},
        {'params':policy.critic.parameters(),'lr':1e-4}])
    if a.init:
        initial=torch.load(a.init,map_location='cpu',weights_only=True)
        if initial['decoder_sha256']!=decoder_hash():raise ValueError('Initial checkpoint decoder mismatch')
        policy.load_state_dict(initial['state_dict'])
    validation_goals=sample_goals(np.random.default_rng(2026),24)
    start=time.monotonic()
    # Fit the actor to the experimentally measured local token response.
    # Only the goal input is varied at first; actual closed-loop PPO follows.
    for _ in range(0 if a.init else 1200):
        goals=sample_goals(rng,256);x=np.repeat(env.reset([0,0,0])[None],256,axis=0)
        x[:,:3]=goals/.05;x[:,3:6]=goals/.05+rng.normal(0,.1,(256,3))
        target=teacher(goals,response);x[:,-64:]=target*rng.uniform(0,1,(256,1))
        pred,_=policy(torch.from_numpy(x));loss=(pred-torch.from_numpy(target)).square().mean()
        opt.zero_grad();loss.backward();opt.step()
    warm,_=evaluate(env,policy,validation_goals)
    best_score=warm['mean_error_cm']+100*warm['unsafe_episodes'];best=copy.deepcopy(policy.state_dict());best_update=0
    save(out,policy,env,0)
    print('warm_start',json.dumps({k:v for k,v in warm.items() if k!='trials'}),flush=True)
    history=[]
    for update in range(1,a.updates+1):
        xs=[];actions=[];logps=[];advantages=[];returns=[];errors=[]
        for goal in sample_goals(rng,a.episodes):
            obs=env.reset(goal);trajectory=[]
            for _ in range(env.steps):
                x=torch.from_numpy(obs)
                with torch.no_grad():
                    mean,value=policy(x);dist=Normal(mean,policy.log_std.exp());action=dist.sample();logp=dist.log_prob(action).sum()
                obs,reward,done,info=env.step(action.numpy())
                if done:reward-=a.terminal_weight*info['error_cm']/5
                trajectory.append((x,action,logp,float(value),reward))
                if done:break
            errors.append(info['error_cm']);adv=0.;next_value=0.
            for x,action,logp,value,reward in reversed(trajectory):
                td=reward+.99*next_value-value;adv=td+.99*.95*adv
                xs.append(x);actions.append(action);logps.append(logp);advantages.append(adv);returns.append(adv+value);next_value=value
        x=torch.stack(xs);action=torch.stack(actions);old_logp=torch.stack(logps)
        adv=torch.tensor(advantages,dtype=torch.float32);ret=torch.tensor(returns,dtype=torch.float32)
        adv=(adv-adv.mean())/(adv.std()+1e-8)
        expert=torch.from_numpy(teacher(x[:,:3].numpy()*.05,response))
        for _ in range(4):
            for ix in torch.randperm(len(x)).split(256):
                mean,value=policy(x[ix]);dist=Normal(mean,policy.log_std.exp())
                ratio=(dist.log_prob(action[ix]).sum(-1)-old_logp[ix]).exp()
                actor_loss=-torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix]).mean()
                loss=actor_loss+.25*(value-ret[ix]).square().mean()+.2*(mean-expert[ix]).square().mean()
                opt.zero_grad();loss.backward()
                if a.separate_grad_clip:
                    nn.utils.clip_grad_norm_(list(policy.actor.parameters())+[policy.log_std],.5)
                    nn.utils.clip_grad_norm_(policy.critic.parameters(),.5)
                else:nn.utils.clip_grad_norm_(policy.parameters(),.5)
                opt.step()
        row={'update':update,'train_mean_error_cm':float(np.mean(errors)),'steps':update*a.episodes*env.steps}
        if update%5==0 or update==a.updates:
            val,_=evaluate(env,policy,validation_goals);score=val['mean_error_cm']+100*val['unsafe_episodes']
            row['validation']={k:v for k,v in val.items() if k!='trials'}
            if score<best_score:
                best_score=score;best_update=update;best=copy.deepcopy(policy.state_dict());save(out,policy,env,update)
            print(json.dumps(row),flush=True)
        history.append(row)
        (out/'progress.json').write_text(json.dumps({'update':update,'updates':a.updates,'best_update':best_update,'seconds':time.monotonic()-start,'history':history},indent=2))
    torch.save({'state_dict':policy.state_dict(),'update':a.updates,'decoder_sha256':decoder_hash()},out/'last_training_state.pt')
    policy.load_state_dict(best)
    # Test set is never used for checkpoint selection.
    test_goals=sample_goals(np.random.default_rng(917381),100)
    trained,episodes=evaluate(env,policy,test_goals,record=True)
    baseline,_=evaluate(env,policy,test_goals,mode='zero')
    rule,_=evaluate(env,policy,test_goals,mode='teacher',response=response)
    report={'algorithm':'Measured local-response behavior cloning plus PPO; best validation checkpoint',
            'decoder':'Frozen released SONIC decoder in every physics rollout','decoder_sha256':decoder_hash(),
            'seed':a.seed,'training_episodes':a.updates*a.episodes,'training_steps':a.updates*a.episodes*env.steps,
            'init_checkpoint':str(a.init) if a.init else None,'terminal_weight':a.terminal_weight,
            'actor_lr':a.actor_lr,'separate_grad_clip':a.separate_grad_clip,
            'best_update':best_update,'seconds':time.monotonic()-start,'warm_start_validation':warm,
            'test_policy':trained,'test_constant_token':baseline,'test_local_response_rule':rule,'history':history,
            'simulation_only':True,'hardware_approved':False,
            'limitations':['1-3 cm right-hand reach only; no ball observation or hit reward',
                           'One settled initial posture; no domain randomization',
                           'Stock Dex3 hands; actual gripper and racket dynamics not calibrated',
                           'All 29 body commands come from SONIC; lower-body movement is penalized, not mechanically locked',
                           'No hardware watchdog or physical deployment validation']}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (out/'rollouts.json').write_text(json.dumps({'episodes':episodes[:12],'parents':env.model.body_parentid.tolist(),
        'body_names':[env.model.body(i).name for i in range(env.model.nbody)],'simulation_only':True}))
    print('FINAL',json.dumps({k:{n:v for n,v in report[k].items() if n!='trials'} for k in ['test_policy','test_constant_token','test_local_response_rule']}),flush=True)

if __name__=='__main__':main()
