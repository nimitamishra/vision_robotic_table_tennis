"""PPO for front-fed table tennis, using only SONIC token actions."""
import argparse, copy, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from decoder_env import ROOT, decoder_hash
from front_env import CONFIG, FrontEnv, config_hash, sample_cases
from train_impact import ImpactPolicy, predict


def evaluate(env, policy, cases, zero=False, record=False):
    rows=[]; episodes=[]
    for case in cases:
        obs=env.reset(case); frames=[]; total=0.
        for _ in range(env.steps):
            obs,reward,done,info=env.step(np.zeros(64) if zero else predict(policy,obs)); total+=reward
            if record:frames.append({'time_s':env.time,'qpos':env.data.qpos.tolist(),'target_m':env.target.tolist()})
            if done:break
        row={k:v for k,v in info.items() if k!='token'};row.update(case=case,reward=total);rows.append(row)
        if record:episodes.append({'metrics':row,'frames':frames,'contact_events':env.contact_events})
    failures={}
    for r in rows:failures[r['failure'] or 'none']=failures.get(r['failure'] or 'none',0)+1
    report={'episodes':len(rows),'face_contacts':sum(r['face_contact'] for r in rows),
        'net_crossings':sum(r['crossed_net'] for r in rows),'legal_returns':sum(r['legal_return'] for r in rows),
        'unsafe_episodes':sum(r['unsafe'] for r in rows),'mean_miss_cm':float(np.mean([r['minimum_miss_cm'] for r in rows])),
        'mean_reward':float(np.mean([r['reward'] for r in rows])),
        'max_lower_body_rad':max(r['lower_body_peak_rad'] for r in rows),
        'max_root_drift_cm':max(r['root_drift_peak_cm'] for r in rows),'failures':failures,'trials':rows}
    return report,episodes


def compact(report):return {k:v for k,v in report.items() if k!='trials'}
def score(r):return (100*r['legal_returns']+20*r['net_crossings']+2*r['face_contacts']-1000*r['unsafe_episodes'])/r['episodes']+.005*r['mean_reward']
class ReturnShapingEnv(FrontEnv):
    """Add a flight-outcome reward after a real racket contact, never set ball velocity."""
    def step(self,action):
        obs,reward,done,info=super().step(action)
        if self.hit and self.time-self.hit_time>.025:
            p=self.ball_position();v=self.ball_velocity()
            roots=np.roots([-.5*9.81,v[2],p[2]-.78])
            times=[float(t.real) for t in roots if abs(t.imag)<1e-8 and t.real>0]
            if times:
                t=max(times);landing=p[:2]+v[:2]*t
                distance=np.linalg.norm(landing-np.array([3.0,0.]))
                reward+=2.*np.exp(-distance/.8)
                # Reward frontward distance, while the real net/table contacts
                # continue to decide whether the return is legal.
                reward+=.4*np.clip(v[0],-2,5)
        return obs,float(reward),done,info


class StrokeShapingEnv(ReturnShapingEnv):
    """Favor a forward-facing racket and a timed forward stroke before contact."""
    def step(self,action):
        before=self.face();incoming=not self.hit;tti=self.tti
        obs,reward,done,info=super().step(action)
        if incoming:
            desired=np.array([.93969262,0.,.34202014])
            reward-=3.*float(np.sum((self.normal()-desired)**2))
            if self.bounces==1:
                forward_speed=(self.face()[0]-before[0])/self.dt
                reward+=2.*float(np.clip(forward_speed,-1,2))*np.exp(-((tti-.08)/.15)**2)
        return obs,float(reward),done,info


def save(out,policy,env,update):
    torch.save({'state_dict':policy.state_dict(),'architecture':'141-128-128-64 actor',
        'action':'64D residual token through frozen SONIC','base_token':env.base.base_token.tolist(),
        'residual_scale':env.base.residual_scale,'decoder_sha256':decoder_hash(),
        'environment_sha256':config_hash(),'environment':CONFIG,'training_stage':CONFIG['stage'],
        'simulation_only':True,'hardware_approved':False,'update':update,
        'return_shaping':isinstance(env,ReturnShapingEnv),'stroke_shaping':isinstance(env,StrokeShapingEnv)},out/'policy.pt')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--updates',type=int,default=120);ap.add_argument('--episodes',type=int,default=16)
    ap.add_argument('--seed',type=int,default=892);ap.add_argument('--init',type=Path)
    ap.add_argument('--return-shaping',action='store_true');ap.add_argument('--stroke-shaping',action='store_true');a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    source_sha=hashlib.sha256(a.init.read_bytes()).hexdigest() if a.init else None
    if (a.output/'policy.pt').exists():raise SystemExit('Run already exists')
    torch.set_num_threads(1);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    env=StrokeShapingEnv() if a.stroke_shaping else ReturnShapingEnv() if a.return_shaping else FrontEnv();policy=ImpactPolicy()
    if a.init:
        cp=torch.load(a.init,map_location='cpu',weights_only=True)
        if cp['environment_sha256']!=config_hash():raise ValueError('Environment mismatch')
        policy.load_state_dict(cp['state_dict'])
        torch.save(cp,a.output/'initial_checkpoint.pt')
    else:
        nn.init.zeros_(policy.actor[-1].weight);nn.init.zeros_(policy.actor[-1].bias)
        policy.log_std.data.fill_(-1.5)
    opt=torch.optim.Adam([{'params':list(policy.actor.parameters())+[policy.log_std],'lr':2e-4},
                          {'params':policy.critic.parameters(),'lr':4e-4}])
    cases=sample_cases(np.random.default_rng(612601),40)
    initial,_=evaluate(env,policy,cases);best_score=score(initial);best=copy.deepcopy(policy.state_dict());best_update=0
    save(a.output,policy,env,0);print('INITIAL',json.dumps(compact(initial)),flush=True)
    started=time.monotonic();history=[];total_steps=0
    for update in range(1,a.updates+1):
        xs=[];actions=[];logps=[];advantages=[];returns=[];hits=[]
        for case in sample_cases(rng,a.episodes):
            obs=env.reset(case);trajectory=[]
            for _ in range(env.steps):
                x=torch.from_numpy(obs)
                with torch.no_grad():
                    mean,value=policy(x);dist=Normal(mean,policy.log_std.exp());action=dist.sample();logp=dist.log_prob(action).sum()
                obs,reward,done,info=env.step(action.numpy());total_steps+=1
                trajectory.append((x,action,logp,float(value),reward))
                if done:break
            hits.append(info['face_contact']);adv=0.;next_v=0.
            for x,action,logp,value,reward in reversed(trajectory):
                td=reward+.99*next_v-value;adv=td+.99*.95*adv
                xs.append(x);actions.append(action);logps.append(logp);advantages.append(adv);returns.append(adv+value);next_v=value
        x=torch.stack(xs);act=torch.stack(actions);old=torch.stack(logps)
        adv=torch.tensor(advantages,dtype=torch.float32);ret=torch.tensor(returns,dtype=torch.float32)
        adv=(adv-adv.mean())/(adv.std()+1e-8)
        for _ in range(4):
            for ix in torch.randperm(len(x)).split(256):
                mean,value=policy(x[ix]);dist=Normal(mean,policy.log_std.exp())
                ratio=(dist.log_prob(act[ix]).sum(-1)-old[ix]).exp()
                loss=-torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix]).mean()+.25*(value-ret[ix]).square().mean()+.002*mean.square().mean()
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),1.);opt.step()
        row={'update':update,'train_face_fraction':float(np.mean(hits)),'steps':total_steps}
        if update%5==0 or update==a.updates:
            val,_=evaluate(env,policy,cases);row['validation']=compact(val)
            if score(val)>best_score:
                best_score=score(val);best_update=update;best=copy.deepcopy(policy.state_dict());save(a.output,policy,env,update)
            print(json.dumps(row),flush=True)
        history.append(row)
        (a.output/'progress.json').write_text(json.dumps({'update':update,'updates':a.updates,'best_update':best_update,
            'seconds':time.monotonic()-started,'history':history},indent=2))
    policy.load_state_dict(best);test=sample_cases(np.random.default_rng(881337),120)
    learned,_=evaluate(env,policy,test);baseline,_=evaluate(env,policy,test,zero=True)
    _,episodes=evaluate(env,policy,test[:16],record=True)
    report={'environment':CONFIG,'environment_sha256':config_hash(),'decoder_sha256':decoder_hash(),
        'algorithm':'PPO, with a separate learned SONIC-token lifting policy before interception',
        'training_steps':total_steps,'training_episodes':a.updates*a.episodes,'best_update':best_update,
        'return_shaping':a.return_shaping or a.stroke_shaping,'stroke_shaping':a.stroke_shaping,'initial_checkpoint':str(a.init) if a.init else None,'initial_sha256':source_sha,
        'simulation_only':True,'hardware_approved':False,'test_policy':learned,'test_constant_token':baseline,
        'history':history,'limitations':['Provisional racket mount, mass and contact dynamics',
        'Perfect simulated state, no sensor noise or latency','Narrow front-feed curriculum; no full rally',
        'SONIC balances all 29 body joints; legs are not locked','No hardware actuation or deployment approval']}
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    (a.output/'rollouts.json').write_text(json.dumps({'episodes':episodes}))
    print('FINAL',json.dumps({'policy':compact(learned),'baseline':compact(baseline)}),flush=True)

if __name__=='__main__':main()
