"""PPO smoke training: ball-free right-hand target policy in fixed-base MuJoCo."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from fixed_arm_env import FixedArmReach

SCALE=np.array([2]*7+[5]*7+[.1]*6+[1],dtype=np.float32)
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(21,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.actor=nn.Linear(64,3)
        self.critic=nn.Linear(64,1)
        self.log_std=nn.Parameter(torch.full((3,),-.9))
    def forward(self,x):
        h=self.body(x)
        return self.actor(h),self.critic(h).squeeze(-1)

def features(obs):return torch.as_tensor(np.asarray(obs,dtype=np.float32)/SCALE)

def evaluate(policy,seed,episodes=80,mode='model'):
    env=FixedArmReach(seed=seed);errors=[];success=0
    for _ in range(episodes):
        obs=env.reset()
        for _ in range(env.steps):
            if mode=='zero':action=np.zeros(3)
            elif mode=='rule':action=(env.goal-env.ready)/.05
            else:
                with torch.no_grad():action=policy(features(obs))[0].numpy()
            obs,_,_,info=env.step(action)
        errors.append(info['error_m'])
    return {'episodes':episodes,'mean_error_cm':float(np.mean(errors)*100),'median_error_cm':float(np.median(errors)*100),'success_under_2cm':float(np.mean(np.array(errors)<=.02))}

def train(out,updates=60,episodes_per_update=24,seed=42):
    torch.set_num_threads(1);torch.manual_seed(seed)
    np.random.seed(seed)
    env=FixedArmReach(seed=seed)
    policy=ActorCritic()
    optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
    initial=evaluate(policy,seed=98765)
    # Warm-start from the known target mapping. PPO then trains dynamics-aware corrections.
    for _ in range(400):
        demo=np.zeros((256,21),dtype=np.float32)
        demo[:,17:20]=np.random.uniform(-.35,.35,(256,3))
        demo_t=torch.as_tensor(demo)
        expert=demo_t[:,17:20]*2
        mean,_=policy(demo_t)
        loss=(mean-expert).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    warm=evaluate(policy,seed=98765)
    zero=evaluate(policy,seed=98765,mode='zero')
    rule=evaluate(policy,seed=98765,mode='rule')
    history=[];start=time.monotonic()
    for update in range(updates):
        all_obs=[];all_action=[];all_logp=[];all_adv=[];all_return=[];episode_errors=[]
        for _ in range(episodes_per_update):
            obs=env.reset();episode=[]
            for _ in range(env.steps):
                x=features(obs)
                with torch.no_grad():
                    mean,value=policy(x);dist=Normal(mean,policy.log_std.exp());action=dist.sample();logp=dist.log_prob(action).sum()
                obs2,reward,done,info=env.step(action.numpy())
                episode.append((x,action,logp,float(value),reward))
                obs=obs2
            episode_errors.append(info['error_m'])
            advantage=0.;next_value=0.
            for x,action,logp,value,reward in reversed(episode):
                delta=reward+.99*next_value-value
                advantage=delta+.99*.95*advantage
                all_obs.append(x);all_action.append(action);all_logp.append(logp)
                all_adv.append(advantage);all_return.append(advantage+value)
                next_value=value
        obs=torch.stack(all_obs);actions=torch.stack(all_action);old_logp=torch.stack(all_logp)
        adv=torch.tensor(all_adv,dtype=torch.float32);returns=torch.tensor(all_return,dtype=torch.float32)
        adv=(adv-adv.mean())/(adv.std()+1e-8)
        for _ in range(4):
            for idx in torch.randperm(len(obs)).split(256):
                mean,value=policy(obs[idx]);dist=Normal(mean,policy.log_std.exp())
                ratio=torch.exp(dist.log_prob(actions[idx]).sum(-1)-old_logp[idx])
                p_loss=-torch.minimum(ratio*adv[idx],torch.clamp(ratio,.8,1.2)*adv[idx]).mean()
                v_loss=(value-returns[idx]).square().mean()
                expert=obs[idx,17:20]*2
                imitation=(mean-expert).square().mean()
                loss=p_loss+.4*v_loss+.5*imitation-.001*dist.entropy().sum(-1).mean()
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
        item={'update':update+1,'train_mean_error_cm':float(np.mean(episode_errors)*100)}
        history.append(item)
        if (update+1)%10==0:print(item,flush=True)
    final=evaluate(policy,seed=98765)
    report={'stage':'ball-free fixed-base MuJoCo right-arm reach policy','algorithm':'behavior-cloning warm start plus PPO with auxiliary target loss','seed':seed,
            'episodes_trained':updates*episodes_per_update,'physics_model':'G1 seven right-arm joints; other joints fixed',
            'initial_policy':initial,'warm_start_policy':warm,'trained_policy':final,'zero_action':zero,'rule_target':rule,
            'seconds':time.monotonic()-start,'history':history,
            'limitations':['No ball or racket contact','No SONIC in training loop','Fixed pelvis and lower body','Not a physical robot policy']}
    out.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':policy.state_dict(),'observation_scale':SCALE.tolist(),'architecture':'ActorCritic 21-64-64-(3,1)','action':'right VR target delta, normalized to 5 cm'},out/'policy.pt')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['episodes_trained','initial_policy','warm_start_policy','trained_policy','zero_action','rule_target','seconds']},indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--updates',type=int,default=60);ap.add_argument('--episodes-per-update',type=int,default=24);ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args();train(a.output,a.updates,a.episodes_per_update,a.seed)
