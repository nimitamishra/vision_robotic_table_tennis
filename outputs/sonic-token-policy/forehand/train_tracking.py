"""Motion-constrained residual PPO. Learns SONIC tokens, never joint commands.
No ball-return claim: this stage learns collision-free physical forehand tracking.
"""
import argparse,copy,time
import torch
from torch import nn
from torch.distributions import Normal
from common import *
from train_prior import load
OUT=HERE/'runs/tracking-v1'
class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(129,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,64))
        self.critic=nn.Sequential(nn.Linear(129,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
        self.log_std=nn.Parameter(torch.full((64,),-1.9))
    def forward(self,x):return self.actor(x),self.critic(x).squeeze(-1)
class MotionEnv:
    def __init__(self,dt=.001):
        self.ms=motions();self.base=make_env(self.ms[0],dt);self.encoder=MotionEncoder();self.prior=load(RUN/'policy.pt');self.cache={}
        self.reset(0)
    def reset(self,index):
        self.m=self.ms[index];e=make_env(self.m,existing=self.base)
        if index not in self.cache:
            for k in range(50):
                q,dq=future(self.m,k*.02-1)
                e.base_token=self.encoder.encode(q,dq,e.data.xmat[e.pelvis].reshape(3,3).T)
                e.step(np.zeros(64))
            self.cache[index]=(e.data.qpos.copy(),e.data.qvel.copy(),np.array(e.history),e.last_action.copy())
        qp,qv,h,a=self.cache[index];e.data.qpos[:]=qp;e.data.qvel[:]=qv;e.data.time=0
        e.history=deque(h.copy(),maxlen=10);e.last_action=a.copy();e.last_residual=np.zeros(64);e.k=0
        mujoco.mj_forward(e.model,e.data)
        self.k=0;self.last=np.zeros(64);self.metrics=[]
        self.tokens=np.stack([self.prior(np.r_[self.m['target'],1-t]) for t in self.m['time']])
        return self.obs()
    def obs(self):
        i=min(self.k,125);e=self.base;m=self.m
        return np.r_[(np.r_[m['target'],1-i*.02]-[.43,-.43,.15,0])/[.1,.1,.1,1],
            e.data.qpos[e.bqa]-m['dof'][i],e.data.qvel[e.bva]*.1,-e.data.xmat[e.pelvis].reshape(3,3)[2],self.last].astype(np.float32)
    def step(self,action):
        e=self.base;m=self.m;i=min(self.k,125)
        action=np.asarray(action);assert action.shape==(64,) and np.isfinite(action).all()
        a=np.clip(np.clip(action,-1,1),self.last-.10,self.last+.10)
        e.base_token=self.tokens[i]+.08*a;e.step(np.zeros(64));self.k+=1
        j=min(self.k,125);q=e.data.qpos[e.bqa];r=e.data.xmat[e.pelvis].reshape(3,3)
        local=(face(e)-e.data.xpos[e.pelvis])@r;target=m['actual_contact_pos'][j]-m['root_pos'][j]
        error=float(np.linalg.norm(local-target));ar=float(np.sqrt(np.mean((q[22:]-m['dof'][j,22:])**2)))
        normal=e.data.xmat[e.wrist].reshape(3,3)[:,1];angle=float(np.arccos(np.clip(normal@(m['wrist_rotation'][j,:,1]),-1,1)))
        leg=float(np.max(abs(q[:12]-m['dof'][0,:12])));drift=float(np.linalg.norm(e.data.qpos[:2]-m['root_pos'][0,:2]))
        unsafe=bool(e.data.qpos[2]<.55 or r[2,2]<.9 or drift>.1 or leg>.3 or not np.isfinite(e.data.qpos).all())
        collision=False
        for ci,c in enumerate(e.data.contact):
            names=[e.model.geom(int(g)).name or '' for g in (c.geom1,c.geom2)]
            if any(n in ('training_face','training_handle') for n in names) and 'training_ball_geom' not in names:
                f=np.zeros(6);mujoco.mj_contactForce(e.model,e.data,ci,f)
                collision|=f[0]>1
        at_hit=abs(j*.02-1)<.06
        reward=(3*np.exp(-(ar/.22)**2)+(3+5*at_hit)*np.exp(-(error/.07)**2)
                +np.exp(-(angle/.3)**2)-2*(leg/.15)**2-8*collision
                -2*max(q[23]+.3,0)**2-.1*np.mean(a*a)-.2*np.mean((a-self.last)**2)-20*unsafe)
        self.last=a
        info={'phase_s':j*.02,'face_error_cm':error*100,'arm_rmse_rad':ar,'normal_error_deg':float(angle*180/np.pi),
              'unsafe':unsafe,'obstacle_contact':bool(collision),'legs_max_rad':leg,'root_drift_cm':100*drift,
              'qpos':e.data.qpos.tolist(),'face_m':face(e).tolist(),'normal':normal.tolist()}
        self.metrics.append(info)
        return self.obs(),float(reward),unsafe or self.k>=125,info
    def report(self):
        rr=self.metrics;hit=min(rr,key=lambda r:abs(r['phase_s']-1))
        return {'motion':self.m['name'],'completed':self.k>=125,'unsafe':any(r['unsafe'] for r in rr),
            'obstacle_contact_frames':sum(r['obstacle_contact'] for r in rr),'mean_face_error_cm':float(np.mean([r['face_error_cm'] for r in rr])),
            'hit_face_error_cm':hit['face_error_cm'],'arm_rmse_rad':float(np.mean([r['arm_rmse_rad'] for r in rr])),
            'hit_normal_error_deg':hit['normal_error_deg'],'peak_leg_rad':max(r['legs_max_rad'] for r in rr),'root_drift_cm':max(r['root_drift_cm'] for r in rr)}

def evaluate(env,policy,ids,save=None):
    rows=[]
    for i in ids:
        obs=env.reset(int(i))
        for _ in range(125):
            with torch.no_grad():act=policy(torch.from_numpy(obs))[0].numpy()
            obs,_,done,_=env.step(act)
            if done:break
        rows.append(env.report())
        if save:
            save.mkdir(parents=True,exist_ok=True)
            (save/f'{env.m["name"]}-tracked_prior.json').write_text(json.dumps({'report':rows[-1],'frames':env.metrics}))
    return {'episodes':len(rows),'unsafe':sum(r['unsafe'] for r in rows),'obstacle_episodes':sum(r['obstacle_contact_frames']>0 for r in rows),
        'mean_face_error_cm':float(np.mean([r['mean_face_error_cm'] for r in rows])),
        'mean_hit_error_cm':float(np.mean([r['hit_face_error_cm'] for r in rows])),
        'mean_arm_rmse_rad':float(np.mean([r['arm_rmse_rad'] for r in rows])),'rows':rows}

def quality(r):return -1000*r['unsafe']-20*r['obstacle_episodes']-r['mean_face_error_cm']-r['mean_hit_error_cm']-10*r['mean_arm_rmse_rad']
def compact(r):return {k:v for k,v in r.items() if k!='rows'}
def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--updates',type=int,default=40);p.add_argument('--episodes',type=int,default=8)
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--init',type=Path);args=p.parse_args();OUT=args.output
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'policy.pt').exists():raise RuntimeError('Use a new run directory')
    torch.set_num_threads(1);torch.manual_seed(521);rng=np.random.default_rng(521)
    env=MotionEnv();policy=Residual();a=np.load(RUN/'paired.npz');tr=a['train_ids'];va=a['validation_ids'];te=a['test_ids']
    if args.init:
        cp=torch.load(args.init,map_location='cpu',weights_only=True)
        if cp['decoder_sha256']!=sha(DECODER) or cp['prior_sha256']!=sha(RUN/'policy.pt') or cp['hardware_approved']:raise ValueError('Incompatible warm-start checkpoint')
        policy.load_state_dict(cp['state_dict'])
    optimizer=torch.optim.Adam([{'params':list(policy.actor.parameters())+[policy.log_std],'lr':.0001},{'params':policy.critic.parameters(),'lr':.0005}])
    baseline=evaluate(env,policy,va);best=quality(baseline);weights=copy.deepcopy(policy.state_dict());best_update=0
    print('BASELINE',json.dumps(compact(baseline)),flush=True);history=[];start=time.monotonic()
    for update in range(1,args.updates+1):
        samples=[]
        for index in rng.choice(tr,args.episodes,replace=True):
            obs=env.reset(int(index));trajectory=[]
            for _ in range(125):
                x=torch.from_numpy(obs)
                with torch.no_grad():
                    mean,value=policy(x);dist=Normal(mean,policy.log_std.exp());action=dist.sample();lp=dist.log_prob(action).sum()
                obs,reward,done,_=env.step(action.numpy());trajectory.append((x,action,lp,float(value),reward))
                if done:break
            adv=0.;nv=0.
            for x,action,lp,v,r in reversed(trajectory):
                adv=r+.99*nv-v+.99*.95*adv;samples.append((x,action,lp,adv,adv+v));nv=v
        x=torch.stack([s[0] for s in samples]);act=torch.stack([s[1] for s in samples]);old=torch.stack([s[2] for s in samples]);adv=torch.tensor([s[3] for s in samples]);ret=torch.tensor([s[4] for s in samples]);adv=(adv-adv.mean())/(adv.std()+1e-8)
        for _ in range(4):
            for ix in torch.randperm(len(x)).split(256):
                mean,v=policy(x[ix]);dist=Normal(mean,policy.log_std.exp());ratio=(dist.log_prob(act[ix]).sum(-1)-old[ix]).exp()
                loss=-torch.minimum(ratio*adv[ix],ratio.clamp(.8,1.2)*adv[ix]).mean()+.01*(v-ret[ix]).square().mean()+.01*mean.square().mean()
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),1.);optimizer.step()
        row={'update':update,'seconds':time.monotonic()-start}
        if update%5==0 or update==args.updates:
            val=evaluate(env,policy,va);row['validation']=compact(val)
            if quality(val)>best:best=quality(val);weights=copy.deepcopy(policy.state_dict());best_update=update
            print(json.dumps(row),flush=True)
        history.append(row);(OUT/'progress.json').write_text(json.dumps({'best_update':best_update,'history':history}))
    policy.load_state_dict(weights)
    torch.save({'state_dict':weights,'simulation_only':True,'hardware_approved':False,'prior_sha256':sha(RUN/'policy.pt'),'decoder_sha256':sha(DECODER),
                'best_update':best_update,'residual_scale':.08,'slew':.10,'observation_dim':129,
                'updates':args.updates,'episodes_per_update':args.episodes,'seed':521},OUT/'policy.pt')
    result=evaluate(env,policy,te,OUT/'rollouts');original=evaluate(env,Residual(),te)
    allrows=evaluate(env,policy,range(20),OUT/'rollouts')
    report={'simulation_only':True,'hardware_approved':False,'stage':'Motion tracking, no ball-return success claim','baseline_validation':baseline,'test':result,'test_without_residual':original,'all_grid':allrows,'best_update':best_update,'history':history}
    (OUT/'report.json').write_text(json.dumps(report,indent=2));print('FINAL',json.dumps(compact(result)),flush=True)
if __name__=='__main__':main()
