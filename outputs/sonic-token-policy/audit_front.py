"""Independent numerical and full lift-to-interception checks, simulation only."""
import argparse,hashlib,json
from pathlib import Path
import mujoco,numpy as np,torch
from decoder_env import ROOT,SEED
from front_env import FrontEnv,sample_cases
from train_front import evaluate,compact,predict
from train_lift import LiftPolicy,observation
from serve_front import load

def ball_drop(dt):
    coefficients=[];penetrations=[]
    for h in np.linspace(.30,.65,20):
        model=mujoco.MjModel.from_xml_string(f'''<mujoco><option timestep="{dt}"/><worldbody>
            <geom name="table" type="plane" size="3 3 .1"/><body pos="0 0 {h}"><freejoint/>
            <geom name="ball" type="sphere" size=".02" mass=".0027"/></body></worldbody><contact>
            <pair geom1="table" geom2="ball" condim="3" solref=".016 .1" solimp=".95 .99 .001 .5 2"
            friction=".1 .1 .001 .0001 .0001"/></contact></mujoco>''')
        data=mujoco.MjData(model);incoming=None;outgoing=0.;penetration=0.
        for _ in range(int(.7/dt)):
            before=data.qvel[2];mujoco.mj_step(model,data)
            penetration=max(penetration,.02-data.qpos[2])
            if data.ncon and incoming is None:incoming=-before
            if incoming:outgoing=max(outgoing,data.qvel[2])
        coefficients.append(outgoing/incoming);penetrations.append(penetration)
    return {'stepsize_s':dt,'drops':len(coefficients),'restitution_min':min(coefficients),
            'restitution_max':max(coefficients),'restitution_mean':float(np.mean(coefficients)),
            'max_penetration_mm':max(penetrations)*1000}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(1);initial_sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    policy,cp=load(a.checkpoint);cases=sample_cases(np.random.default_rng(234908),60)
    results={}
    for name,dt in [('nominal',.001),('half_timestep',.0005)]:
        env=FrontEnv();env.model.opt.timestep=dt;env.base.substeps=round(.02/dt)
        result,_=evaluate(env,policy,cases);results[name]=result
        print(name,json.dumps(compact(result)),flush=True)
    lift=LiftPolicy();lift.load_state_dict(torch.load(ROOT/'runs/front-lift-v1/policy.pt',weights_only=True)['state_dict'])
    rows=[]
    for case in cases[:20]:
        e=FrontEnv();ready=e.base.seed;e.base.seed=dict(np.load(SEED));e.base.on_substep=None;e.base.reset([0,0,0])
        last=e.base.seed['token'].copy();e.base.seed=ready;lift_unsafe=False
        e.data.qpos[e.ball_qa:e.ball_qa+3]=[6,0,3];e.data.qpos[e.ball_qa+3:e.ball_qa+7]=[1,0,0,0]
        e.data.qvel[e.ball_va:e.ball_va+6]=0;mujoco.mj_forward(e.model,e.data)
        for k in range(250):
            with torch.no_grad():token=lift(torch.from_numpy(observation(e.base,last,k*.02))).numpy().clip(-.6,.6)
            e.base.base_token=token;_,_,_,info=e.base.step(np.zeros(64));last=token
            if info['unsafe']:lift_unsafe=True;break
        height=float(e.face()[2]);e.base.base_token=np.array(cp['base_token'],dtype=np.float32)
        e.base.start_joints=e.data.qpos[e.base.bqa].copy();e.base.start_root=e.data.qpos[:3].copy();e.base.start=e.base.point().copy()
        e.base.last_residual=np.zeros(64);obs=e.reset(case,reset_robot=False);e.base.on_substep=e._contacts
        for _ in range(e.steps):
            obs,_,done,info=e.step(predict(policy,obs))
            if done:break
        rows.append({'lift_height_cm':height*100,'lift_unsafe':lift_unsafe,
            **{k:v for k,v in info.items() if k!='token'}})
    results['full_lift_pipeline']={'episodes':len(rows),'lift_min_height_cm':min(r['lift_height_cm'] for r in rows),
        'unsafe_episodes':sum(r['unsafe'] or r['lift_unsafe'] for r in rows),
        'face_contacts':sum(r['face_contact'] for r in rows),'net_crossings':sum(r['crossed_net'] for r in rows),
        'legal_returns':sum(r['legal_return'] for r in rows),'trials':rows}
    results['ball_drop_checks']=[ball_drop(.001),ball_drop(.0005)]
    if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()!=initial_sha:raise RuntimeError('Checkpoint changed during audit')
    results.update(checkpoint_sha256=initial_sha,simulation_only=True,hardware_approved=False)
    (a.checkpoint.parent/'audit.json').write_text(json.dumps(results,indent=2))
    print('FULL PIPELINE',json.dumps(compact(results['full_lift_pipeline'])),flush=True)
    print('CONTACT DYNAMICS',results['ball_drop_checks'],flush=True)
if __name__=='__main__':main()
