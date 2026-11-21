"""Independent return event validation and timestep sensitivity."""
import argparse
from return_env import *
from train_return import evaluate

def validate_events(folder):
    checked=0
    for p in sorted(folder.glob('*.json')):
        doc=json.loads(p.read_text());r=doc['report'];events=doc['events'];frames=doc['frames']
        if r['legal_return']:
            before=[e for e in events if e['type']=='table' and NEAR<e['position'][0]<NET]
            hit=[e for e in events if e['type']=='face'];land=[e for e in events if e['type']=='table' and NET<e['position'][0]<FAR]
            assert len(before)==1 and len(hit)==1 and len(land)==1
            assert before[0]['time_s']<hit[0]['time_s']<land[0]['time_s']
            assert r['failure'] is None and r['net_crossing_m'][2]>TOP+.1525+.02
            assert abs(r['net_crossing_m'][1])<WIDTH/2-.02 and abs(land[0]['position'][1])<WIDTH/2-.02
            assert np.asarray(frames[-1]['qpos']).shape[0]>40
            checked+=1
    return checked

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run
    cp=json.loads((out/'policy.json').read_text());params=np.array(cp['params']);report=json.loads((out/'report.json').read_text())
    assert cp['decoder_sha256']==sha(DECODER) and cp['reference_sha256']==sha(out/'forward_reference.npz')
    n=validate_events(out/'test_rollouts');wide=validate_events(out/'wide_rollouts')
    cases=sample_cases(np.random.default_rng(9917),20);e=ReturnEnv(out/'forward_reference.npz',dt=.0005)
    half=evaluate(e,params,cases,out/'half_dt_rollouts');validate_events(out/'half_dt_rollouts')
    # Ball/table restitution diagnostic; isolated from the robot and from learning.
    drops=[]
    for dt in (.001,.0005):
        e=ReturnEnv(out/'forward_reference.npz',dt=dt)
        for height in (.90,1.1,1.3):
            e.reset(params,NOMINAL);e.base.on_substep=None;e.data.qpos[e.qa:e.qa+3]=[2.3,0,height];e.data.qvel[e.va:e.va+6]=0;mujoco.mj_forward(e.model,e.data)
            pre=0.;post=0.;seen=False;penetration=0.
            for k in range(1000):
                v=e.bv()[2];mujoco.mj_step(e.model,e.data);contacts=[c for c in e.data.contact if e.bg in (c.geom1,c.geom2) and e.tg in (c.geom1,c.geom2)]
                if contacts and not seen:pre=abs(v);seen=True
                if contacts:penetration=max(penetration,max(-c.dist for c in contacts))
                if seen and not contacts and e.bv()[2]>0:post=e.bv()[2];break
            drops.append({'dt_s':dt,'height_m':height,'restitution':float(post/pre) if pre else None,'max_penetration_mm':1000*penetration})
    result={'simulation_only':True,'hardware_approved':False,'verified_test_returns':n,'verified_wide_returns':wide,'half_dt':half,
        'test_1ms_first20':[r for r in report['test']['trials'][:20]],'table_drop_diagnostics':drops,
        'policy_sha256':sha(out/'policy.json'),'decoder_sha256':sha(DECODER),'environment_source_sha256':sha(HERE/'return_env.py'),
        'limitations':'Provisional contact model; no spin/drag, sensing latency, mount calibration or hardware validation.'}
    (out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({'verified_returns':n,'half_dt':half['summary'],'drops':drops},indent=2),flush=True)
if __name__=='__main__':main()
