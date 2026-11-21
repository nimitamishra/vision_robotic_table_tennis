"""Add fixed-world errors and a smaller-timestep check; never fit on test data."""
import torch
from common import *
from train_tracking import MotionEnv,Residual,evaluate,OUT

def augment(directory):
    ms={m['name']:m for m in motions()};rows=[]
    for path in sorted(directory.glob('*.json')):
        doc=json.loads(path.read_text());m=ms[doc['report']['motion']];frames=doc['frames']
        for f in frames:
            ref=np.array([np.interp(f['phase_s'],m['time'],m['actual_contact_pos'][:,j]) for j in range(3)])
            f['world_face_error_cm']=float(np.linalg.norm(np.array(f['face_m'])-ref)*100)
        active=[f for f in frames if f.get('time_s',0)>=0];hit=min(active,key=lambda f:abs(f['phase_s']-1))
        doc['report']['hit_world_error_cm']=hit['world_face_error_cm']
        doc['report']['mean_world_error_cm']=float(np.mean([f['world_face_error_cm'] for f in active]));rows.append(doc['report'])
        path.write_text(json.dumps(doc))
    return rows

def main():
    torch.set_num_threads(1);cp=torch.load(OUT/'policy.pt',weights_only=True);policy=Residual();policy.load_state_dict(cp['state_dict']);policy.eval()
    data=np.load(RUN/'paired.npz');ids=data['test_ids'];env=MotionEnv(dt=.0005)
    half=evaluate(env,policy,ids,OUT/'half_dt_rollouts')
    mainrows=augment(OUT/'rollouts');halfrows=augment(OUT/'half_dt_rollouts');priorrows=augment(RUN/'rollouts')
    test_names=set(data['names'][ids]);selected=[r for r in mainrows if r['motion'] in test_names]
    report={'simulation_only':True,'hardware_approved':False,'checkpoint_sha256':sha(OUT/'policy.pt'),'prior_sha256':sha(RUN/'policy.pt'),
        'position_error_definitions':{'hit_world_error_cm':'Against fixed source contact target in the table/world frame at t=1.00s',
            'hit_face_error_cm':'Against source contact location after aligning each current pelvis frame'},
        'test_target_count':len(selected),'test_mean_hit_world_error_cm':float(np.mean([r['hit_world_error_cm'] for r in selected])),
        'test_mean_hit_pelvis_aligned_error_cm':float(np.mean([r['hit_face_error_cm'] for r in selected])),
        'half_timestep':half,'half_timestep_world_hit_error_cm':float(np.mean([r['hit_world_error_cm'] for r in halfrows])),
        'all_grid_world_hit_error_cm':float(np.mean([r['hit_world_error_cm'] for r in mainrows])),
        'all_grid':mainrows,'held_out':selected,
        'limitations':['20 kinematic trajectories; 16 training / 2 validation / 2 testing. Two test trajectories are not a broad reliability estimate.',
            'Target coordinates are relative to pelvis at stroke start, held fixed through a 2.5-second stroke. X=0.43m only.',
            'Warm-up uses the reference encoder. Initial pose is initialized in simulation; home-to-ready transition is not validated.',
            'Tracker observation uses reference joint poses from the motion grid plus current proprioception; it is not a stand-alone four-input online controller.',
            'No ball, contact timing perception, return landing, hardware mount, or hardware validation in this stage.',
            'Some low targets still contact the table; the full grid is not collision-free.']}
    (OUT/'final_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('all_grid','held_out','half_timestep')},indent=2),flush=True)
if __name__=='__main__':main()
