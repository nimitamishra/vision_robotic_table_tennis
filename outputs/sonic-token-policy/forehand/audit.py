import torch
from common import *
from train_prior import load

def validate_source():
    sys.path.insert(0,str(SOURCE/'scripts'))
    from build_table_tennis_forehand_bundle import load_robot
    robot=load_robot(SOURCE/'assets/vicar/g1.urdf',load_meshes=False)
    emax=0.;targetmax=0.;joint_margin=float('inf');rows=[]
    env=make_env(motions()[0]);mujoco_max=0.
    for m in motions():
        assert list(m['joint_names'])==list(robot.actuated_joint_names)
        assert np.max(abs(m['dof'][:,:12]-m['dof'][0,:12]))==0
        assert np.max(abs(m['root_pos']-m['root_pos'][0]))==0
        assert np.max(abs(m['dof'][0]-m['dof'][-1]))==0
        err=[]
        for k in range(126):
            robot.update_cfg(m['cfg_urdf'][k]);tf=robot.get_transform('right_wrist_yaw_link')
            p=m['root_pos'][k]+tf[:3,3]+tf[:3,:3]@CONTACT
            err.append(np.linalg.norm(p-m['actual_contact_pos'][k]));targetmax=max(targetmax,np.linalg.norm(p-m['target_contact_pos'][k]))
            env.data.qpos[:3]=m['root_pos'][k];env.data.qpos[3:7]=m['root_wxyz'][k];env.data.qpos[env.bqa]=m['dof'][k]
            mujoco.mj_forward(env.model,env.data);mujoco_max=max(mujoco_max,np.linalg.norm(face(env)-p))
        limits=env.model.jnt_range[env.model.actuator_trnid[env.body_act,0]]
        margin=min(float(np.min(m['dof']-limits[:,0])),float(np.min(limits[:,1]-m['dof'])))
        joint_margin=min(joint_margin,margin);emax=max(emax,max(err))
        rows.append({'motion':m['name'],'saved_fk_max_error_m':max(err),'joint_limit_margin_rad':margin})
    result={'motions':len(rows),'saved_fk_max_error_m':emax,'target_max_error_m':targetmax,'mujoco_vs_urdf_fk_max_error_m':mujoco_max,'minimum_joint_limit_margin_rad':joint_margin,'rows':rows}
    assert emax<1e-7 and targetmax<.002 and mujoco_max<1e-5 and joint_margin>=0
    (RUN/'source_validation.json').write_text(json.dumps(result,indent=2));print('SOURCE',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

def main():
    torch.set_num_threads(1);validate_source();policy=load(RUN/'policy.pt');rows=[];directory=RUN/'rollouts';directory.mkdir(exist_ok=True)
    for m in motions():
        for name,p in [('encoded_reference',None),('learned_prior',policy)]:
            frames,report=rollout(m,p);report['controller']=name;rows.append(report)
            (directory/f'{m["name"]}-{name}.json').write_text(json.dumps({'report':report,'frames':frames}))
            print(json.dumps(report),flush=True)
    result={'simulation_only':True,'hardware_approved':False,'checkpoint_sha256':sha(RUN/'policy.pt'),'rows':rows}
    for controller in ('encoded_reference','learned_prior'):
        rr=[r for r in rows if r['controller']==controller]
        result[controller]={'episodes':len(rr),'completed':sum(r['completed'] for r in rr),'unsafe':sum(r['unsafe'] for r in rr),
            'obstacle_episodes':sum(r['obstacle_contact_frames']>0 for r in rr),'mean_hit_error_cm':float(np.mean([r['hit_face_error_cm'] for r in rr])),
            'mean_arm_rmse_rad':float(np.mean([r['arm_rmse_rad'] for r in rr]))}
    (RUN/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
if __name__=='__main__':main()
