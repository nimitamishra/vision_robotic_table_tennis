from probe_transition import *
from scipy.optimize import least_squares
out=HERE/'runs/transition-20261120'
snapshot=json.loads((out/'robot-start.json').read_text());env=initialize(snapshot);m,d=env.model,env.data
q0=d.qpos[env.bqa].copy();start=d.geom_xpos[m.geom('training_face').id].copy();target=start+np.array([0,0,.03]);normal=d.xmat[env.wrist].reshape(3,3)[:,1].copy()
bounds=m.jnt_range[m.actuator_trnid[env.body_act[22:],0]]
def residual(q):
    d.qpos[env.bqa[22:]]=q;mujoco.mj_forward(m,d)
    return np.r_[20*(d.geom_xpos[m.geom('training_face').id]-target),2*(d.xmat[env.wrist].reshape(3,3)[:,1]-normal),.08*(q-q0[22:])]
fit=least_squares(residual,q0[22:],bounds=(bounds[:,0]+.01,bounds[:,1]-.01));goal=q0.copy();goal[22:]=fit.x
reports=[]
for supported in (True,False):
    name='small_lift_supported' if supported else 'small_lift_free'
    report,frames=run(snapshot,name,4,supported=supported,goal_override=goal,round_trip=True)
    report['requested_raise_cm']=3;report['reference_max_joint_change_deg']=float(np.degrees(np.abs(goal-q0)).max())
    report['initial_face_m']=start.tolist();report['target_face_m']=target.tolist()
    report['actual_peak_raise_cm']=100*max(f['face_m'][2]-start[2] for f in frames)
    (out/(name+'.json')).write_text(json.dumps({'report':report,'frames':frames}));reports.append(report)
    print(json.dumps(report),flush=True)
(out/'small-lift-report.json').write_text(json.dumps(reports,indent=2))
