"""MuJoCo candidate with the SAME compiled C++ output envelope; no robot transport."""
import ctypes
import probe_transition as probe
from probe_transition import *
from decoder_env import KP,KD
from scipy.optimize import least_squares
parser=argparse.ArgumentParser()
parser.add_argument('--snapshot',type=Path,default=HERE/'runs/transition-20261120/robot-start.json')
parser.add_argument('--output',type=Path,default=HERE/'runs/transition-20261120')
args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
snapshot=json.loads(args.snapshot.read_text())
lib=ctypes.CDLL(str(HERE.parents[1]/'G1-Studio/trial-safety/libtrial_guard.so'))
ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
lib.trial_new.argtypes=[ctypes.c_int];lib.trial_new.restype=ctypes.c_void_p
lib.trial_delete.argtypes=[ctypes.c_void_p]
lib.trial_fault.argtypes=[ctypes.c_void_p];lib.trial_fault.restype=ctypes.c_char_p
lib.trial_joint.argtypes=[ctypes.c_void_p];lib.trial_joint.restype=ctypes.c_int
lib.trial_apply.argtypes=[ctypes.c_void_p,ctypes.c_longlong,ptr,ptr,ptr,ptr,ptr,ptr]
original=probe.initialize;guard_state={}
def guarded_initialize(*args,**kwargs):
 e=original(*args,**kwargs);handle=lib.trial_new(1);guard_state['handle']=handle;guard_state['fault_time']=None
 result=np.empty(87);held=None;last_tick=-1
 guard_state['initial_face']=e.data.geom_xpos[e.model.geom('training_face').id].copy()
 guard_state['fault_state']=None
 def filter_command(target):
  nonlocal held,last_tick
  tick=round(e.data.time/.001)
  if held is None or tick-last_tick>=2:
   last_tick=tick
   lib.trial_apply(handle,round(e.data.time*1e9),np.ascontiguousarray(e.data.qpos[e.bqa]),np.ascontiguousarray(e.data.qvel[e.bva]),np.ascontiguousarray(target),KP,KD,result)
   held=tuple(x.copy() for x in np.split(result,3))
   if lib.trial_fault(handle) and guard_state['fault_time'] is None:
    guard_state['fault_time']=float(e.data.time);e.stop_requested=True
    joint=lib.trial_joint(handle)
    guard_state['fault_state']={'joint_index':joint,'joint_name':e.model.joint(int(e.model.actuator_trnid[e.body_act[joint],0])).name if joint>=0 else None,'dq_rad_s':float(e.data.qvel[e.bva[joint]]) if joint>=0 else None}
  return held
 e.command_filter=filter_command
 return e
# Solve only a 3 cm Cartesian reference change, preserving the face normal.
e=original(snapshot);m,d=e.model,e.data;q0=d.qpos[e.bqa].copy();start=d.geom_xpos[m.geom('training_face').id].copy();normal=d.xmat[e.wrist].reshape(3,3)[:,1].copy();goalpos=start+[0,0,.03]
bounds=m.jnt_range[m.actuator_trnid[e.body_act[22:],0]]
def residual(q):
 d.qpos[e.bqa[22:]]=q;mujoco.mj_forward(m,d)
 return np.r_[20*(d.geom_xpos[m.geom('training_face').id]-goalpos),2*(d.xmat[e.wrist].reshape(3,3)[:,1]-normal),.08*(q-q0[22:])]
fit=least_squares(residual,q0[22:],bounds=(bounds[:,0]+.01,bounds[:,1]-.01));goal=q0.copy();goal[22:]=fit.x
probe.initialize=guarded_initialize
reports=[]
for name,hold,ground,hands in [('guarded_hold',True,False,True),('guarded_small_lift',False,False,True),('guarded_grounded_hold',True,True,True),('guarded_grounded_lift',False,True,True),('guarded_grounded_no_finger_hold',True,True,False),('guarded_grounded_no_finger_lift',False,True,False)]:
 report,frames=probe.run(snapshot,name,4,hold=hold,goal_override=q0 if hold else goal,round_trip=not hold,ground_feet=ground,stock_hand_contacts=hands)
 fault=lib.trial_fault(guard_state['handle']);report.update(command_guard_fault=fault.decode() if fault else None,command_guard_fault_time_s=guard_state['fault_time'],command_guard_fault_state=guard_state['fault_state'],other_joints_held=True)
 report['diagnostic_pass'] &= fault is None
 report['initial_face_m']=guard_state['initial_face'].tolist();report['snapshot_sha256']=sha(args.snapshot)
 report['actual_peak_raise_cm']=100*max([0.]+[f['face_m'][2]-guard_state['initial_face'][2] for f in frames])
 (out/(name+'.json')).write_text(json.dumps({'report':report,'frames':frames}));reports.append(report);lib.trial_delete(guard_state['handle'])
 print(json.dumps(report),flush=True)
(out/'guarded-report.json').write_text(json.dumps(reports,indent=2))
