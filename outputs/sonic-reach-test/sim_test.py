"""SONIC reach experiment, loopback MuJoCo only. Never accepts a robot NIC."""
import os,sys,time,json,signal,subprocess,argparse
from pathlib import Path
from collections import deque
from checks import settled
from feedback import update_correction
import numpy as np
import mujoco,zmq,msgpack
from token_stream import TokenReplay, token_packet, validate_token
ROOT=Path(__file__).resolve().parent
REPO=Path('/home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl')
sys.path.insert(0,str(REPO))
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.simulator_factory import init_channel,SimulatorFactory
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message,build_planner_message

def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data));tmp.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--duration',type=float,default=90);ap.add_argument('--position-feedback',action='store_true');ap.add_argument('--freeze-reach-feedback',action='store_true',help='Keep the settled hold correction constant during reach');ap.add_argument('--release-support',action='store_true',help='Legacy flag; grounded startup now always releases support');ap.add_argument('--delta',type=float,nargs=3,default=[.05,0,0]);ap.add_argument('--absolute-right-target',type=float,nargs=3,help='Pelvis-relative right tracking-point goal, simulation only');ap.add_argument('--right-joint-goal',type=float,nargs=7,help='Simulation-only right-arm joint target in MuJoCo order');ap.add_argument('--initial-right-q',type=float,nargs=7,help='Simulation-only initial right-arm joint position');ap.add_argument('--hold-initial-upper-body',action='store_true',help='Condition SONIC on the initial joint pose from startup, simulation only');ap.add_argument('--hold-initial-vr',action='store_true',help='Also condition startup on the initial three tracking points');ap.add_argument('--startup-joint-feedback',action='store_true',help='Bounded simulation-only feedback for the initial right-arm joint pose');ap.add_argument('--sim-right-arm-direct-servo',action='store_true',help='Loopback-only hybrid test: direct right-arm motor targets, SONIC for balance');ap.add_argument('--sim-arm-gravity-comp',action='store_true',help='MuJoCo-only right-arm gravity/bias feedforward; never sent to robot');ap.add_argument('--sim-arm-extra-kp',type=float,default=0.,help='Additional MuJoCo-only right-arm proportional torque gain');ap.add_argument('--sim-arm-extra-kd',type=float,default=0.,help='Additional MuJoCo-only right-arm damping torque gain');ap.add_argument('--sim-online-arm-ik',action='store_true',help='Plan a short arm reach from the settled simulated pose');ap.add_argument('--sim-right-arm-joint-feedback',action='store_true',help='Bounded joint correction during the simulated reach hold');ap.add_argument('--sim-right-arm-cartesian-feedback',action='store_true',help='Bounded hand-position correction during the simulated reach hold');ap.add_argument('--standing-seconds',type=float,default=6);ap.add_argument('--stage-length',type=float,help='Maximum distance of each simulated waypoint in metres');ap.add_argument('--stage-hold-seconds',type=float,default=1.5);ap.add_argument('--ramp-seconds',type=float,default=2);ap.add_argument('--hold-seconds',type=float,default=5);ap.add_argument('--auto-reach',action='store_true');ap.add_argument('--policy-checkpoint',type=Path);ap.add_argument('--record-token-state',action='store_true',help='Record live encoder tokens from the loopback controller');ap.add_argument('--token-replay',type=Path,help='Replay a recorded simulation latent sequence during reach');a=ap.parse_args();delta=np.array(a.delta);
    if not np.all(np.isfinite(delta)) or np.linalg.norm(delta)>.050001:raise SystemExit('Displacement must be finite and at most 5 cm.')
    if a.absolute_right_target is not None and (not np.all(np.isfinite(a.absolute_right_target)) or a.policy_checkpoint):raise SystemExit('Absolute target must be finite and cannot be combined with a policy checkpoint.')
    if a.right_joint_goal is not None and (a.absolute_right_target is None or not np.all(np.isfinite(a.right_joint_goal)) or a.policy_checkpoint):raise SystemExit('Right-joint goal requires a finite absolute tracking target and no policy checkpoint.')
    if a.initial_right_q is not None and ((a.right_joint_goal is None and not a.sim_online_arm_ik) or not np.all(np.isfinite(a.initial_right_q))):raise SystemExit('Initial right-arm joints require an IK mode or a finite right-joint goal.')
    if a.hold_initial_upper_body and (a.initial_right_q is None or a.position_feedback):raise SystemExit('Holding initial joints requires initial-right-q and no VR position feedback.')
    if a.hold_initial_vr and not a.hold_initial_upper_body:raise SystemExit('Holding initial VR points requires holding initial upper-body joints.')
    if a.startup_joint_feedback and not a.hold_initial_upper_body:raise SystemExit('Startup joint feedback requires holding initial upper-body joints.')
    if a.sim_right_arm_direct_servo and (not a.hold_initial_upper_body or a.startup_joint_feedback or a.hold_initial_vr):raise SystemExit('Direct right-arm servo needs initial upper-body hold, no external joint feedback, and no VR hold.')
    if a.sim_arm_gravity_comp and not a.sim_right_arm_direct_servo:raise SystemExit('Gravity compensation requires the loopback-only direct arm servo.')
    if not np.isfinite(a.sim_arm_extra_kp) or not 0<=a.sim_arm_extra_kp<=80 or not np.isfinite(a.sim_arm_extra_kd) or not 0<=a.sim_arm_extra_kd<=10:raise SystemExit('Extra simulator gains must be within 0-80 Kp and 0-10 Kd.')
    if (a.sim_arm_extra_kp or a.sim_arm_extra_kd) and not a.sim_right_arm_direct_servo:raise SystemExit('Extra simulator gains require the loopback-only direct arm servo.')
    if a.sim_online_arm_ik and (not a.sim_right_arm_direct_servo or a.initial_right_q is None or a.right_joint_goal is not None or a.absolute_right_target is not None or a.policy_checkpoint):raise SystemExit('Online IK requires loopback direct servo, initial joints, and a relative short reach only.')
    if a.sim_right_arm_joint_feedback and (not a.sim_right_arm_direct_servo or (a.right_joint_goal is None and not a.sim_online_arm_ik)):raise SystemExit('Right-arm joint feedback requires direct servo and a joint goal.')
    if a.sim_right_arm_cartesian_feedback and (not a.sim_right_arm_direct_servo or (a.right_joint_goal is None and not a.sim_online_arm_ik) or a.sim_right_arm_joint_feedback):raise SystemExit('Cartesian feedback requires direct servo and a joint goal, without joint feedback.')
    if not 6<=a.standing_seconds<=20:raise SystemExit('Standing period must be 6-20 s.')
    if a.freeze_reach_feedback and not a.position_feedback:raise SystemExit('Freeze-reach-feedback requires position-feedback.')
    if a.stage_length is not None and (a.absolute_right_target is None or not .01<=a.stage_length<=.05 or a.policy_checkpoint or a.right_joint_goal is not None):raise SystemExit('Stage length requires an absolute target, no joint goal, and must be 1-5 cm.')
    if not 1<=a.stage_hold_seconds<=3:raise SystemExit('Stage hold must be 1-3 s.')
    if not 2<=a.ramp_seconds<=12 or not 3<=a.hold_seconds<=15:raise SystemExit('Ramp must be 2-12 s and hold 3-15 s.')
    replay=TokenReplay(a.token_replay) if a.token_replay else None
    if replay is not None:
        if (not a.auto_reach or a.policy_checkpoint or a.sim_right_arm_direct_servo
                or a.right_joint_goal or a.initial_right_q or a.hold_initial_upper_body):
            raise SystemExit('Token replay requires automatic standard-controller simulation, without joint/direct-servo targets.')
        source=json.loads((replay.path.parent/'config.json').read_text())
        if (not np.allclose(source['delta_m'],delta) or source['ramp_seconds']!=a.ramp_seconds
                or source['hold_seconds']!=a.hold_seconds
                or replay.times[-1]<a.ramp_seconds+a.hold_seconds-.1):
            raise SystemExit('Replay trajectory and duration must match the recorded source.')
        if source.get('sim_right_arm_direct_servo') or source.get('sim_arm_gravity_comp'):
            raise SystemExit('Cannot validate latent replay from a direct-servo source.')
    out=a.output.resolve()
    if a.policy_checkpoint:a.policy_checkpoint=a.policy_checkpoint.resolve()
    os.chdir(REPO)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'samples.jsonl').exists():raise SystemExit('Choose a new output directory; previous samples will not be overwritten.')
    atomic(out/'config.json',{'delta_m':delta.tolist(),'absolute_right_target_m':a.absolute_right_target,'right_joint_goal_rad':a.right_joint_goal,'initial_right_q_rad':a.initial_right_q,'hold_initial_upper_body':a.hold_initial_upper_body,'hold_initial_vr':a.hold_initial_vr,'startup_joint_feedback':a.startup_joint_feedback,'sim_right_arm_direct_servo':a.sim_right_arm_direct_servo,'sim_arm_gravity_comp':a.sim_arm_gravity_comp,'sim_arm_extra_kp':a.sim_arm_extra_kp,'sim_arm_extra_kd':a.sim_arm_extra_kd,'sim_online_arm_ik':a.sim_online_arm_ik,'sim_right_arm_joint_feedback':a.sim_right_arm_joint_feedback,'sim_right_arm_cartesian_feedback':a.sim_right_arm_cartesian_feedback,'standing_seconds':a.standing_seconds,'stage_length_m':a.stage_length,'stage_hold_seconds':a.stage_hold_seconds,'ramp_seconds':a.ramp_seconds,'hold_seconds':a.hold_seconds,'auto_reach':a.auto_reach,'position_feedback':a.position_feedback,'freeze_reach_feedback':a.freeze_reach_feedback,'policy_checkpoint':str(a.policy_checkpoint.resolve()) if a.policy_checkpoint else None})
    saved_config=json.loads((out/'config.json').read_text());saved_config.update(record_token_state=a.record_token_state,token_replay=str(replay.path) if replay else None);atomic(out/'config.json',saved_config)
    policy=None
    if a.policy_checkpoint:
        import torch
        sys.path.insert(0,str(ROOT.parent/'rl-interception'))
        from train_reach_policy import ActorCritic,features
        policy=ActorCritic();policy.load_state_dict(torch.load(a.policy_checkpoint,map_location='cpu',weights_only=True)['state_dict']);policy.eval()
    # Refuse to share DDS with another local controller/simulator.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:cmd=(proc/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if cmd and (Path(os.fsdecode(cmd[0])).name=='g1_deploy_onnx_ref' or any(x.endswith(b'/run_sim_loop.py') for x in cmd)):
            raise SystemExit('Another SONIC controller/simulator exists. Close it before this simulation test.')
    config=SimLoopConfig(interface='lo',enable_onscreen=False,enable_offscreen=False).load_wbc_yaml()
    config.update(INTERFACE='lo',DOMAIN_ID=0,USE_JOYSTICK=False)
    init_channel(config);sim=SimulatorFactory.create_simulator(config,onscreen=False,offscreen=False);env=sim.sim_env
    model,data=env.mj_model,env.mj_data
    env.elastic_band.point=env.elastic_band.point.astype(float)
    env.check_fall=lambda: None
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(model.nbody)]
    print('Bodies:',names,flush=True)
    right_arm_body_ids={i for i,name in enumerate(names) if name and name.startswith(('right_shoulder','right_elbow','right_wrist','right_hand'))}
    ids=[model.body(n).id for n in ['left_wrist_yaw_link','right_wrist_yaw_link','torso_link']]
    arm_names=('right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow','right_wrist_roll','right_wrist_pitch','right_wrist_yaw')
    qadr=[model.joint(n+'_joint').qposadr[0] for n in arm_names];vadr=[model.joint(n+'_joint').dofadr[0] for n in arm_names]
    arm_limits=np.array([model.jnt_range[model.joint(name+'_joint').id] for name in arm_names])
    upper_names=('waist_yaw','waist_roll','waist_pitch','left_shoulder_pitch','right_shoulder_pitch','left_shoulder_roll','right_shoulder_roll','left_shoulder_yaw','right_shoulder_yaw','left_elbow','right_elbow','left_wrist_roll','right_wrist_roll','left_wrist_pitch','right_wrist_pitch','left_wrist_yaw','right_wrist_yaw')
    upper_qadr=[model.joint(n+'_joint').qposadr[0] for n in upper_names]
    upper_vadr=[model.joint(n+'_joint').dofadr[0] for n in upper_names]
    right_upper_idx=(4,6,8,10,12,14,16)
    for label,values in [('goal',a.right_joint_goal),('initial',a.initial_right_q)]:
        if values is None:continue
        for name,value,(low,high) in zip(arm_names,values,arm_limits):
            if not low+.02<=value<=high-.02:raise SystemExit(f'{name} {label} is outside joint limits.')
    if a.initial_right_q is not None:
        data.qpos[qadr]=a.initial_right_q
        data.qvel[vadr]=0
        mujoco.mj_forward(model,data)
    startup_upper_target=np.array(data.qpos[upper_qadr]) if a.hold_initial_upper_body else None
    # While TensorRT loads, no controller torques are present and a seeded arm
    # would simply fall. Pin only the simulation's initial upper-body joints
    # until the controller has captured its INIT target.
    startup_pose_pin=bool(a.hold_initial_upper_body)
    if a.sim_arm_gravity_comp or a.sim_arm_extra_kp or a.sim_arm_extra_kd:
        original_body_torques=env.compute_body_torques
        def compensated_body_torques():
            motor_torques=original_body_torques()
            if not startup_pose_pin and env.unitree_bridge.low_cmd:
                # Diagnostic controls exist only inside this loopback simulator.
                # The simulator still clips each motor to its configured limit.
                cmds=env.unitree_bridge.low_cmd.motor_cmd[22:29]
                motor_torques[22:29]+=a.sim_arm_extra_kp*(np.array([c.q for c in cmds])-data.qpos[qadr])
                motor_torques[22:29]+=a.sim_arm_extra_kd*(np.array([c.dq for c in cmds])-data.qvel[vadr])
            if not startup_pose_pin and a.sim_arm_gravity_comp:
                motor_torques[22:29]+=data.qfrc_bias[vadr]
            return motor_torques
        env.compute_body_torques=compensated_body_torques
    pelvis=model.body('pelvis').id;offsets=np.array([[.18,-.025,0],[.18,.025,0],[0,0,.35]])
    ctx=zmq.Context();pub=ctx.socket(zmq.PUB);pub.bind('tcp://127.0.0.1:15576')
    logfile=(out/'controller.log').open('w');controller_env=os.environ.copy();controller_env['SONIC_SIM_HOLD_INIT_UPPER_BODY']='1' if a.hold_initial_upper_body else '0';controller_env['SONIC_SIM_RIGHT_ARM_SERVO']='1' if a.sim_right_arm_direct_servo else '0';controller=subprocess.Popen(['bash',str(ROOT/'controller.sh')],stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True,env=controller_env)
    token_sub=ctx.socket(zmq.SUB);token_sub.setsockopt(zmq.SUBSCRIBE,b'g1_debug');token_sub.setsockopt(zmq.LINGER,0);token_sub.connect('tcp://127.0.0.1:15577')
    token_log=(out/'tokens.jsonl').open('w') if a.record_token_state else None
    token_index=0;last_token_controller_index=-1;replay_observed=0;replay_matches=0;sent_tokens=deque(maxlen=10)
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    foot_ids={model.body('left_ankle_roll_link').id,model.body('right_ankle_roll_link').id};ground_since=None
    started=time.monotonic();phase='loading';phase_start=started;targets=None;orientations=None;baseline=None;upper_baseline=None;upper_command=None;upper_velocity=None;trial_start=None;records=[];correction=np.zeros((3,3));startup_joint_correction=np.zeros(7);reach_joint_correction=np.zeros(7);command_targets=None;hold_history=deque(maxlen=50);ready=False;last_publish=0;last_save=0;command_id=None;previous_policy_target=None;stage_count=1;stage_index=0;stage_start=None;stage_history=deque(maxlen=200);dynamic_joint_goal=None
    def local_points():
        root_R=data.xmat[pelvis].reshape(3,3)
        world=np.array([data.xpos[i]+data.xmat[i].reshape(3,3)@off for i,off in zip(ids,offsets)])
        return (world-data.xpos[pelvis])@root_R,world
    def begin_reach(now,actual):
        nonlocal baseline,upper_baseline,dynamic_joint_goal,trial_start,phase,phase_start,stage_index,stage_start,stopped
        if a.sim_online_arm_ik:
            from scipy.optimize import least_squares
            baseline=actual.copy()
            upper_baseline=np.array(data.qpos[upper_qadr])
            start_q=np.array(data.qpos[qadr])
            desired=baseline[1]+delta
            scratch=mujoco.MjData(model)
            scratch.qpos[:]=data.qpos
            lower=arm_limits[:,0]+.02;upper=arm_limits[:,1]-.02
            def hand_point(q):
                scratch.qpos[qadr]=q
                mujoco.mj_forward(model,scratch)
                hand=scratch.xpos[ids[1]]+scratch.xmat[ids[1]].reshape(3,3)@offsets[1]
                return (hand-scratch.xpos[pelvis])@scratch.xmat[pelvis].reshape(3,3)
            solved=least_squares(lambda q:np.r_[hand_point(q)-desired,.003*(q-start_q)],
                                 np.clip(start_q,lower,upper),bounds=(lower,upper),max_nfev=120)
            error=float(np.linalg.norm(hand_point(solved.x)-desired))
            change=float(np.max(np.abs(solved.x-start_q)))
            max_penetration=0.
            for fraction in np.linspace(0,1,11):
                hand_point(start_q+fraction*(solved.x-start_q))
                max_penetration=max([max_penetration]+[max(0.,-float(contact.dist)) for contact in scratch.contact
                    if model.geom_bodyid[contact.geom1] in right_arm_body_ids or model.geom_bodyid[contact.geom2] in right_arm_body_ids])
            atomic(out/'online_ik.json',{'simulation_only':True,'start_q_rad':start_q.tolist(),
                'goal_q_rad':solved.x.tolist(),'target_local_m':desired.tolist(),
                'position_error_cm':error*100,'max_joint_change_rad':change,
                'max_planned_arm_contact_penetration_mm':max_penetration*1000})
            if max_penetration>.001:
                phase='online_ik_collision';stopped=True;return
            if not solved.success or error>.005 or change>.15:
                phase='online_ik_failed';stopped=True;return
            dynamic_joint_goal=solved.x
        trial_start=now;phase='reach';phase_start=now;stage_index=1;stage_start=now
        if replay is not None:pub.send(build_command_message(False,False,False))
    if a.hold_initial_vr:
        targets=local_points()[0].copy()
        root_inv=data.xquat[pelvis].copy();root_inv[1:]*=-1
        orientations=[]
        for i in ids:
            q=np.zeros(4);mujoco.mju_mulQuat(q,root_inv,data.xquat[i]);orientations.extend(q)
    try:
      with (out/'samples.jsonl').open('w') as samples:
       while not stopped and time.monotonic()-started<a.duration:
        tick=time.monotonic()
        if startup_pose_pin:
            data.qpos[upper_qadr]=startup_upper_target
            data.qvel[upper_vadr]=0
            mujoco.mj_forward(model,data)
        env.sim_step();now=time.monotonic()
        if startup_pose_pin and '[SIM] INIT preserves measured upper-body pose' in (out/'controller.log').read_text(errors='replace'):
            startup_pose_pin=False
        actual,world=local_points()
        for _ in range(100):
            try:wire=token_sub.recv(zmq.NOBLOCK)
            except zmq.Again:break
            packet=msgpack.unpackb(wire[len(b'g1_debug'):],raw=False)
            idx=packet.get('index',-1)
            try:token=validate_token(packet.get('token_state'))
            except (ValueError,TypeError):continue
            if idx<=last_token_controller_index:continue
            last_token_controller_index=idx
            if replay is not None and phase in ('reach','complete') and sent_tokens:
                replay_observed+=1
                replay_matches+=int(any(np.max(np.abs(token-t))<1e-5 for t in sent_tokens))
            if token_log is not None:
                token_log.write(json.dumps({'simulation_only':True,'phase':phase,
                    'trial_elapsed_s':None if trial_start is None else time.monotonic()-trial_start,
                    'controller_index':idx,'token_state':token.tolist(),
                    'debug':packet})+'\n');token_log.flush()

        if phase not in ['loading','connecting','lowering','grounding'] and data.xpos[pelvis,2]<.45:
            phase='unstable';stopped=True
        if controller.poll() is not None:phase='controller_exited';stopped=True
        if phase=='loading':
            if 'G1Deploy object created successfully' in (out/'controller.log').read_text(errors='replace'):
                phase='connecting';phase_start=now
        touching=set()
        for c in data.contact:
            b1,b2=model.geom_bodyid[c.geom1],model.geom_bodyid[c.geom2]
            if b1==0 and b2 in foot_ids:touching.add(int(b2))
            if b2==0 and b1 in foot_ids:touching.add(int(b1))
        if phase=='connecting' and 'Init Done' in (out/'controller.log').read_text(errors='replace'):
            phase='lowering';phase_start=now
        if phase=='lowering':
            env.elastic_band.point[2]=max(.65,1.-.08*(now-phase_start))
            if touching==foot_ids:
                if ground_since is None:ground_since=now
                if now-ground_since>=.5:phase='grounding';phase_start=now
            else:ground_since=None
            if now-phase_start>10:phase='ground_contact_failed';stopped=True
        if phase=='grounding':
            scale=max(0.,1-(now-phase_start)/2)
            env.elastic_band.kp_pos=10000*scale;env.elastic_band.kd_pos=1000*scale
            env.elastic_band.kp_ang=1000*scale;env.elastic_band.kd_ang=10*scale
            if scale==0:
                env.elastic_band.enable=False
                pub.send(build_command_message(True,False,True));phase='standing';phase_start=now
        if phase=='standing' and now-phase_start>a.standing_seconds:
            baseline=actual.copy();targets=baseline.copy();orientations=[]
            upper_baseline=np.array(data.qpos[upper_qadr])
            if a.hold_initial_upper_body and np.max(np.abs(data.qpos[qadr]-a.initial_right_q))>.15:
                phase='initial_posture_not_held';stopped=True
            if a.absolute_right_target is not None:
                delta=np.asarray(a.absolute_right_target)-baseline[1]
                if not stopped and np.linalg.norm(delta)>(.8 if a.right_joint_goal is not None else .6):
                    phase='target_too_far';stopped=True
                if a.stage_length is not None:
                    stage_count=max(1,int(np.ceil(np.linalg.norm(delta)/a.stage_length)))
            if not stopped:
                root_inv=data.xquat[pelvis].copy();root_inv[1:]*=-1
                for i in ids:
                    q=np.zeros(4);mujoco.mju_mulQuat(q,root_inv,data.xquat[i]);orientations.extend(q)
                phase='hold';phase_start=now
        command=out/'command.json'
        if command.exists():
            msg=json.loads(command.read_text())
            if msg.get('id')!=command_id:
                command_id=msg.get('id')
                if msg.get('action')=='stop':stopped=True
                elif msg.get('action')=='reach' and phase=='hold' and ready:begin_reach(now,actual)
        if a.auto_reach and phase=='hold' and ready:
            begin_reach(now,actual)
        if phase=='hold' and now-phase_start>20 and not ready:
            phase='baseline_unsettled';stopped=True
        if phase=='reach':
            elapsed=now-stage_start;u=np.clip(elapsed/a.ramp_seconds,0,1);blend=3*u*u-2*u*u*u
            fraction=(stage_index-1+blend)/stage_count
            targets=baseline.copy();targets[1]+=delta*fraction
            if a.right_joint_goal is not None or dynamic_joint_goal is not None:
                upper_command=upper_baseline.copy();upper_velocity=np.zeros(17)
                command_start=(upper_baseline[list(right_upper_idx)] if a.sim_online_arm_ik else (startup_upper_target[list(right_upper_idx)] if (a.startup_joint_feedback or a.sim_right_arm_direct_servo) else upper_baseline[list(right_upper_idx)]))
                change=np.asarray(dynamic_joint_goal if a.sim_online_arm_ik else a.right_joint_goal)-command_start
                upper_command[list(right_upper_idx)]=command_start+change*blend
                if u<1:upper_velocity[list(right_upper_idx)]=change*6*u*(1-u)/a.ramp_seconds
            if a.stage_length is not None and elapsed>=a.ramp_seconds:
                stage_history.append((now,float(np.linalg.norm(actual[1]-targets[1])*100)))
            stage_hold=a.hold_seconds if stage_index==stage_count else a.stage_hold_seconds
            recent=[(t,e) for t,e in stage_history if t>=now-1]
            stage_settled=bool(recent) and recent[0][0]<=now-.9 and max(e for _,e in recent)<=2
            if elapsed>=a.ramp_seconds+stage_hold and (a.stage_length is None or stage_settled):
                if stage_index==stage_count:phase='complete';phase_start=now
                else:stage_index+=1;stage_start=now;stage_history.clear()
            elif a.stage_length is not None and elapsed>=a.ramp_seconds+max(stage_hold+3,5):
                phase='stage_tracking_failed';stopped=True
        if now-last_publish>=.02:
            if a.startup_joint_feedback and phase in ('standing','hold'):
                error=np.asarray(a.initial_right_q)-data.qpos[qadr]
                step=np.clip(.5*error,-.12,.12)*min(.1,now-last_publish)
                startup_joint_correction=np.clip(startup_joint_correction+step,-.6,.6)
                commanded=np.clip(np.asarray(a.initial_right_q)+startup_joint_correction,arm_limits[:,0]+.02,arm_limits[:,1]-.02)
                startup_joint_correction=commanded-np.asarray(a.initial_right_q)
                startup_upper_target[list(right_upper_idx)]=commanded
            if targets is not None:
                if a.position_feedback and (policy is None or phase=='hold') and not ((a.freeze_reach_feedback or a.right_joint_goal is not None or a.sim_online_arm_ik) and phase=='reach'):
                    correction=update_correction(correction,targets-actual,now-last_publish)
                command_targets=targets+correction
                if policy is not None and phase=='reach':
                    remain=max(0.,1.-(now-trial_start)/7.)
                    policy_obs=np.r_[data.qpos[qadr],data.qvel[vadr],actual[1]-baseline[1],targets[1]-baseline[1],remain].astype(np.float32)
                    with torch.no_grad():action=policy(features(policy_obs))[0].numpy()
                    requested=baseline[1]+np.clip(action,-1,1)*.05
                    previous=baseline[1] if previous_policy_target is None else previous_policy_target
                    step=np.clip(requested-previous,-.03*max(.02,now-last_publish),.03*max(.02,now-last_publish))
                    command_targets[1]=previous+step
                    previous_policy_target=command_targets[1].copy()
            joint_mode=a.hold_initial_upper_body or ((a.right_joint_goal is not None or a.sim_online_arm_ik) and phase in ('reach','complete'))
            joint_target=(upper_command if phase in ('reach','complete') else startup_upper_target) if joint_mode else None
            joint_velocity=(upper_velocity if phase in ('reach','complete') else np.zeros(17)) if joint_mode else None
            if (a.sim_right_arm_joint_feedback or a.sim_right_arm_cartesian_feedback) and phase in ('reach','complete') and joint_target is not None:
                if now-stage_start>=a.ramp_seconds:
                    if a.sim_right_arm_joint_feedback:
                        error=np.asarray(a.right_joint_goal)-data.qpos[qadr]
                        step=np.clip(.6*error,-.12,.12)*min(.1,now-last_publish)
                    else:
                        jacp=np.zeros((3,model.nv));jacr=np.zeros((3,model.nv))
                        mujoco.mj_jac(model,data,jacp,jacr,world[1],ids[1])
                        jac=jacp[:,vadr]
                        error_world=data.xmat[pelvis].reshape(3,3)@(targets[1]-actual[1])
                        delta_q=jac.T@np.linalg.solve(jac@jac.T+.005*np.eye(3),error_world)
                        step=np.clip(.3*delta_q,-.12*min(.1,now-last_publish),.12*min(.1,now-last_publish))
                    reach_joint_correction=np.clip(reach_joint_correction+step,-.3,.3)
                joint_target=joint_target.copy()
                nominal=joint_target[list(right_upper_idx)]
                commanded=np.clip(nominal+reach_joint_correction,arm_limits[:,0]+.02,arm_limits[:,1]-.02)
                reach_joint_correction=commanded-nominal
                joint_target[list(right_upper_idx)]=commanded
            send_vr=command_targets is not None and (not joint_mode or (a.hold_initial_vr and phase not in ('reach','complete')))
            planner_wire=build_planner_message(0,[0,0,0],[1,0,0],speed=0,
                upper_body_position=joint_target.tolist() if joint_target is not None else None,
                upper_body_velocity=joint_velocity.tolist() if joint_velocity is not None else None,
                vr_3pt_position=command_targets.ravel().tolist() if send_vr else None,
                vr_3pt_orientation=orientations if send_vr else None)
            if replay is not None and phase in ('reach','complete'):
                try:
                    external_token=replay.at(now-trial_start);wire=token_packet(external_token,token_index)
                except ValueError:phase='token_replay_exhausted';stopped=True;continue
                pub.send(wire);sent_tokens.append(external_token);token_index+=1
            else:pub.send(planner_wire)
            last_publish=now
        if now-last_save>=.05:
            if phase=='hold' and targets is not None:
                hold_history.append((now,float(np.max(np.linalg.norm(actual-targets,axis=1))*100)))
            ready=bool(phase=='hold' and settled(hold_history,now) and (not a.hold_initial_upper_body or np.max(np.abs(data.qpos[qadr]-a.initial_right_q))<=.15))
            record={'trial_elapsed_s':None if trial_start is None else now-trial_start,'requested_displacement_m':delta.tolist(),'stage_index':stage_index,'stage_count':stage_count,'feet_in_contact':len(touching),'sim_time_s':float(data.time),'ready_for_reach':ready,'virtual_support':bool(env.elastic_band.enable),'startup_pose_pinned':startup_pose_pin,'time':now,'elapsed':now-started,'phase':phase,'root_height_m':float(data.xpos[pelvis,2]),'right_joint_positions_rad':data.qpos[qadr].tolist(),'startup_joint_correction_rad':startup_joint_correction.tolist(),'reach_joint_correction_rad':reach_joint_correction.tolist(),'actual_local_m':actual.tolist(),'actual_world_m':world.tolist(),'position_feedback':a.position_feedback,'command_local_m':command_targets.tolist() if command_targets is not None else None,'correction_m':correction.tolist(),'target_local_m':targets.tolist() if targets is not None else None,'body_positions':data.xpos.tolist(),'parents':model.body_parentid.tolist(),'root_rotation':data.xmat[pelvis].reshape(3,3).tolist(),'root_position':data.xpos[pelvis].tolist()}
            record['full_qpos']=data.qpos.tolist()
            record['full_qvel']=data.qvel.tolist()
            record['right_arm_contacts']=[[names[model.geom_bodyid[c.geom1]],names[model.geom_bodyid[c.geom2]],float(c.dist)] for c in data.contact if model.geom_bodyid[c.geom1] in right_arm_body_ids or model.geom_bodyid[c.geom2] in right_arm_body_ids]
            if env.unitree_bridge.low_cmd:
                cmds=env.unitree_bridge.low_cmd.motor_cmd[22:29]
                record['right_joint_motor_command_rad']=[float(cmd.q) for cmd in cmds]
                record['right_joint_motor_kp']=[float(cmd.kp) for cmd in cmds]
                record['right_joint_motor_torque_nm']=env.torques[env.body_joint_index[22:29]-1].tolist()
                record['right_joint_bias_torque_nm']=data.qfrc_bias[vadr].tolist()
                record['right_joint_torque_limit_nm']=env.torque_limit[env.body_joint_index[22:29]-1].tolist()
            if targets is not None:record['right_error_cm']=float(np.linalg.norm(actual[1]-targets[1])*100)
            atomic(out/'state.json',record);samples.write(json.dumps(record)+'\n');samples.flush();last_save=now;records.append(record)
        if a.auto_reach and phase=='complete' and records and records[-1]['phase']=='complete':stopped=True
        time.sleep(max(0,env.sim_dt-(time.monotonic()-tick)))
    finally:
        pub.send(build_command_message(False,True,True));time.sleep(.2)
        if controller.poll() is None:
            os.killpg(controller.pid,signal.SIGTERM)
            try:controller.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(controller.pid,signal.SIGKILL);controller.wait()
        if replay is not None:
            atomic(out/'token_replay_result.json',{'simulation_only':True,'source':str(replay.path),'sent_packets':token_index,'observed_controller_tokens':replay_observed,'matching_controller_tokens':replay_matches,'controller_confirmed_external_input':'[Token Safety] ✓ First token received!' in (out/'controller.log').read_text(errors='replace')})
        if token_log is not None:token_log.close()
        token_sub.close(0)
        logfile.close();sim.close();pub.close(0);ctx.term()
        if (out/'state.json').exists():
            final_state=json.loads((out/'state.json').read_text());final_state['phase']=phase;final_state['ready_for_reach']=False;atomic(out/'state.json',final_state)
        atomic(out/'finished.json',{'phase':phase,'samples':len(records),'simulation_only':True,'virtual_support':bool(env.elastic_band.enable)})
        subprocess.run([sys.executable,str(ROOT/'summarize.py'),str(out)],check=False)
if __name__=='__main__':main()
