"""SONIC reach experiment, loopback MuJoCo only. Never accepts a robot NIC."""
import os,sys,time,json,signal,subprocess,argparse
from pathlib import Path
from collections import deque
from checks import settled
import numpy as np
import mujoco,zmq
ROOT=Path(__file__).resolve().parent
REPO=Path('/home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl')
sys.path.insert(0,str(REPO));os.chdir(REPO)
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.simulator_factory import init_channel,SimulatorFactory
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message,build_planner_message

def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data));tmp.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--duration',type=float,default=90);ap.add_argument('--release-support',action='store_true');a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'samples.jsonl').exists():raise SystemExit('Choose a new output directory; previous samples will not be overwritten.')
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
    env.check_fall=lambda: None
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(model.nbody)]
    print('Bodies:',names,flush=True)
    ids=[model.body(n).id for n in ['left_wrist_yaw_link','right_wrist_yaw_link','torso_link']]
    pelvis=model.body('pelvis').id;offsets=np.array([[.18,-.025,0],[.18,.025,0],[0,0,.35]])
    ctx=zmq.Context();pub=ctx.socket(zmq.PUB);pub.bind('tcp://127.0.0.1:15576')
    logfile=(out/'controller.log').open('w');controller=subprocess.Popen(['bash',str(ROOT/'controller.sh')],stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    started=time.monotonic();phase='loading';phase_start=started;targets=None;orientations=None;baseline=None;trial_start=None;records=[];hold_history=deque(maxlen=50);ready=False;last_publish=0;last_save=0;command_id=None
    def local_points():
        root_R=data.xmat[pelvis].reshape(3,3)
        world=np.array([data.xpos[i]+data.xmat[i].reshape(3,3)@off for i,off in zip(ids,offsets)])
        return (world-data.xpos[pelvis])@root_R,world
    try:
      with (out/'samples.jsonl').open('w') as samples:
       while not stopped and time.monotonic()-started<a.duration:
        tick=time.monotonic();env.sim_step();now=time.monotonic()
        actual,world=local_points()
        if phase not in ['loading','connecting','standing_supported'] and data.xpos[pelvis,2]<.45:
            phase='unstable';stopped=True
        if controller.poll() is not None:phase='controller_exited';stopped=True
        if phase=='loading':
            if 'G1Deploy object created successfully' in (out/'controller.log').read_text(errors='replace'):
                phase='connecting';phase_start=now
        if phase=='connecting' and now-phase_start>1:
            pub.send(build_command_message(True,False,True));phase='standing_supported';phase_start=now
        if phase=='standing_supported' and now-phase_start>8:
            env.elastic_band.enable=not a.release_support;phase='standing';phase_start=now
        if phase=='standing' and now-phase_start>6:
            baseline=actual.copy();targets=baseline.copy();orientations=[]
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
                elif msg.get('action')=='reach' and phase=='hold' and ready:trial_start=now;phase='reach';phase_start=now
        if phase=='hold' and now-phase_start>20 and not ready:
            phase='baseline_unsettled';stopped=True
        if phase=='reach':
            elapsed=now-trial_start;u=np.clip(elapsed/2,0,1);blend=3*u*u-2*u*u*u
            targets=baseline.copy();targets[1]+=np.array([.05,0,0])*blend
            if elapsed>=7:phase='complete';phase_start=now
        if now-last_publish>=.02:
            pub.send(build_planner_message(0,[0,0,0],[1,0,0],speed=0,vr_3pt_position=targets.ravel().tolist() if targets is not None else None,vr_3pt_orientation=orientations))
            last_publish=now
        if now-last_save>=.05:
            if phase=='hold' and targets is not None:
                hold_history.append((now,float(np.max(np.linalg.norm(actual-targets,axis=1))*100)))
            ready=phase=='hold' and settled(hold_history,now)
            record={'ready_for_reach':ready,'virtual_support':not a.release_support,'time':now,'elapsed':now-started,'phase':phase,'root_height_m':float(data.xpos[pelvis,2]),'actual_local_m':actual.tolist(),'actual_world_m':world.tolist(),'target_local_m':targets.tolist() if targets is not None else None,'body_positions':data.xpos.tolist(),'parents':model.body_parentid.tolist(),'root_rotation':data.xmat[pelvis].reshape(3,3).tolist(),'root_position':data.xpos[pelvis].tolist()}
            if targets is not None:record['right_error_cm']=float(np.linalg.norm(actual[1]-targets[1])*100)
            atomic(out/'state.json',record);samples.write(json.dumps(record)+'\n');samples.flush();last_save=now;records.append(record)
        time.sleep(max(0,env.sim_dt-(time.monotonic()-tick)))
    finally:
        pub.send(build_command_message(False,True,True));time.sleep(.2)
        if controller.poll() is None:
            os.killpg(controller.pid,signal.SIGTERM)
            try:controller.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(controller.pid,signal.SIGKILL);controller.wait()
        logfile.close();sim.close();pub.close(0);ctx.term()
        if (out/'state.json').exists():
            final_state=json.loads((out/'state.json').read_text());final_state['phase']=phase;final_state['ready_for_reach']=False;atomic(out/'state.json',final_state)
        atomic(out/'finished.json',{'phase':phase,'samples':len(records),'simulation_only':True,'virtual_support':not a.release_support})
        subprocess.run([sys.executable,str(ROOT/'summarize.py'),str(out)],check=False)
if __name__=='__main__':main()
