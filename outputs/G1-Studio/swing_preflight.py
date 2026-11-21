"""Read-only DDS check for a future physical token trial. Never publishes commands."""
import argparse, json, time, threading, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SONIC=Path('/home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl/gear_sonic_deploy')

def check_pose(q,dq,expected):
    q,dq,expected=(np.asarray(a,float) for a in (q,dq,expected))
    if any(a.shape!=(29,) or not np.isfinite(a).all() for a in (q,dq,expected)):
        raise ValueError('Expected 29 finite measured positions, velocities and reference positions')
    delta=np.abs(q-expected)
    return {'max_joint_difference_deg':float(np.degrees(delta.max())),
            'worst_joint_index':int(delta.argmax()),
            'worst_joint_name':('right elbow' if int(delta.argmax())==25 else 'joint '+str(int(delta.argmax()))),
            'max_right_arm_difference_deg':float(np.degrees(delta[22:].max())),
            'max_measured_speed_rad_s':float(np.abs(dq).max()),
            'start_pose_matches':bool(delta.max()<np.radians(5) and np.abs(dq).max()<.15)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--interface',default='enp130s0');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report={'read_only':True,'hardware_ready':False,'captured_at_unix':time.time(),'checks':{},'blockers':[]}
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        lock=threading.Lock();samples=[]
        def receive(msg):
            sample=(time.monotonic(),int(msg.tick),[float(m.q) for m in msg.motor_state[:29]],
                    [float(m.dq) for m in msg.motor_state[:29]],
                    [float(x) for x in msg.imu_state.quaternion], [float(x) for x in msg.imu_state.gyroscope])
            with lock:samples.append(sample);del samples[:-250]
        ChannelFactoryInitialize(0,a.interface)
        subscriber=ChannelSubscriber('rt/lowstate',LowState_);subscriber.Init(receive,10)
        deadline=time.monotonic()+4
        while time.monotonic()<deadline:
            time.sleep(.05)
            with lock:
                if len(samples)>20 and samples[-1][0]-samples[0][0]>.2:break
        with lock:captured=list(samples)
        subscriber.Close()
        if len(captured)<2 or time.monotonic()-captured[-1][0]>.15 or captured[-1][1]<=captured[0][1]:
            raise RuntimeError('No fresh, advancing robot LowState on selected Ethernet interface')
        _,tick,q,dq,quat,gyro=captured[-1]
        report.update(robot_tick=tick,measured_q_rad=q,measured_dq_rad_s=dq,imu_quaternion_wxyz=quat,imu_gyroscope_rad_s=gyro)
        report['checks']['fresh_joint_feedback']=True
        expected=json.loads((ROOT/'token-safety/expected_start.json').read_text())
        report['pose_comparison']=check_pose(q,dq,expected['q_rad'])
        report['checks']['start_pose_matches']=report['pose_comparison']['start_pose_matches']
        if not report['checks']['start_pose_matches']:
            report['blockers'].append('Actual robot posture differs from the simulation start; validate a gradual transition first.')
    except Exception as e:
        report['checks'].setdefault('fresh_joint_feedback',False)
        report['blockers'].append(str(e))
    proof_path=ROOT/'token-safety/build-proof.json'
    try:
        proof=json.loads(proof_path.read_text());binary=SONIC/'target/release/g1_deploy_onnx_ref'
        report['checks']['watchdog_binary_matches']=hashlib.sha256(binary.read_bytes()).hexdigest()==proof['binary_sha256']
    except (OSError,ValueError):report['checks']['watchdog_binary_matches']=False
    if not report['checks']['watchdog_binary_matches']:report['blockers'].append('Receiver watchdog build is missing or changed.')
    # Neither a GUI toggle nor a hand-edited approval flag can supply these missing checks.
    report['blockers'] += ['Real gripper and racket mounting orientation is not calibrated.',
        'A slow stroke and its start/return transitions have not passed a full-body simulation audit from this measured pose.']
    report['checks']['physical_stroke_validated']=False
    a.output.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(a.output)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
