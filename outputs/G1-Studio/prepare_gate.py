"""Read-only launch gate for the fixed five-degree preparation test."""
import argparse,hashlib,json,subprocess,time,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SONIC=Path('/home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl/gear_sonic_deploy')
p=argparse.ArgumentParser();p.add_argument('interface');a=p.parse_args()
if a.interface!='enp130s0':raise SystemExit('Preparation is configured only for the verified robot Ethernet interface enp130s0')
for proc in Path('/proc').iterdir():
 if proc.name.isdigit():
  try:cmd=(proc/'cmdline').read_bytes().split(b'\0')
  except OSError:continue
  if cmd and Path(os.fsdecode(cmd[0])).name=='g1_deploy_onnx_ref':raise SystemExit('Another SONIC controller is running')
proof=json.loads((ROOT/'trial-safety/preparation-loopback-results.json').read_text())
if proof['binary_sha256']!=hashlib.sha256((SONIC/'target/release/g1_deploy_onnx_ref').read_bytes()).hexdigest():raise SystemExit('Preparation binary changed since testing')
if {r['case'] for r in proof['results']}!={'complete','operator_stop','feedback_loss','velocity','prep_speed','prep_effort'}:raise SystemExit('Incomplete preparation tests')
if not all(r['exit_code']==0 and r['damping_latched'] and r.get('serialized_crc_valid') and r.get('preparation_kp')==40.0 for r in proof['results']):raise SystemExit('Preparation stop tests did not pass')
report=ROOT/'trial-safety/preparation-preflight.json'
subprocess.run([sys.executable,str(ROOT/'swing_preflight.py'),'--interface',a.interface,'--output',str(report)],check=True,stdout=subprocess.DEVNULL)
fresh=json.loads(report.read_text());reference=json.loads((ROOT/'trial-safety/current-robot.json').read_text())
if not fresh['checks'].get('fresh_joint_feedback') or time.time()-fresh['captured_at_unix']>5:raise SystemExit('No fresh measured joint feedback')
q=np.array(fresh['measured_q_rad']);dq=np.array(fresh['measured_dq_rad_s']);baseline=np.array(reference['measured_q_rad'])
if not np.isfinite(np.r_[q,dq]).all() or np.max(np.abs(q-baseline))>np.radians(3):raise SystemExit('Pose changed by more than 3 degrees from the checked preparation pose')
if np.max(np.abs(dq))>.1:raise SystemExit('Robot is moving; preparation refused')
print('[PREPARATION] Fresh measured pose matches the checked small-motion start. Full trained swing remains locked.',flush=True)
