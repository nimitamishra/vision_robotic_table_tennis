"""Read-only physical LowState/LowCmd recording; never publishes or commands."""
import argparse,json,time,threading
from pathlib import Path
from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_,LowCmd_
from unitree_sdk2py.utils.crc import CRC
crc=CRC()
p=argparse.ArgumentParser();p.add_argument('--interface',default='enp130s0');p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=60);a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True);f=a.output.open('w',buffering=1);lock=threading.Lock();count={'state':0,'command':0}
def write(row):
 with lock:f.write(json.dumps(row,separators=(',',':'))+'\n');count[row['type']]+=1
ChannelFactoryInitialize(0,a.interface)
def state(m):
 write({'type':'state','t':time.monotonic(),'tick':m.tick,'q':[x.q for x in m.motor_state[:29]],'dq':[x.dq for x in m.motor_state[:29]],'tau_est':[x.tau_est for x in m.motor_state[:29]],'mode':[x.mode for x in m.motor_state[:29]],'mode_machine':m.mode_machine,'status':[x.motorstate for x in m.motor_state[:29]]})
def command(m):
 write({'type':'command','t':time.monotonic(),'q':[x.q for x in m.motor_cmd[:29]],'mode':[x.mode for x in m.motor_cmd[:29]],'mode_machine':m.mode_machine,'crc_ok':m.crc==crc.Crc(m),'tau':[x.tau for x in m.motor_cmd[:29]],'kp':[x.kp for x in m.motor_cmd[:29]],'kd':[x.kd for x in m.motor_cmd[:29]]})
s=ChannelSubscriber('rt/lowstate',LowState_);s.Init(state,100)
c=ChannelSubscriber('rt/lowcmd',LowCmd_);c.Init(command,100)
print('Read-only physical preparation recorder started',flush=True)
try:time.sleep(a.seconds)
finally:s.Close();c.Close();f.close();print(json.dumps(count),flush=True)
