"""Own SONIC's PTY and request its damping exit on GUI heartbeat loss.

No motor transport. No token publication. O/o is the verified global stop key
in SONIC's InterfaceManager. A hung controller cannot be stopped by this helper;
its alarm deliberately does not claim that process death would damp the robot.
"""
import argparse, os, pty, select, signal, subprocess, sys, time, tty
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--heartbeat',type=Path,required=True)
    p.add_argument('--timeout',type=float,default=1.2);p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command or not .2<=a.timeout<=3:raise SystemExit('Invalid supervisor configuration')
    def fresh():
        try:age=time.monotonic()-float(a.heartbeat.read_text())
        except (OSError,ValueError):return False
        return 0<=age<a.timeout
    if not fresh():raise SystemExit('No fresh GUI heartbeat: refusing to start SONIC')
    if os.isatty(0):tty.setraw(0)
    master,slave=pty.openpty();child=subprocess.Popen(command,stdin=slave,stdout=slave,stderr=slave,start_new_session=True)
    os.close(slave);os.set_blocking(master,False);os.set_blocking(1,False)
    stopping=None;reason=None;last_stop=0.;last_alarm=0.;stdin_open=True
    def emit(data):
        # A frozen GUI must not block the watchdog by filling its output PTY.
        if isinstance(data,str):data=data.encode()
        try:os.write(1,data)
        except (BlockingIOError,BrokenPipeError):pass
    def stop(why):
        nonlocal stopping,reason
        if stopping is None:
            stopping=time.monotonic();reason=why
            emit(f'\r\n[STOP REQUEST] {why}; sending SONIC O for damping exit.\r\n')
    def on_signal(signum,frame):stop(f'supervisor signal {signum}')
    signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
    try:
        while child.poll() is None:
            now=time.monotonic()
            if not fresh():stop('GUI heartbeat lost')
            if stopping is not None and now-last_stop>=.1:
                try:os.write(master,b'o')
                except OSError:pass
                last_stop=now
            if stopping is not None and now-stopping>2 and now-last_alarm>2:
                emit('\r\n[STOP UNCONFIRMED] SONIC has not exited. USE THE PHYSICAL EMERGENCY STOP.\r\n');last_alarm=now
            ready,_,_=select.select(([0] if stdin_open else [])+[master],[],[],.02)
            if 0 in ready:
                try:data=os.read(0,1024)
                except OSError:data=b''
                if not data:stdin_open=False;stop('GUI input closed')
                elif any(c in data for c in (b'o',b'O',b'\x1b')):stop('operator stop')
                elif stopping is None:
                    try:os.write(master,data)
                    except OSError:stop('controller input unavailable')
            if master in ready:
                try:data=os.read(master,65536)
                except OSError:data=b''
                if data:emit(data)
        # Drain the final controller messages, including its actual Stop() log.
        while select.select([master],[],[],0)[0]:
            try:data=os.read(master,65536)
            except OSError:break
            if not data:break
            emit(data)
        emit(f'\r\n[SUPERVISOR] Controller exited, code={child.returncode}. Process exit is not measured motor-state confirmation.\r\n')
    finally:os.close(master)
    return child.returncode
if __name__=='__main__':sys.exit(main())
