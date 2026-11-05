import os,signal,time,subprocess,json
from pathlib import Path
out=Path('outputs/g1-ball-session').resolve();f=out/'processes.json';d=json.loads(f.read_text())
for name in ('measure','viser'):
 try:
  cmd=Path(f'/proc/{d[name]}/cmdline').read_bytes()
  expected=b'measure_ball.py' if name=='measure' else b'stereo_viser.py'
  if expected in cmd:os.kill(d[name],signal.SIGINT)
 except (ProcessLookupError,FileNotFoundError):pass
time.sleep(2)
py='/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/work/vision-venv/bin/python';g1='/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/repo/g1'
for name,cmd in {'measure':[py,'-u',str(out/'measure_ball.py'),'--g1-dir',g1,'--output',str(out),'--seconds','1800'],'viser':[py,'-u',g1+'/stereo_viser.py','--port','8080','--ball-port','5563']}.items():
 with (out/(name+'.log')).open('a') as log:
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);d[name]=p.pid
f.write_text(json.dumps(d));print(d)
