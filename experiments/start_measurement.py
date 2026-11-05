import subprocess,json
from pathlib import Path
base=Path('/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser')
out=Path('/home/jingxuan/Documents/Codex/2026-10-16/ni/outputs/g1-ball-session')
commands={
 'ball': ['/home/jingxuan/Documents/Codex/2026-09-22/wo/work/yolo_env/bin/python','-u',str(base/'repo/g1/ball_worker.py'),'--publish','tcp://127.0.0.1:5561','--model',str(base/'models/pingpong_v1.pt'),'--include-frame'],
 'measure':['/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/work/vision-venv/bin/python','-u',str(out/'measure_ball.py'),'--g1-dir',str(base/'repo/g1'),'--output',str(out),'--seconds','900']}
pids={}
for name,cmd in commands.items():
 with (out/(name+'.log')).open('a') as log:
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  pids[name]=p.pid
(out/'processes.json').write_text(json.dumps(pids));print(pids)
