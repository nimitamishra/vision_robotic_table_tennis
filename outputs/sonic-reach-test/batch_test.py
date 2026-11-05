"""Sequential isolated simulation trials; never launches physical robot control."""
import argparse,json,subprocess,sys,time,html
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SIM=Path('/home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl/.venv_sim/bin/python')
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
cases=[('left',[0,.05,0]),('right',[0,-.05,0]),('up',[0,0,.05]),('down',[0,0,-.05])]
results=[]
def save():
 (out/'results.json').write_text(json.dumps(results,indent=2))
 lines=[]
 for r in results:
  value=lambda k: '—' if k not in r else f'{r[k]:.2f}'
  lines.append(f"<tr><td>{r['trial']}</td><td>{r.get('exit_phase','error')}</td><td>{'PASS' if r.get('within_2cm_for_final_second') else 'NOT PASSED'}</td><td>{value('mean_final_error_cm')}</td><td>{value('max_final_error_cm')}</td><td>{value('min_root_height_m')}</td></tr>")
 (out/'results.html').write_text('<!doctype html><meta charset="utf-8"><title>SONIC directional reach results</title><style>body{font:18px system-ui;margin:40px;background:#101c2a;color:#eef}td,th{padding:12px;border-bottom:1px solid #456;text-align:left}</style><h1>SONIC directional reach results</h1><p>Simulation only. Each trial starts from a fresh grounded initialization. Target displacement: 5 cm, 2-second ramp, 5-second hold. Position feedback enabled.</p><p>PASS means right-hand position error stayed within 2 cm during the final second. Unsettled initial holds are recorded as failures, not retried or hidden.</p><table><tr><th>Trial</th><th>Exit</th><th>Result</th><th>Mean error (cm)</th><th>Max error (cm)</th><th>Min pelvis height (m)</th></tr>'+''.join(lines)+'</table><p>Not a validation of orientation, racket accuracy, balance recovery or fast interception.</p>')
for repeat in range(1,3):
 for name,delta in cases:
  run=out/f'{name}_{repeat}';run.mkdir();print('START',run.name,flush=True)
  with (run/'simulator.log').open('w') as log:
   code=subprocess.call([str(SIM),str(ROOT/'sim_test.py'),'--output',str(run),'--duration','90','--position-feedback','--auto-reach','--delta',*map(str,delta)],stdout=log,stderr=subprocess.STDOUT)
  report=json.loads((run/'report.json').read_text()) if (run/'report.json').exists() else {'exit_phase':'process_error'}
  report.update(trial=run.name,returncode=code);results.append(report);save()
  print('RESULT',run.name,report.get('exit_phase'),report.get('mean_final_error_cm'),flush=True)
print('DONE',out,flush=True)
