"""Loopback-only simulation/video UI for browsers without WebGL. No robot IO."""
import argparse,json,threading,subprocess,sys,uuid
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit
from evaluate import load_run
HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=52181);p.add_argument('--run',type=Path,default=HERE/'runs/bracket-v4');a=p.parse_args();run=a.run.resolve()
 env=load_run(run);jobs={};lock=threading.Lock()
 def simulate(key,inputs):
  try:
   r=env.run(**inputs);folder=run/'interactive'/key;folder.mkdir(parents=True)
   (folder/'interactive_00.json').write_text(json.dumps({'report':r,'frames':env.frames,'events':env.events}))
   proc=subprocess.run([sys.executable,str(HERE/'render.py'),str(folder),'--group','interactive'],capture_output=True,text=True,timeout=90)
   if proc.returncode:raise RuntimeError('Video rendering failed: '+proc.stderr[-400:])
   jobs[key]={'status':'done','report':r,'video':f'/{run.name}/interactive/{key}/interactive.mp4'}
  except Exception as exc:jobs[key]={'status':'error','error':str(exc)}
  finally:lock.release()
 class Handler(SimpleHTTPRequestHandler):
  def __init__(self,*args,**kw):super().__init__(*args,directory=str(run.parent),**kw)
  def send_json(self,status,value):
   data=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
  def do_GET(self):
   if self.path=='/':self.send_response(302);self.send_header('Location',f'/{run.name}/index.html');self.end_headers();return
   if self.path.startswith('/api/job/'):
    self.send_json(200,jobs.get(self.path.rsplit('/',1)[-1],{'status':'error','error':'Unknown job'}));return
   super().do_GET()
  def do_POST(self):
   if self.path!='/api/run':self.send_json(404,{'error':'Unknown endpoint'});return
   origin=self.headers.get('Origin')
   if origin and urlsplit(origin).netloc not in (f'127.0.0.1:{a.port}',f'localhost:{a.port}'):
    self.send_json(403,{'error':'Local UI origin required'});return
   try:
    length=int(self.headers.get('Content-Length','0'))
    if not 0<length<1024:raise ValueError('Expected small JSON input')
    inputs=json.loads(self.rfile.read(length));inputs={k:float(inputs[k]) for k in ('y_m','z_m','deadline_s')}
    if not(-.53<=inputs['y_m']<=-.33 and .078<=inputs['z_m']<=.228 and .8<=inputs['deadline_s']<=1.):raise ValueError('Target/time outside supported range')
   except (ValueError,KeyError,TypeError) as exc:self.send_json(400,{'error':str(exc)});return
   if not lock.acquire(blocking=False):self.send_json(409,{'error':'A simulation is already running'});return
   key=uuid.uuid4().hex;jobs[key]={'status':'running'}
   threading.Thread(target=simulate,args=(key,inputs),daemon=True).start();self.send_json(202,{'job':key})
 print(f'Simulation and replay UI: http://127.0.0.1:{a.port}/',flush=True)
 ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
