"""Exercise stop delivery and GUI-watchdog failure using harmless PTY children."""
import os,sys,time,tempfile,uuid
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS','--disable-gpu')
from PySide6.QtTest import QTest
from app import *
app=W.QApplication([]);w=Main();w.show()
MOCK='import os,tty;tty.setraw(0);print("MOCK_READY",flush=True)\nwhile True:\n c=os.read(0,1)\n if c==b"o":print("MOCK_DAMPING_EXIT",flush=True);break\n'

def spawn(timeout=1.2):
 w.heartbeat()
 p=Process('test-stop-'+uuid.uuid4().hex[:8],[sys.executable,str(ROOT/'sonic_supervisor.py'),'--heartbeat',str(w.sonic_heartbeat),'--timeout',str(timeout),'--',sys.executable,'-u','-c',MOCK],terminal=True)
 w.procs['sonic']=p
 end=time.monotonic()+2
 while time.monotonic()<end:
  app.processEvents();p.pump()
  if b'MOCK_READY' in p.path.read_bytes():return p
  time.sleep(.01)
 raise AssertionError('Mock failed to start')

def finished(p,events=True):
 end=time.monotonic()+3
 while p.alive() and time.monotonic()<end:
  if events:app.processEvents()
  p.pump();time.sleep(.01)
 p.pump();assert not p.alive(),p.path.read_text();assert b'MOCK_DAMPING_EXIT' in p.path.read_bytes(),p.path.read_text()
 p.close();w.procs.pop('sonic');w.reset_stop()

# Actual Esc key event while a text input has focus; latch blocks restart.
p=spawn();w.keyinput.setFocus();QTest.keyClick(w.keyinput,C.Qt.Key_Escape);assert w.stop_latched
w.start_sonic();assert w.procs['sonic'] is p
finished(p)
# Button uses the same path, without waiting for a dialog.
p=spawn();w.stop_button.click();assert w.stop_latched;finished(p)
# Deliberately freeze Qt heartbeat/event processing: independent supervisor exits.
p=spawn(timeout=.35);finished(p,events=False)
# Closing the app's PTY input is also handled as a stop request by its supervisor.
# Missing heartbeat refuses startup before the mock can run.
w.sonic_heartbeat.unlink(missing_ok=True)
r=subprocess.run([sys.executable,str(ROOT/'sonic_supervisor.py'),'--heartbeat',str(w.sonic_heartbeat),'--',sys.executable,'-u','-c',MOCK],capture_output=True,timeout=2)
assert r.returncode and b'MOCK_READY' not in r.stdout
w.tabs.setCurrentIndex(w.sonic_tab_index);app.processEvents();w.grab().save(str(ROOT/'sonic-stop-preview.png'))
w.close();print('PASS: Esc, red STOP, restart latch, GUI freeze watchdog, missing-heartbeat refusal; no robot commands.')
