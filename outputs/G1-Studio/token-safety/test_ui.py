import sys,os,json
os.environ['QT_QPA_PLATFORM']='offscreen';os.environ['QTWEBENGINE_CHROMIUM_FLAGS']='--disable-gpu'
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import *
a=W.QApplication([]);w=Main();w.show();w.tabs.setCurrentIndex(w.sonic_tab_index)
assert not w.swing_play.isEnabled()
w.swing_report.write_text((ROOT/'token-safety/preflight-current.json').read_text())
w.swing_check_pending=True;w.update_swing_check()
assert '93.9' in w.swing_status.text() and 'locked' in w.swing_status.text()
assert not w.procs
for _ in range(10):a.processEvents();time.sleep(.02)
w.grab().save(str(ROOT/'token-safety/launcher-preview.png'))
w.close();print('PASS: read-only report displayed, physical play locked, no processes started')
