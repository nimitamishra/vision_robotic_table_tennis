"""Local desktop console for the existing G1 pipeline. No services start on launch."""
import os, sys, json, time, socket, signal, subprocess, select, pty, re
from pathlib import Path
from PySide6 import QtCore as C, QtGui as G, QtWidgets as W
from PySide6 import QtNetwork as N
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings
import zmq, msgpack

ROOT=Path(__file__).resolve().parent
BASE=Path('/home/jingxuan/Documents/Codex')
VISION=BASE/'2026-10-02/zhzeze/outputs/G1-Viser'
G1=VISION/'repo/g1'
PY=BASE/'2026-10-02/zhzeze/work/vision-venv/bin/python'
YOLO=BASE/'2026-09-22/wo/work/yolo_env/bin/python'
SIM_PY=BASE/'2026-09-15/nih/work/GR00T-WholeBodyControl/.venv_sim/bin/python'
SONIC=BASE/'2026-09-15/nih/work/GR00T-WholeBodyControl/gear_sonic_deploy'
OLD=ROOT.parent
RUN=ROOT/'runtime'; RUN.mkdir(exist_ok=True)
RECORDINGS=OLD/'ball-recorder/recordings'

def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0)); return s.getsockname()[1]

class Process:
    def __init__(self,name,argv,cwd=None,terminal=False):
        self.name=name; self.path=RUN/(name+'.log'); self.fd=None
        self.log=self.path.open('ab',buffering=0)
        env=os.environ.copy()
        # Qt's bundled library search paths must not leak into CUDA subprocesses.
        for key in ('QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH'):
            env.pop(key,None)
        if terminal:
            self.fd,slave=pty.openpty()
            self.p=subprocess.Popen(argv,cwd=cwd,stdin=slave,stdout=slave,stderr=slave,start_new_session=True,env=env)
            os.close(slave)
        else:
            self.p=subprocess.Popen(argv,cwd=cwd,stdin=subprocess.DEVNULL,stdout=self.log,stderr=self.log,start_new_session=True,env=env)
    def alive(self): return self.p.poll() is None
    def pump(self):
        if self.fd is not None:
            for _ in range(20):
                if not select.select([self.fd],[],[],0)[0]: break
                try: data=os.read(self.fd,65536)
                except OSError: break
                if not data: break
                self.log.write(data)
    def key(self,key):
        if self.alive() and self.fd is not None: os.write(self.fd,key.encode())
    def stop(self):
        if self.alive(): os.killpg(self.p.pid,signal.SIGINT if self.name=='record' else signal.SIGTERM)
    def close(self):
        if self.fd is not None: os.close(self.fd); self.fd=None
        self.log.close()

class Streams(C.QThread):
    packet=C.Signal(str,object)
    error=C.Signal(str)
    def __init__(self,host,port,camera_port=5560): super().__init__(); self.host=host; self.port=port; self.camera_port=camera_port
    def run(self):
        ctx=zmq.Context(); poll=zmq.Poller(); sockets={}; last={}
        try:
            for name,url in [('camera',f'tcp://{self.host}:{self.camera_port}'),('detection',f'tcp://127.0.0.1:{self.port}')]:
                s=ctx.socket(zmq.SUB); s.setsockopt(zmq.SUBSCRIBE,b''); s.setsockopt(zmq.CONFLATE,1);s.connect(url);poll.register(s,zmq.POLLIN);sockets[s]=name
            while not self.isInterruptionRequested():
                for s,_ in poll.poll(100):
                    d=msgpack.unpackb(s.recv(),raw=False);name=sockets[s];now=time.monotonic()
                    if now-last.get(name,0)>.1: self.packet.emit(name,d);last[name]=now
        except Exception as e: self.error.emit(str(e))
        finally:
            for s in sockets:s.close(0)
            ctx.term()

class Main(W.QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('G1 Studio');self.resize(1440,920)
        self.procs={};self.stream=None;self.seen={};self.images={};self.pending=False;self.loaded=False;self.reported=set()
        self.stop_latched=False;self.stop_requested_at=None
        self.camera_port=free_port();self.ball_port=free_port();self.measure_port=free_port();self.viser_port=free_port();self.trajectory_port=free_port();self.trajectory_pub=free_port()
        self.session=RUN/time.strftime('session-%Y%m%d-%H%M%S');self.session.mkdir(exist_ok=True)
        self.sonic_heartbeat=self.session/'sonic-heartbeat'
        outer=W.QWidget();self.setCentralWidget(outer);layout=W.QVBoxLayout(outer);layout.setContentsMargins(22,18,22,18)
        title=W.QLabel('G1 Studio  /  Robot & Vision');title.setObjectName('title');layout.addWidget(title)
        safety=W.QHBoxLayout();layout.addLayout(safety)
        self.stop_button=self.button('STOP SONIC · Esc',self.emergency_stop,safety)
        self.stop_button.setStyleSheet('QPushButton{background:#b91c1c;color:white;font-size:22px;font-weight:700;padding:16px;border:2px solid #f87171} QPushButton:hover{background:#dc2626}')
        self.stop_status=W.QLabel('Software stop · Esc works while this app is focused. Keep the physical emergency stop available.')
        self.stop_status.setWordWrap(True);safety.addWidget(self.stop_status,1)
        self.button('Reset Stop Latch',self.reset_stop,safety)
        self.stop_shortcut=G.QShortcut(G.QKeySequence('Esc'),self);self.stop_shortcut.setContext(C.Qt.ApplicationShortcut);self.stop_shortcut.activated.connect(self.emergency_stop)
        W.QApplication.instance().installEventFilter(self)
        bar=W.QHBoxLayout();layout.addLayout(bar)
        self.host=W.QLineEdit('192.168.123.164');self.host.setMaximumWidth(175)
        bar.addWidget(W.QLabel('Robot IP'));bar.addWidget(self.host)
        self.connect_btn=self.button('Connect / Start Vision',self.start_vision,bar)
        self.button('Stop Vision',self.stop_vision,bar)
        self.button('Start Camera Service',self.remote_camera,bar)
        bar.addStretch();self.state=W.QLabel('Disconnected');bar.addWidget(self.state)
        self.tabs=W.QTabWidget();layout.addWidget(self.tabs,1)
        overview=W.QWidget();v=W.QVBoxLayout(overview)
        self.summary=W.QLabel('Ready. Connect the robot, then select Connect / Start Vision.');self.summary.setWordWrap(True);v.addWidget(self.summary)
        split=W.QSplitter();v.addWidget(split,1)
        self.camera=self.image_panel('Stereo Camera · Left / Right',split)
        self.detect=self.image_panel('YOLO Detection · Left / Right',split)
        self.metrics=W.QLabel('Waiting for camera and detections');self.metrics.setWordWrap(True);v.addWidget(self.metrics)
        self.tabs.addTab(overview,'Camera && Detection')
        self.web=QWebEngineView();self.web.settings().setAttribute(QWebEngineSettings.WebGLEnabled,True)
        self.web.setHtml('<html style="background:#101a29;color:#d8e4f5;font:20px sans-serif"><body><h2>Viser 3D View</h2><p>Connect the camera to load the robot, table and ball positions.</p></body></html>')
        self.tabs.addTab(self.web,'Viser 3D View')
        traj=W.QWidget();tv=W.QVBoxLayout(traj);tr=W.QHBoxLayout();tv.addLayout(tr)
        self.button('Start Trajectory',self.start_trajectory,tr)
        self.button('Stop Trajectory',self.stop_trajectory,tr);tr.addStretch()
        tv.addWidget(W.QLabel('Single ball · velocity + gravity forecast · no robot motion commands'))
        self.shadow_status=W.QLabel('Policy preview: waiting for trajectory · read-only')
        self.shadow_status.setFixedHeight(26)
        self.shadow_status.setToolTip('The policy preview never sends a command to SONIC.')
        tv.addWidget(self.shadow_status)
        self.shadow_run=None
        self.trajectory_web=QWebEngineView();tv.addWidget(self.trajectory_web,1)
        self.trajectory_web.setHtml('<body style="background:#101a29;color:#d8e4f5;font:18px sans-serif">Start Vision, then Start Trajectory. Predictions stop at the contact-height plane.</body>')
        self.trajectory_loaded=False
        self.tabs.addTab(traj,'Trajectory')
        sonic=W.QWidget();s=W.QVBoxLayout(sonic)
        info=W.QLabel('SONIC runs locally and controls G1 over Ethernet. Starting moves the robot toward its default pose; enable policy control separately.\nPrepare fall protection and the physical emergency stop. Software exit does not replace the physical emergency stop.');info.setWordWrap(True);s.addWidget(info)
        row=W.QHBoxLayout();s.addLayout(row);row.addWidget(W.QLabel('Robot Interface'))
        self.nic=W.QComboBox();self.nic.addItems([n for _,n in socket.if_nameindex() if n!='lo']);self.nic.setCurrentText('enp130s0');row.addWidget(self.nic)
        self.button('Start SONIC (Real Robot)',self.start_sonic,row);self.button('Slow Preparation · 5° out and back',lambda:self.start_sonic(preparation=True),row);self.button('Exit SONIC · O',lambda:self.sonic_key('o'),row);row.addStretch()
        controls=W.QHBoxLayout();s.addLayout(controls)
        for text,key in [('Enable Control · ]',']'),('Toggle Planner · Enter','\n'),('Slow Walk · 1','1'),('Planner: Stop Motion · R','r')]:
            self.button(text,lambda checked=False,k=key:self.sonic_key(k),controls)
        s.addWidget(W.QLabel('Keys are sent only to SONIC. R has a different meaning in reference-motion mode; check the current mode in the log.'))
        keyrow=W.QHBoxLayout();s.addLayout(keyrow)
        self.keyinput=W.QLineEdit();self.keyinput.setPlaceholderText('One control key: w / s / , / . (releasing a key does not immediately stop motion)')
        keyrow.addWidget(self.keyinput);self.button('Send Key',self.send_key,keyrow)
        self.soniclog=W.QPlainTextEdit();self.soniclog.setReadOnly(True);s.addWidget(self.soniclog,1)
        self.tabs.addTab(sonic,'SONIC Control')
        self.sonic_tab_index=self.tabs.indexOf(sonic)
        token_warning=W.QLabel('Physical token trial: run the read-only check first. The trained stroke is locked until the actual starting pose, racket mount and slow transitions are validated.');token_warning.setWordWrap(True);s.insertWidget(1,token_warning)
        trial=W.QHBoxLayout();s.insertLayout(2,trial)
        self.button('Check Robot for Slow Swing (Read Only)',self.check_swing,trial)
        self.button('Preview Trained Swing in Simulation',self.preview_swing,trial)
        self.swing_play=self.button('Play Physical Swing · Not Ready',lambda:None,trial)
        self.swing_play.setEnabled(False);self.swing_play.setToolTip('No physical token publisher is enabled. A measured-pose transition and racket audit are required.')
        self.swing_status=W.QLabel('No current swing check. Checking reads robot joints only; it sends no motion commands.')
        self.swing_status.setWordWrap(True);self.swing_status.setFixedHeight(92);s.insertWidget(3,self.swing_status)
        self.swing_report=self.session/'swing-preflight.json';self.swing_check_pending=False
        token=W.QWidget();tl=W.QVBoxLayout(token);tr=W.QHBoxLayout();tl.addLayout(tr)
        self.token_stage=W.QComboBox();self.token_stage.addItems(['Short reach','Ball contact','Front table tennis','Forehand motion','Forehand return','Hitting plane · Direct bracket'])
        if (OLD/'sonic-token-policy/hitting-plane-preview.json').exists():self.token_stage.setCurrentText('Hitting plane · Direct bracket')
        elif (OLD/'sonic-token-policy/return-preview.json').exists():self.token_stage.setCurrentText('Forehand return')
        elif (OLD/'sonic-token-policy/forehand-preview.json').exists():self.token_stage.setCurrentText('Forehand motion')
        elif (OLD/'sonic-token-policy/front-deployment.json').exists():self.token_stage.setCurrentText('Front table tennis')
        elif (OLD/'sonic-token-policy/impact-deployment.json').exists():self.token_stage.setCurrentText('Ball contact')
        tr.addWidget(self.token_stage)
        self.button('Start Token Simulation',self.start_token_sim,tr)
        self.button('Open Simulation Replay',self.open_simulation_replay,tr)
        self.button('Stop Token Simulation',self.stop_token_sim,tr);tr.addStretch()
        tl.addWidget(W.QLabel('Learned 64D token → frozen SONIC decoder → MuJoCo · simulation only'))
        self.token_web=QWebEngineView();tl.addWidget(self.token_web,1)
        self.token_web.setHtml('<body style="background:#101a29;color:#d8e4f5;font:18px sans-serif">Select Hitting plane · Direct bracket, then Start Token Simulation. Enter a target and time, then Run strike. Use Open Simulation Replay if WebGL is unavailable.</body>')
        self.token_tab_index=self.tabs.addTab(token,'Token Policy Sim')
        self.token_port=free_port();self.token_loaded=False;self.token_requested=False
        record=W.QWidget();r=W.QVBoxLayout(record)
        r.addWidget(W.QLabel('Record original stereo images, matched detection messages and timestamps for review and training.'))
        rr=W.QHBoxLayout();r.addLayout(rr);self.seconds=W.QSpinBox();self.seconds.setRange(5,600);self.seconds.setValue(30);self.seconds.setSuffix(' s');rr.addWidget(self.seconds)
        self.button('Start Recording (5s Countdown)',self.record,rr);self.button('Stop && Save',self.end_record,rr);self.button('Open Recordings Folder',lambda:G.QDesktopServices.openUrl(C.QUrl.fromLocalFile(str(RECORDINGS))),rr);rr.addStretch()
        self.recordlog=W.QPlainTextEdit();self.recordlog.setReadOnly(True);r.addWidget(self.recordlog)
        self.button('Open Previous Multi-ball Comparison',lambda:G.QDesktopServices.openUrl(C.QUrl.fromLocalFile(str(OLD/'ball-recorder/multiball-review/before-after.mp4'))),r)
        r.addWidget(W.QLabel('Live 3D detection uses the existing single-ball pipeline. Previous multi-ball results are available as offline replay.'))
        self.tabs.addTab(record,'Recording && Replay')
        logs=W.QWidget();l=W.QVBoxLayout(logs);self.logpick=W.QComboBox();self.logpick.addItems(['relay','worker','measure','viser','trajectory','shadow','camera-ssh','record','sonic','token-sim','swing-check']);l.addWidget(self.logpick)
        self.logtext=W.QPlainTextEdit();self.logtext.setReadOnly(True);l.addWidget(self.logtext);self.tabs.addTab(logs,'Logs')
        self.statusBar().showMessage('Opening this app does not start SONIC or send motion commands.')
        self.timer=C.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(400)
        self.safety_timer=C.QTimer(self);self.safety_timer.timeout.connect(self.safety_tick);self.safety_timer.start(150)
        self.setStyleSheet('''QWidget{background:#101a29;color:#e0e9f5;font-size:14px} QLabel#title{font-size:27px;font-weight:650;padding-bottom:12px} QPushButton{background:#233b55;border:1px solid #365371;border-radius:6px;padding:10px} QPushButton:hover{background:#325777} QPushButton:disabled{background:#182230;color:#768393;border-color:#303a48} QLineEdit,QComboBox,QSpinBox{background:#19293d;padding:7px;border:1px solid #365371;border-radius:4px} QTabBar::tab{padding:13px 23px;background:#17263a} QTabBar::tab:selected{background:#295575} QPlainTextEdit{background:#0b1320;font-family:monospace;font-size:12px;border:1px solid #273c55}''')
    def button(self,text,fn,layout):
        b=W.QPushButton(text);b.clicked.connect(fn);layout.addWidget(b);return b
    def image_panel(self,title,parent):
        widget=W.QWidget();v=W.QVBoxLayout(widget);v.addWidget(W.QLabel(title));label=W.QLabel('Waiting for data');label.setAlignment(C.Qt.AlignCenter);label.setMinimumSize(320,240);v.addWidget(label,1);parent.addWidget(widget);return label
    def notice(self,text): self.statusBar().showMessage(text);self.summary.setText(text)
    def launch(self,name,args,cwd=None,terminal=False):
        old=self.procs.get(name)
        if old and old.alive():return old
        if old:old.close()
        self.reported.discard(name)
        try:self.procs[name]=Process(name,[str(a) for a in args],cwd,terminal);return self.procs[name]
        except Exception as e:self.notice(f'{name} failed to start：{e}');return None
    def alive(self,name):return name in self.procs and self.procs[name].alive()
    def eventFilter(self,obj,event):
        if event.type()==C.QEvent.KeyPress and event.key()==C.Qt.Key_Escape:
            self.emergency_stop();return True
        return super().eventFilter(obj,event)
    def heartbeat(self):
        tmp=self.sonic_heartbeat.with_suffix('.tmp');tmp.write_text(str(time.monotonic()));tmp.replace(self.sonic_heartbeat)
    def emergency_stop(self):
        self.stop_latched=True
        if self.stop_requested_at is None:self.stop_requested_at=time.monotonic()
        # Only owned token senders may be stopped here. The simulation viewer
        # has no robot transport and is deliberately independent.
        for name in ('token-live','token-publisher'):
            if self.alive(name):self.procs[name].stop()
        if self.alive('sonic'):
            self.procs['sonic'].key('o')
            self.stop_status.setText('STOP REQUESTED — waiting for SONIC damping exit. If motion continues, use the physical emergency stop.')
        else:self.stop_status.setText('STOP LATCHED — no SONIC process owned by this app is running. This does not confirm external controllers are stopped.')
    def reset_stop(self):
        if self.alive('sonic'):
            self.stop_status.setText('Cannot reset while SONIC is still running. Use the physical emergency stop if needed.');return
        self.stop_latched=False;self.stop_requested_at=None
        self.stop_status.setText('Stop latch reset. Controller remains off; start it explicitly when ready.')
    def safety_tick(self):
        self.update_swing_check()
        if self.alive('sonic'):
            self.heartbeat()
            if self.stop_latched:
                self.procs['sonic'].key('o')
                if self.stop_requested_at is not None and time.monotonic()-self.stop_requested_at>2:
                    self.stop_status.setText('STOP UNCONFIRMED — SONIC has not exited. USE THE PHYSICAL EMERGENCY STOP.')
        elif self.stop_latched and self.stop_requested_at is not None:
            self.stop_status.setText('STOP LATCHED — managed SONIC process is off. Check the robot; process exit does not verify motor state.')
    def check_swing(self):
        if self.alive('swing-check'):return
        self.swing_report.unlink(missing_ok=True)
        if self.launch('swing-check',[SIM_PY,ROOT/'swing_preflight.py','--interface',self.nic.currentText(),'--output',self.swing_report]):
            self.swing_check_pending=True
            self.swing_status.setText('Reading current robot joints. No motion commands are sent.')
    def preview_swing(self):
        self.token_stage.setCurrentText('Hitting plane · Direct bracket' if (OLD/'sonic-token-policy/hitting-plane-preview.json').exists() else 'Forehand return');self.start_token_sim()
    def open_simulation_replay(self):
        manifest=OLD/'sonic-token-policy/hitting-plane-preview.json'
        if not manifest.exists():self.notice('Simulation evaluation is not ready.');return
        replay=Path(json.loads(manifest.read_text())['run'])/'index.html'
        if not replay.exists():self.notice('Replay has not been rendered yet.');return
        self.token_requested=False
        if not self.alive('sim-replay'):
            self.replay_port=free_port()
            self.launch('sim-replay',[SIM_PY,OLD/'sonic-token-policy/hitting_plane/app_server.py','--run',str(replay.parent),'--port',str(self.replay_port)])
        self.replay_requested=True
        self.token_web.setHtml('<body style="background:#101a29;color:white;font:18px sans-serif">Starting simulation and replay…</body>')
        self.tabs.setCurrentIndex(self.token_tab_index)
    def update_swing_check(self):
        if not self.swing_check_pending or self.alive('swing-check'):return
        self.swing_check_pending=False
        try:
            report=json.loads(self.swing_report.read_text())
            pose=report.get('pose_comparison',{})
            prefix=(f"Read-only check: maximum start-pose difference {pose['max_joint_difference_deg']:.1f} degrees ({pose.get('worst_joint_name', 'joint '+str(pose['worst_joint_index']))}). " if pose else 'Read-only check could not compare the current posture. ')
            self.swing_status.setText(prefix+'Physical playback remains locked. '+ ' '.join(report['blockers']))
        except (OSError,ValueError,KeyError):
            self.swing_status.setText('Robot check failed. Physical playback remains locked; see swing-check log.')
    def start_token_sim(self):
        stage=self.token_stage.currentText()
        manifest_name,viewer={'Short reach':('deployment.json','serve.py'),
            'Ball contact':('impact-deployment.json','serve_impact.py'),
            'Front table tennis':('front-deployment.json','serve_front.py'),
            'Forehand motion':('forehand-preview.json','forehand/serve.py'),
            'Forehand return':('return-preview.json','forehand/serve_return.py'),
            'Hitting plane · Direct bracket':('hitting-plane-preview.json','hitting_plane/serve.py')}[stage]
        manifest=OLD/'sonic-token-policy'/manifest_name
        if not manifest.exists():
            self.notice('Token policy training/evaluation is not complete.');return
        if self.alive('token-sim'):
            self.token_requested=True;self.token_loaded=False;self.replay_requested=False
            self.tabs.setCurrentIndex(self.token_tab_index);return
        self.replay_requested=False
        self.token_requested=True;self.token_loaded=False
        args=[SIM_PY,OLD/'sonic-token-policy'/viewer,'--port',str(self.token_port),'--autoplay']
        if stage=='Hitting plane · Direct bracket':args += ['--run',json.loads(manifest.read_text())['run']]
        self.launch('token-sim',args)
        self.token_stage.setEnabled(False)
        self.tabs.setCurrentIndex(self.token_tab_index)
    def stop_token_sim(self):
        self.token_requested=False
        self.token_stage.setEnabled(True)
        if self.alive('token-sim'):self.procs['token-sim'].stop()
        self.token_loaded=False
        self.token_web.setHtml('<body style="background:#101a29;color:#d8e4f5;font:18px sans-serif">Token simulation stopped.</body>')
    def start_vision(self):
        if any(self.alive(n) for n in ('worker','measure','viser','relay')) and not self.stream:self.notice('Vision services are stopping. Please reconnect shortly.');return
        if self.stream:self.notice('Vision is connected or waiting for the camera. Check status and logs.');return
        host=self.host.text().strip()
        try:socket.inet_aton(host)
        except OSError:self.notice('Enter a valid robot IPv4 address.');return
        self.host.setEnabled(False);self.seen.clear()
        self.launch('relay',[PY,'-u',ROOT/'camera_relay.py','--camera',f'tcp://{host}:5560','--publish',f'tcp://127.0.0.1:{self.camera_port}'])
        self.stream=Streams('127.0.0.1',self.ball_port,self.camera_port);self.stream.packet.connect(self.receive);self.stream.error.connect(self.notice);self.stream.start()
        self.pending=True;self.wait_started=time.monotonic();self.notice('Waiting for camera data. Select Start Camera Service if it is not running.')
    def remote_camera(self):
        # SSH runs in a visible terminal; the password remains in SSH, never in app config/logs.
        host=self.host.text().strip()
        try:socket.inet_aton(host)
        except OSError:self.notice('Invalid IP address');return
        if self.alive('camera-ssh'):self.notice('Camera SSH window is already open');return
        remote="pgrep -f '^/usr/bin/python3 -u /home/unitree/codex_camera/stereo_pose_server.py' >/dev/null || (nohup /usr/bin/python3 -u /home/unitree/codex_camera/stereo_pose_server.py --fps 60 --width 480 --height 270 >/tmp/codex-stereo-pose-server.log 2>&1 </dev/null &); pgrep -f '^/home/unitree/codex_camera/.venv/bin/python /home/unitree/codex_camera/joint_state_server.py( |$)' >/dev/null || (nohup env LD_LIBRARY_PATH=/usr/local/lib PYTHONPATH=/home/unitree/Projects/gr00t/external_dependencies/unitree_sdk2_python /home/unitree/codex_camera/.venv/bin/python /home/unitree/codex_camera/joint_state_server.py --port 5562 >/tmp/codex-joint-state-server.log 2>&1 </dev/null &); sleep 2; tail -n 8 /tmp/codex-stereo-pose-server.log"
        self.launch('camera-ssh',['gnome-terminal','--wait','--title=G1 Camera Service','--','bash','-c','ssh -o ConnectTimeout=5 "$1" "$2"; printf "\\nPress Enter to close..."; read -r _','g1-camera',f'unitree@{host}',remote])
        self.notice('Camera SSH terminal opened. Enter the robot password when prompted. The camera service sends no motion commands.')
    def receive(self,name,d):
        self.seen[name]=time.monotonic()
        jpg=d.get('jpg' if name=='camera' else 'preview_jpg')
        if jpg:
            im=G.QImage.fromData(jpg)
            if not im.isNull():self.images[name]=im;self.show_image(name)
        if name=='camera':self.camera_id=d.get('left_id')
        else:
            boxes=d.get('boxes',[[],[]]);self.metrics.setText(f"Detection frame {d.get('frame_id')}  |  L {len(boxes[0])} / R {len(boxes[1])} candidates  |  Inference {d.get('model_ms',0):.1f} ms  |  {d.get('inference_fps',0):.1f} FPS\nSee Viser for the 3D center and contour / YOLO fallback status.")
    def show_image(self,name):
        label=self.camera if name=='camera' else self.detect
        label.setPixmap(G.QPixmap.fromImage(self.images[name]).scaled(label.size(),C.Qt.KeepAspectRatio,C.Qt.SmoothTransformation))
    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'images'):
            for name in self.images:self.show_image(name)
    def start_pipeline(self):
        host=self.host.text().strip();self.pending=False
        self.launch('worker',[YOLO,'-u',G1/'ball_worker.py','--camera',f'tcp://127.0.0.1:{self.camera_port}','--publish',f'tcp://127.0.0.1:{self.ball_port}','--model',VISION/'models/pingpong_v1.pt','--include-frame'])
        self.launch('measure',[PY,'-u',OLD/'g1-ball-session/measure_ball.py','--g1-dir',G1,'--output',self.session,'--input',f'tcp://127.0.0.1:{self.ball_port}','--publish',f'tcp://127.0.0.1:{self.measure_port}'])
        self.launch('viser',[PY,'-u',G1/'stereo_viser.py','--host',host,'--camera-host','127.0.0.1','--stream-port',str(self.camera_port),'--port',str(self.viser_port),'--ball-port',str(self.measure_port)])
        self.notice('Camera connected. Starting YOLO, ball localization and Viser.')
    def start_trajectory(self):
        if not self.alive('measure'):self.notice('Start Vision before starting Trajectory.');return
        self.trajectory_loaded=False
        trajectory=self.launch('trajectory',[PY,'-u',OLD/'ball-trajectory/run.py','--input',f'tcp://127.0.0.1:{self.measure_port}','--publish',f'tcp://127.0.0.1:{self.trajectory_pub}','--viser-port',str(self.trajectory_port),'--output',OLD/'ball-trajectory/runs'])
        if trajectory and not self.alive('shadow'):
            self.shadow_run=self.session/f'policy-shadow-{time.monotonic_ns()}'
            self.launch('shadow',[SIM_PY,'-u',OLD/'rl-interception/live_shadow.py','--trajectory',f'tcp://127.0.0.1:{self.trajectory_pub}','--pose','tcp://127.0.0.1:5565','--output',self.shadow_run])
    def stop_trajectory(self):
        if self.alive('shadow'):self.procs['shadow'].stop()
        if self.alive('trajectory'):self.procs['trajectory'].stop()
        self.trajectory_loaded=False
        self.shadow_status.setText('Policy preview: stopped · read-only')
    def check_trajectory_view(self):
        if self.alive('trajectory') and not self.trajectory_loaded:
            try:
                with socket.create_connection(('127.0.0.1',self.trajectory_port),timeout=.05):pass
                self.trajectory_web.load(C.QUrl(f'http://127.0.0.1:{self.trajectory_port}'));self.trajectory_loaded=True
            except OSError:pass
    def stop_vision(self):
        if self.alive('record'):self.notice('Stop recording and wait for export to finish before stopping vision.');return
        self.pending=False
        self.stop_trajectory()
        for name in ['viser','measure','worker','relay']:
            if self.alive(name):self.procs[name].stop()
        if self.stream:self.stream.requestInterruption();self.stream.wait(1500);self.stream=None
        self.host.setEnabled(True);self.loaded=False;self.seen.clear();self.images.clear()
        self.camera.clear();self.camera.setText('Stopped');self.detect.clear();self.detect.setText('Stopped')
        self.notice('Vision stop requested. SONIC and the robot camera service remain unchanged.')
    def start_sonic(self,checked=False,preparation=False):
        if self.stop_latched:self.notice('Stop is latched. Reset Stop Latch before starting SONIC.');return
        if self.alive('sonic'):self.notice('SONIC is already running in this app');return
        # Reject an already-running external controller instead of launching a second controller.
        for p in Path('/proc').iterdir():
            if p.name.isdigit():
                try:cmd=(p/'cmdline').read_bytes().split(b'\0')
                except OSError:continue
                if cmd and Path(os.fsdecode(cmd[0])).name=='g1_deploy_onnx_ref':self.notice('An external SONIC controller is running. Exit it in its original window before starting another.');return
        if preparation:
            self.preparation_log=self.session/('preparation-'+time.strftime('%H%M%S')+'.jsonl')
            if not self.launch('prep-observer',[SIM_PY,ROOT/'observe_preparation.py','--interface',self.nic.currentText(),'--output',self.preparation_log,'--seconds','90']):return
        wrapper=ROOT/('prepare.sh' if preparation else 'sonic.sh')
        self.heartbeat()
        if self.launch('sonic',[sys.executable,ROOT/'sonic_supervisor.py','--heartbeat',self.sonic_heartbeat,'--','bash',wrapper,self.nic.currentText()],SONIC,True):
            self.preparation_active=preparation
            if preparation:
                self.stop_status.setText('Slow preparation: measured pose hold, then 5° outward and return. STOP / Esc requests damping.')
                self.notice('Preparation starts after initialization. The trained swing is not running.');return
            self.stop_status.setText('SONIC initializing — this moves the robot toward its default pose. STOP / Esc requests damping exit.')
            self.notice('SONIC is initializing. Check the SONIC log. No enable-control key has been sent.')
    def sonic_key(self,key):
        if key.lower()=='o':self.emergency_stop();return
        if self.stop_latched:self.notice('Stop is latched; motion keys are blocked.');return
        if not self.alive('sonic'):self.notice('SONIC is not running');return
        if getattr(self,'preparation_active',False):self.notice('Preparation runs one fixed sequence. Only STOP / Esc is enabled.');return
        self.procs['sonic'].key(key)
    def send_key(self):
        k=self.keyinput.text()
        if len(k)!=1 or k not in 'wWsSaAdDeEqQrRtTnNpPoO,.1234567890[]':self.notice('Enter one valid control key');return
        self.sonic_key(k);self.keyinput.clear()
    def record(self):
        if time.monotonic()-self.seen.get('camera',0)>2:self.notice('No fresh camera frames. Recording cannot start.');return
        self.launch('record',[PY,'-u',ROOT/'record.py','--seconds',str(self.seconds.value()),'--camera',f'tcp://127.0.0.1:{self.camera_port}','--detector',f'tcp://127.0.0.1:{self.ball_port}','--output',RECORDINGS])
    def end_record(self):
        if self.alive('record'):self.procs['record'].stop();self.notice('Stopping recording and exporting video. Wait for completion in the log.')
    def update_log(self,widget,name):
        p=RUN/(name+'.log')
        if not p.exists():return
        with p.open('rb') as f:f.seek(max(0,p.stat().st_size-18000));data=f.read().decode(errors='replace')
        data=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',data)
        if widget.toPlainText()!=data:widget.setPlainText(data);widget.verticalScrollBar().setValue(widget.verticalScrollBar().maximum())
    def tick(self):
        if getattr(self,'replay_requested',False) and self.alive('sim-replay'):
            try:
                with socket.create_connection(('127.0.0.1',self.replay_port),timeout=.05):pass
                self.token_web.setUrl(C.QUrl(f'http://127.0.0.1:{self.replay_port}/'));self.replay_requested=False
            except OSError:pass
        for name,p in self.procs.items():
            p.pump()
            if not p.alive() and name not in self.reported:
                self.reported.add(name)
                if name=='sonic' and getattr(self,'preparation_active',False):
                    self.preparation_active=False
                    self.stop_status.setText('Preparation controller exited. Check measured motion and SONIC log before another movement.')
                    self.notice('Preparation finished or stopped. The recorded joint feedback determines whether it moved successfully.')
                if name=='token-sim':self.token_stage.setEnabled(True)
                if p.p.returncode not in (0,-15,-2):self.notice(f'{name} exited with code {p.p.returncode}. See Logs for details.')
        self.check_trajectory_view()
        if self.token_requested and self.alive('token-sim') and not self.token_loaded:
            try:
                with socket.create_connection(('127.0.0.1',self.token_port),timeout=.05):pass
                self.token_web.load(C.QUrl(f'http://127.0.0.1:{self.token_port}'));self.token_loaded=True
            except OSError:pass
        if self.shadow_run and self.alive('shadow'):
            latest=self.shadow_run/'latest.json'
            if latest.exists():
                try:
                    preview=json.loads(latest.read_text())
                    status=preview.get('status','waiting')
                    mismatch=' · arm pose differs from training' if preview.get('training_pose_mismatch') else ''
                    self.shadow_status.setText(f'Policy preview: {status}{mismatch} · read-only')
                except (OSError,ValueError):pass
        now=time.monotonic();camera=now-self.seen.get('camera',0)<2;det=now-self.seen.get('detection',0)<2
        self.state.setText(f"Camera {'Online' if camera else 'Offline'}  ·  Detection {'Online' if det else 'No fresh data'}  ·  SONIC {'Process running' if self.alive('sonic') else 'Stopped'}")
        if self.pending and camera:
            self.start_pipeline()
            if '--trajectory' in sys.argv:self.start_trajectory()
        elif self.pending and now-self.wait_started>10:self.notice('No camera data yet. Check Ethernet and robot power, or start the camera service. Still waiting.');self.wait_started=now
        if self.alive('viser') and not self.loaded:
            try:
                with socket.create_connection(('127.0.0.1',self.viser_port),timeout=.05):pass
                self.web.load(C.QUrl(f'http://127.0.0.1:{self.viser_port}'));self.loaded=True
            except OSError:pass
        for name,fresh in [('camera',camera),('detection',det)]:
            if not fresh and name in self.images:
                label=self.camera if name=='camera' else self.detect
                label.clear();label.setText('Stream unavailable · waiting for fresh frames')
        self.update_log(self.logtext,self.logpick.currentText());self.update_log(self.soniclog,'sonic');self.update_log(self.recordlog,'record')
    def closeEvent(self,event):
        if self.alive('sonic'):
            W.QMessageBox.information(self,'SONIC is running','Exit the controller from the SONIC tab and ensure the robot is supported before closing the app.');event.ignore();return
        if self.alive('record'):
            self.end_record();W.QMessageBox.information(self,'Saving recording','Recording stop requested. Wait for export to complete before closing.');event.ignore();return
        self.stop_vision()
        self.stop_token_sim()
        for p in self.procs.values():
            if p.alive() and p.name!='camera-ssh':
                try:p.p.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(p.p.pid,signal.SIGKILL);p.p.wait(timeout=2)
            p.close()
        event.accept()

if __name__=='__main__':
    app=W.QApplication(sys.argv)
    lock=C.QLockFile(str(RUN/'app.lock'))
    if not lock.tryLock(100):
        client=N.QLocalSocket();client.connectToServer(str(RUN/'ui.sock'))
        if client.waitForConnected(500):
            client.write(b'show-sonic' if '--sonic-panel' in sys.argv else b'show');client.waitForBytesWritten(500);client.disconnectFromServer();sys.exit(0)
        W.QMessageBox.information(None,'G1 Studio','G1 Studio is already open. Switch to the existing window.');sys.exit(0)
    window=Main();window.show()
    N.QLocalServer.removeServer(str(RUN/'ui.sock'));ui_server=N.QLocalServer()
    def activate_window():
        client=ui_server.nextPendingConnection()
        def receive_activation():
            request=bytes(client.read(64))
            if request in (b'show',b'show-sonic'):
                if request==b'show-sonic':window.tabs.setCurrentIndex(window.sonic_tab_index)
                window.showNormal();window.raise_();window.activateWindow()
            client.disconnectFromServer()
        client.readyRead.connect(receive_activation);client.disconnected.connect(client.deleteLater)
        if client.bytesAvailable():receive_activation()
    ui_server.newConnection.connect(activate_window);ui_server.listen(str(RUN/'ui.sock'))
    if '--sonic-panel' in sys.argv:window.tabs.setCurrentIndex(window.sonic_tab_index)
    if '--prepare-test' in sys.argv:C.QTimer.singleShot(800,lambda:window.start_sonic(preparation=True))
    if '--swing-check' in sys.argv:C.QTimer.singleShot(500,window.check_swing)
    if '--start-vision' in sys.argv:C.QTimer.singleShot(300,window.start_vision)
    if '--token-sim' in sys.argv:C.QTimer.singleShot(300,window.start_token_sim)
    sys.exit(app.exec())
