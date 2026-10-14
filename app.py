#!/usr/bin/env python3
"""Open the repository's Viser scene in a dedicated desktop window."""

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QIcon
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QLabel,
                               QMainWindow, QPushButton, QStackedWidget, QToolBar)


APP_DIR = Path(__file__).resolve().parent
VISER_SCRIPT = APP_DIR / "g1" / "stereo_viser.py"
BALL_SCRIPT = APP_DIR / "g1" / "ball_worker.py"
BALL_MODEL = APP_DIR / "models" / "pingpong_v1.pt"
LOG_PATH = Path(tempfile.gettempdir()) / f"g1-viser-{os.getuid()}.log"
MODEL_LABELS = {
    "pingpong_v1": "V1 · Baseline",
    "pingpong_v2": "V2 · Synthetic mix",
    "pingpong_v3": "V3 · Negative samples",
    "pingpong_v4": "V4 · No hand-held balls",
    "pingpong_v5": "V5 · New sessions",
    "public_pingpong_yolo11n": "Public pretrained model",
}


def find_python():
    for candidate in (APP_DIR / ".venv" / "bin" / "python",):
        if candidate.is_file():
            return candidate
    raise RuntimeError("Python environment not found. Run install.sh first.")


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def find_ball_python():
    candidates = [os.environ.get("G1_BALL_PYTHON"),
                  str(APP_DIR / ".ball-venv" / "bin" / "python")]
    return next((candidate for candidate in candidates if candidate and Path(candidate).is_file()), None)


class G1ViserWindow(QMainWindow):
    def __init__(self, host, initial_model=BALL_MODEL):
        super().__init__()
        self.host = host
        self.setWindowTitle("G1 Viser")
        self.setWindowIcon(QIcon.fromTheme("applications-graphics"))
        self.resize(1280, 800)
        self.port = free_port()
        self.ball_port = free_port()
        self.ball_ready_file = Path(tempfile.gettempdir()) / f"g1-ball-ready-{os.getuid()}-{self.ball_port}"
        self.started = time.monotonic()
        self.ball_python = find_ball_python()
        self.ball_process = None
        self.active_model = None
        self.pending_model = None
        self.previous_model = None
        self.pending_since = 0.0
        self.extra_models = {Path(initial_model).resolve()}

        self.stack = QStackedWidget()
        self.message = QLabel("Starting G1 stereo infrared 3D view…")
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webview = QWebEngineView()
        self.webview.settings().setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        self.webview.loadFinished.connect(self.loaded_page)
        self.stack.addWidget(self.message)
        self.stack.addWidget(self.webview)
        self.setCentralWidget(self.stack)
        self.statusBar().showMessage("Connecting to the G1 stereo infrared cameras…")

        toolbar = QToolBar("Ball detection model", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addWidget(QLabel("  Ball model  "))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(220)
        toolbar.addWidget(self.model_combo)
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_models)
        toolbar.addWidget(refresh_button)
        browse_button = QPushButton("Choose .pt…")
        browse_button.clicked.connect(self.browse_model)
        toolbar.addWidget(browse_button)
        switch_button = QPushButton("Switch model")
        switch_button.clicked.connect(self.switch_model)
        toolbar.addWidget(switch_button)
        self.model_status = QLabel("  Preparing ball model…")
        toolbar.addWidget(self.model_status)
        self.refresh_models(initial_model)

        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = LOG_PATH.open("a")
        self.log_file.write(f"\n=== Launch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        self.log_file.flush()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.worker_env = env
        if self.ball_python and Path(initial_model).is_file():
            self.start_ball(Path(initial_model).resolve())
        else:
            self.model_status.setText("  Ball detection unavailable: model or CUDA environment missing")
            self.log_file.write("Ball detection unavailable: model or CUDA Python environment missing. Run install-ball.sh.\n")
            self.log_file.flush()
        self.process = subprocess.Popen(
            [str(find_python()), "-u", str(VISER_SCRIPT), "--host", host, "--port", str(self.port),
             "--ball-port", str(self.ball_port)],
            cwd=VISER_SCRIPT.parent,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_ready)
        self.timer.start(300)
        self.ball_timer = QTimer(self)
        self.ball_timer.timeout.connect(self.check_ball)
        self.ball_timer.start(300)

    def refresh_models(self, preferred=None):
        selected = str(preferred or self.model_combo.currentData() or self.active_model or BALL_MODEL)
        paths = {path.resolve() for path in (APP_DIR / "models").glob("*.pt")}
        paths.update(path for path in self.extra_models if path.is_file())
        self.model_combo.clear()
        for path in sorted(paths, key=lambda value: (value.name != "pingpong_v1.pt", value.name)):
            self.model_combo.addItem(MODEL_LABELS.get(path.stem, path.stem), str(path))
            self.model_combo.setItemData(self.model_combo.count() - 1, str(path), Qt.ItemDataRole.ToolTipRole)
        index = self.model_combo.findData(selected)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)

    def browse_model(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Choose a ping-pong ball detection model", str(APP_DIR / "models"),
                                                  "PyTorch models (*.pt)")
        if filename:
            path = Path(filename).resolve()
            self.extra_models.add(path)
            self.refresh_models(path)
            self.switch_model()

    def switch_model(self):
        path = self.model_combo.currentData()
        if not path:
            return
        path = Path(path).resolve()
        if self.pending_model == path:
            return
        if self.active_model == path and self.ball_process and self.ball_process.poll() is None:
            self.model_status.setText(f"  Active model: {path.stem}")
            return
        if not self.ball_python:
            self.model_status.setText("  CUDA Python environment not found. Run install-ball.sh.")
            return
        if not path.is_file():
            self.model_status.setText(f"  Model file not found: {path.name}")
            return
        self.previous_model = self.active_model
        self.stop_ball()
        self.start_ball(path)

    def start_ball(self, path):
        self.ball_ready_file.unlink(missing_ok=True)
        self.pending_model = path
        self.pending_since = time.monotonic()
        self.model_status.setText(f"  Loading: {path.stem}…")
        self.ball_process = subprocess.Popen(
            [self.ball_python, "-u", str(BALL_SCRIPT), "--camera", f"tcp://{self.host}:5560",
             "--publish", f"tcp://127.0.0.1:{self.ball_port}", "--model", str(path),
             "--ready-file", str(self.ball_ready_file)],
            cwd=BALL_SCRIPT.parent, stdout=self.log_file, stderr=subprocess.STDOUT,
            env=self.worker_env,
        )

    def stop_ball(self):
        if self.ball_process is not None and self.ball_process.poll() is None:
            self.ball_process.terminate()
            try:
                self.ball_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.ball_process.kill()
                self.ball_process.wait()
        self.ball_ready_file.unlink(missing_ok=True)

    def check_ball(self):
        if self.ball_process is None:
            return
        if self.pending_model is not None:
            if self.ball_ready_file.is_file() and self.ball_ready_file.read_text().strip() == str(self.ball_process.pid):
                self.active_model = self.pending_model
                self.pending_model = None
                self.previous_model = None
                self.model_status.setText(f"  Active model: {self.active_model.stem}")
                return
            if self.ball_process.poll() is not None or time.monotonic() - self.pending_since > 30:
                failed = self.pending_model
                fallback = self.previous_model
                self.pending_model = None
                self.previous_model = None
                self.stop_ball()
                self.log_file.write(f"Ball model failed to load: {failed}\n")
                self.log_file.flush()
                if fallback and fallback != failed and fallback.is_file():
                    self.model_status.setText(f"  {failed.stem} failed; restoring {fallback.stem}…")
                    self.start_ball(fallback)
                    self.refresh_models(fallback)
                else:
                    self.active_model = None
                    self.model_status.setText(f"  {failed.stem} failed; see {LOG_PATH}")
        elif self.ball_process.poll() is not None:
            self.active_model = None
            self.model_status.setText(f"  Ball detector stopped; see {LOG_PATH}")

    def check_ready(self):
        if self.process.poll() is not None:
            self.timer.stop()
            self.show_error(f"Viser failed to start (exit code {self.process.returncode}).\nLog: {LOG_PATH}")
            return
        try:
            with urlopen(f"http://127.0.0.1:{self.port}/", timeout=0.15) as response:
                if response.status != 200:
                    return
        except OSError:
            if time.monotonic() - self.started > 45:
                self.timer.stop()
                self.show_error(f"Viser startup timed out.\nLog: {LOG_PATH}")
            return
        self.timer.stop()
        self.webview.setUrl(QUrl(f"http://127.0.0.1:{self.port}/"))
        self.stack.setCurrentWidget(self.webview)
        self.statusBar().showMessage("G1 stereo infrared · 60 FPS ball detection and 3D localization")

    def loaded_page(self, ok):
        if not ok:
            self.show_error(f"The 3D view failed to load.\nLog: {LOG_PATH}")

    def show_error(self, message):
        self.message.setText(message)
        self.stack.setCurrentWidget(self.message)
        self.statusBar().showMessage("Connection failed")

    def closeEvent(self, event):
        self.timer.stop()
        self.ball_timer.stop()
        self.stop_ball()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log_file.close()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.123.164", help="G1 camera host")
    parser.add_argument("--model", type=Path, default=BALL_MODEL, help="initial ball detector .pt")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    window = G1ViserWindow(args.host, args.model)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
