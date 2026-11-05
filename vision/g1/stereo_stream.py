"""Client for the G1 stereo IR publisher on TCP 5560."""

import cv2
import msgpack
import numpy as np
import threading
import zmq


class StereoStream:
    def __init__(self, host="192.168.123.164", port=5560):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)
        self.socket.connect(f"tcp://{host}:{port}")
        self._condition = threading.Condition()
        self._latest = self._receive()
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)
        self._sequence = 1
        self._seen = 0
        self._stopped = False
        self.calibration = self._latest["calibration"]
        self.left_intr = self.calibration["left_intrinsics"]
        self.right_intr = self.calibration["right_intrinsics"]
        self.T_left_right = np.asarray(self.calibration["T_left_right"], dtype=float)
        self.T_left_color = np.asarray(self.calibration["T_left_color"], dtype=float)
        self.last_frame_monotonic_ns = None
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _receive(self):
        return msgpack.unpackb(self.socket.recv(), raw=False)

    def _drain(self):
        while not self._stopped:
            try:
                packet = self._receive()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            with self._condition:
                self._latest = packet
                self._sequence += 1
                self._condition.notify_all()

    def read(self):
        with self._condition:
            if not self._condition.wait_for(lambda: self._sequence != self._seen or self._stopped, 5):
                raise TimeoutError("no stereo IR frames received for 5 seconds")
            if self._stopped:
                raise ConnectionError("stereo IR stream closed")
            packet = self._latest
            self._seen = self._sequence
        self.last_frame_monotonic_ns = packet.get("capture_monotonic_ns")
        pair = cv2.imdecode(np.frombuffer(packet["jpg"], np.uint8), cv2.IMREAD_GRAYSCALE)
        width = self.left_intr["width"]
        height = self.left_intr["height"]
        if pair is None or pair.shape != (height, 2 * width):
            raise ValueError(f"unexpected stereo frame size: {None if pair is None else pair.shape}")
        return pair[:, :width].copy(), pair[:, width:].copy(), packet["left_id"], packet["right_id"]

    def close(self):
        self._stopped = True
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=2)
        self.socket.close()
        self.context.term()
