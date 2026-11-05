"""Keep recent G1 waist encoder samples for camera-frame timestamp matching."""

from collections import deque
import threading

import msgpack
import zmq


class JointStream:
    def __init__(self, host, port=5562):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVTIMEO, 500)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{host}:{port}")
        self.samples = deque(maxlen=100)
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self):
        while not self.stopped:
            try:
                sample = msgpack.unpackb(self.socket.recv(), raw=False)
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            if "sample_monotonic_ns" not in sample or "waist_yaw_rad" not in sample:
                continue
            with self.lock:
                self.samples.append(sample)

    def nearest(self, frame_monotonic_ns, max_delta_ms=100):
        if frame_monotonic_ns is None:
            return None
        with self.lock:
            if not self.samples:
                return None
            sample = min(self.samples,
                         key=lambda item: abs(item["sample_monotonic_ns"] - frame_monotonic_ns))
        delta_ms = abs(sample["sample_monotonic_ns"] - frame_monotonic_ns) / 1e6
        return sample if delta_ms <= max_delta_ms else None

    def close(self):
        self.stopped = True
        self.thread.join(timeout=1)
        self.socket.close()
        self.context.term()
