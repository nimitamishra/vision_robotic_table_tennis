"""Read-only G1 motor-state relay for synchronized stereo localization.

Run on the G1 with a Python environment containing unitree_sdk2py. This
subscribes to rt/lowstate; it never publishes motor commands.
"""

import argparse
import threading
import time

import msgpack
import zmq
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


WAIST_YAW_INDEX = 12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5562)
    parser.add_argument("--interface", help="G1 DDS network interface; default SDK selection")
    args = parser.parse_args()

    if args.interface:
        ChannelFactoryInitialize(0, args.interface)
    else:
        ChannelFactoryInitialize(0)

    lock = threading.Lock()
    latest = None

    def on_state(msg):
        nonlocal latest
        sample = {
            "sample_monotonic_ns": time.monotonic_ns(),
            "waist_yaw_rad": float(msg.motor_state[WAIST_YAW_INDEX].q),
            "tick": int(msg.tick),
        }
        with lock:
            latest = sample

    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(on_state, 10)
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.SNDHWM, 1)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"Read-only G1 waist state on {args.port}; waiting for rt/lowstate", flush=True)
    last_sample_time = None
    try:
        while True:
            with lock:
                sample = latest
            if sample is not None and sample["sample_monotonic_ns"] != last_sample_time:
                publisher.send(msgpack.packb(sample, use_bin_type=True))
                last_sample_time = sample["sample_monotonic_ns"]
            time.sleep(0.005)
    finally:
        publisher.close()
        context.term()


if __name__ == "__main__":
    main()
