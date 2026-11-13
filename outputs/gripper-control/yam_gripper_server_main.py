#!/usr/bin/env python3
"""
YAM Gripper Server - ZMQ REP server for multi-gripper control.
"""

import logging
import signal
import threading
import zmq
import msgpack
import numpy as np
from typing import List, Dict, Any, Optional

from .yam_gripper import YamGripper
from . import dm_protocol as dm
from .calibration_cache import load_calibration, save_calibration
from .can_interfaces import CAN_INTERFACE_NAMES, interface_exists
from .yam_gripper_protocol import HandSide


class YamGripperServer:
    def __init__(
        self,
        port: int = 5558,
        gripper_configs: Optional[List[Dict[str, Any]]] = None,
        auto_detect: bool = True,
        motor_kwargs: Optional[Dict[HandSide, Dict[str, Any]]] = None,
        auto_detect_sides: Optional[List[HandSide]] = None,
        require_calibration_cache: bool = False,
    ):
        """Initialize YamGripperServer with multiple grippers.

        Args:
            port: ZMQ server port (default: 5558)
            gripper_configs: List of gripper configuration dicts, each containing
                kwargs for YamGripper constructor. If None and auto_detect is True,
                automatically scans and configures grippers.
            auto_detect: If True, automatically scan for motors and match to sides (default: True)
            motor_kwargs: Per-side YamGripper kwargs merged into auto-detected configs

        Example (manual configuration):
            gripper_configs = [
                {"gripper_name": "left", "interface_type": "socketcan", "channel": "can_gripper_l", "motor_id": 0x06},
                {"gripper_name": "right", "interface_type": "socketcan", "channel": "can_gripper_r", "motor_id": 0x08},
            ]
            server = YamGripperServer(port=5558, gripper_configs=gripper_configs, auto_detect=False)

        Example (auto-detect):
            server = YamGripperServer(port=5558)  # Automatically scans and configures
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

        self.motor_kwargs = motor_kwargs or {}
        self.auto_detect_sides = auto_detect_sides
        self.require_calibration_cache = require_calibration_cache

        # Auto-detect motors or use provided configs
        if gripper_configs is None and auto_detect:
            gripper_configs = self._auto_detect_grippers()
        elif gripper_configs is None:
            # Default: single gripper on the left interface
            gripper_configs = [{"interface_type": "socketcan", "channel": CAN_INTERFACE_NAMES["left"]}]

        # Create grippers from configs
        # Store in a dict indexed by HandSide for clear left/right mapping
        self.grippers: Dict[HandSide, YamGripper] = {}

        for config in gripper_configs:
            side = config.get("side", None)
            if side is not None and isinstance(side, HandSide):
                side_name = "left" if side == HandSide.LEFT else "right"
                cached = load_calibration(side_name, config["motor_id"])
                if cached is None and self.require_calibration_cache:
                    raise RuntimeError(
                        f"No cached calibration for {side_name} gripper; "
                        "refusing automatic calibration. Move the gripper clear "
                        "of the robot before calibrating it intentionally."
                    )
                if cached is not None:
                    print(f"Using cached calibration for the {side_name} gripper "
                          f"(delete /tmp/yam_gripper_calibration_{side_name}.json to recalibrate)")
                    config = {**config, **cached}
                gripper = YamGripper(**{k: v for k, v in config.items() if k != "side"})
                self.grippers[side] = gripper
                if cached is None:
                    path = save_calibration(
                        side_name,
                        gripper.motor_id,
                        {
                            "zero_distance_angle": gripper.zero_distance_angle,
                            "max_distance_angle": gripper.max_distance_angle,
                            "gear_radius": gripper.gear_radius,
                        },
                    )
                    print(f"Saved {side_name} gripper calibration to {path}")
                print(f"Initialized {side_name} gripper on channel {config['channel']}, motor_id 0x{config['motor_id']:02X}")
            else:
                # Legacy support: if no side specified, just create grippers in order
                gripper = YamGripper(**config)
                idx = len(self.grippers)
                self.grippers[HandSide(idx)] = gripper
                print(f"Initialized gripper {idx}: {config.get('gripper_name', f'gripper_{idx}')}")

        # Start all grippers in zero-torque mode (backdrivable) until the
        # first ZMQ command arrives and calls set_target_width_mm.
        for gripper in self.grippers.values():
            gripper.set_zero_torque()

        if len(self.grippers) == 0:
            print(f"WARNING: YAM Gripper Server started on port {port} with NO grippers configured!")
            print("  Server will run but all commands will return error feedback.")
        else:
            print(f"YAM Gripper Server started on port {port} with {len(self.grippers)} gripper(s)")

    def _auto_detect_grippers(self) -> List[Dict[str, Any]]:
        """Scan the udev-named gripper interfaces for motors.

        Every selected gripper is required: a missing interface or a bus
        without exactly one motor raises, so the service exits.
        """
        sides = [
            (HandSide.LEFT, "left"),
            (HandSide.RIGHT, "right"),
        ]
        if self.auto_detect_sides is not None:
            sides = [(side, name) for side, name in sides
                     if side in self.auto_detect_sides]
        if not sides:
            raise ValueError("At least one gripper side must be selected")

        import can as _can
        configs: List[Dict[str, Any]] = []
        for side, name in sides:
            channel = CAN_INTERFACE_NAMES[name]
            if not interface_exists(channel):
                raise RuntimeError(
                    f"No {channel} interface for the {name} gripper — dongle "
                    f"disconnected or udev rules missing. Run `python -m "
                    f"yam_gripper_server.configure_dongle_sides` to install "
                    f"the udev rules."
                )
            # Scan that bus for the gripper motor.
            with _can.Bus(interface="socketcan", channel=channel) as bus:
                motors = dm.scan_motors_on_bus(bus, min_id=1, max_id=16, timeout=0.1)
            if len(motors) != 1:
                raise RuntimeError(
                    f"{channel} has {len(motors)} motors (expected 1) — "
                    f"exiting so systemd retries."
                )
            motor_id, _resp_id = motors[0]
            configs.append({
                "gripper_name": name,
                "interface_type": "socketcan",
                "channel": channel,
                "motor_id": motor_id,
                "side": side,
                **self.motor_kwargs.get(side, {}),
            })
            print(f"Matched {name} gripper: channel={channel} "
                  f"motor_id=0x{motor_id:02X}")
        return configs

    def _create_default_feedback(self, gripper_id: int, target_width_mm: float) -> Dict[str, Any]:
        """Create default feedback for unavailable grippers."""
        return {
            "gripper_id": gripper_id,
            "target_width_mm": target_width_mm,
            "measured_width_mm": 0.0,
            "measured_width_timestamp_monotonic_s": None,
            "temp_mos": 0.0,
            "temp_rotor": 0.0,
        }

    def run(self):
        """Run the ZMQ server to handle incoming gripper commands.

        Receives a list of commands, each a dict with:
            {
                "gripper_id": int,  # 0 = LEFT, 1 = RIGHT (see HandSide enum)
                "target_width_mm": float,
                "max_torque": float,  # optional motor torque limit in Nm; None or absent keeps the current limit
            }
        Sends back a list of feedback dicts, each with:
            {
                "gripper_id": int,
                "target_width_mm": float,
                "measured_width_mm": float,
                # time.monotonic() when the motor feedback behind
                # measured_width_mm was cached; freezes when feedback stops,
                # None until the first feedback or for unavailable grippers.
                "measured_width_timestamp_monotonic_s": float | None,
                "temp_mos": float,
                "temp_rotor": float,
            }
        """
        shutdown_requested = False

        def _signal_handler(signum, frame):
            nonlocal shutdown_requested
            shutdown_requested = True
            print(f"\nReceived signal {signum}, shutting down...")

        signal.signal(signal.SIGTERM, _signal_handler)

        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        try:
            while not shutdown_requested:
                for side, gripper in self.grippers.items():
                    if gripper.motor_chain.fatal_error is not None:
                        print(
                            f"FATAL: {gripper.motor_chain.fatal_error} — the "
                            f"bound socket can never recover. Exiting so "
                            f"systemd restarts the service and re-binds."
                        )
                        raise SystemExit(1)
                # Poll with a timeout so we can check the shutdown flag
                events = poller.poll(timeout=1000)
                if not events:
                    continue

                message = self.socket.recv()
                commands = msgpack.unpackb(message, raw=False)

                # commands is a list of dicts: [{"gripper_id": 0, "target_width_mm": 50.0}, ...]
                # gripper_id 0 = LEFT, gripper_id 1 = RIGHT
                feedbacks = []

                for cmd in commands:
                    gripper_id = cmd["gripper_id"]
                    target_width_mm = cmd["target_width_mm"]

                    # Map gripper_id to HandSide
                    try:
                        side = HandSide(gripper_id)
                    except ValueError:
                        print(f"Error: Invalid gripper_id {gripper_id}. Must be 0 (LEFT) or 1 (RIGHT)")
                        feedbacks.append(self._create_default_feedback(gripper_id, target_width_mm))
                        continue

                    if side not in self.grippers:
                        side_name = "left" if side == HandSide.LEFT else "right"
                        feedbacks.append(self._create_default_feedback(gripper_id, target_width_mm))
                        continue

                    side_name = "left" if side == HandSide.LEFT else "right"

                    gripper = self.grippers[side]
                    gripper.set_target_width_mm(np.array([target_width_mm]), max_torque=cmd.get("max_torque"))
                    states = gripper.read_states()

                    feedbacks.append({
                        "gripper_id": gripper_id,
                        "target_width_mm": target_width_mm,
                        "measured_width_mm": states[0].width_mm,
                        "measured_width_timestamp_monotonic_s": states[0].timestamp_monotonic,
                        "temp_mos": states[0].temp_mos,
                        "temp_rotor": states[0].temp_rotor,
                    })

                self.socket.send(msgpack.packb(feedbacks))

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            for gripper in self.grippers.values():
                gripper.shutdown()
            self.socket.close()
            self.context.term()


def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(
        description="YAM Gripper Server - Multi-gripper control via ZMQ"
    )
    parser.add_argument("--port", type=int, default=5558, help="ZMQ server port")
    parser.add_argument(
        "--right-only", action="store_true",
        help="Auto-detect only the right gripper; left-side commands return unavailable feedback",
    )
    parser.add_argument(
        "--require-calibration-cache", action="store_true",
        help="Refuse to start if calibration is missing, instead of moving jaws automatically",
    )
    parser.add_argument(
        "--no-auto-detect",
        action="store_true",
        help="Disable automatic motor detection and use manual configuration"
    )
    parser.add_argument(
        "--num-grippers",
        type=int,
        default=None,
        help="Number of grippers for manual mode (creates grippers on channels 0 to num-1)"
    )
    parser.add_argument("--kp", type=float, default=None, help="Position control proportional gain")
    parser.add_argument("--kd", type=float, default=None, help="Position control derivative gain")
    args = parser.parse_args()

    # Build per-side motor kwargs from optional CLI args
    side_kwargs = {}
    if args.kp is not None:
        side_kwargs["kp"] = args.kp
    if args.kd is not None:
        side_kwargs["kd"] = args.kd

    motor_kwargs = {
        HandSide.LEFT: side_kwargs,
        HandSide.RIGHT: side_kwargs,
    }

    if args.no_auto_detect:
        # Manual configuration mode: gripper index maps to HandSide (0=LEFT, 1=RIGHT)
        num_grippers = args.num_grippers if args.num_grippers is not None else 1
        gripper_configs = [
            {"gripper_name": f"gripper_{i}", "interface_type": "socketcan",
             "channel": list(CAN_INTERFACE_NAMES.values())[i],
             **motor_kwargs[HandSide(i)]}
            for i in range(num_grippers)
        ]
        server = YamGripperServer(port=args.port, gripper_configs=gripper_configs, auto_detect=False)
    else:
        # Auto-detect mode (default) - pass motor kwargs to auto-detected configs
        server = YamGripperServer(
            port=args.port,
            auto_detect=True,
            motor_kwargs=motor_kwargs,
            auto_detect_sides=[HandSide.RIGHT] if args.right_only else None,
            require_calibration_cache=args.require_calibration_cache,
        )

    server.run()


if __name__ == "__main__":
    main()
