import unittest
from unittest.mock import patch

from yam_gripper_server import __main__ as server_module
from yam_gripper_server.yam_gripper_protocol import HandSide


class FakeBus:
    def __init__(self, channel):
        self.channel = channel

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class FakeSocket:
    def bind(self, _address):
        pass


class FakeContext:
    def socket(self, _type):
        return FakeSocket()


class RightOnlyTests(unittest.TestCase):
    def test_auto_detect_scans_only_right_side(self):
        server = server_module.YamGripperServer.__new__(
            server_module.YamGripperServer
        )
        server.auto_detect_sides = [HandSide.RIGHT]
        server.motor_kwargs = {}
        scanned = []

        def make_bus(*, interface, channel):
            self.assertEqual(interface, "socketcan")
            scanned.append(channel)
            return FakeBus(channel)

        with patch("can.Bus", side_effect=make_bus), patch.object(
            server_module, "interface_exists", return_value=True
        ), patch.object(
            server_module.dm, "scan_motors_on_bus", return_value=[(8, 9)]
        ):
            configs = server._auto_detect_grippers()

        self.assertEqual(scanned, ["can_gripper_r"])
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["side"], HandSide.RIGHT)

    def test_missing_cache_cannot_start_calibration(self):
        config = {
            "gripper_name": "right",
            "interface_type": "socketcan",
            "channel": "can_gripper_r",
            "motor_id": 8,
            "side": HandSide.RIGHT,
        }
        with patch.object(server_module.zmq, "Context", return_value=FakeContext()), patch.object(
            server_module, "load_calibration", return_value=None
        ), patch.object(server_module, "YamGripper") as motor:
            with self.assertRaisesRegex(RuntimeError, "No cached calibration"):
                server_module.YamGripperServer(
                    gripper_configs=[config],
                    auto_detect=False,
                    require_calibration_cache=True,
                )
            motor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
