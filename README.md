# G1 stereo table localization and ball vision

This repository contains a desktop Viser application for Unitree G1 stereo IR cameras, AprilTag table localization, measured-waist-angle tracking, and optional GPU ping-pong ball detection. The app is read-only with respect to robot motors. It does not send motion commands.

The table frame is defined by tag 5 at the middle of the near edge: +X runs across the table, +Y points toward the net, and +Z points up. `g1/calibration.json` contains the local camera/table calibration and a tape-measured waist correction. Its fallback tag 0/1/2 poses are disabled because those tags moved; recalibrate them against tag 5 before enabling them. The reported distance is unavailable when no valid reference tag is visible.

## Desktop setup

On Ubuntu with a desktop display:

```sh
./install.sh
./run.sh --host 192.168.123.164
```

The application opens a dedicated window. Ball detection is optional. Put a compatible `.pt` model in `models/`, then run `./install-ball.sh` on an NVIDIA CUDA computer. Select the model in the toolbar. Model weights, Python environments, recorded video, and live test data are deliberately excluded from Git.

## G1 camera and joint streams

Run `g1/stereo_pose_server.py` on the G1 in a Python environment with `pyrealsense2`, OpenCV, NumPy, MessagePack, and ZeroMQ. It publishes synchronized 480×270 left/right IR pairs at 60 fps on TCP 5560 and includes a robot monotonic timestamp for each frame. Example:

```sh
python3 -u g1/stereo_pose_server.py --fps 60 --width 480 --height 270
```

Run `g1/joint_state_server.py` on the G1 in an environment with `unitree_sdk2py`, CycloneDDS, MessagePack, and ZeroMQ. It subscribes to `rt/lowstate`, reads the measured waist-yaw encoder (motor 12), and publishes timestamped samples on TCP 5562. It never publishes motor commands:

```sh
python3 -u g1/joint_state_server.py --port 5562
```

The desktop matches camera frames with waist samples using the G1's monotonic clock. It uses current URDF forward kinematics to recover pelvis pose and animate the model. If joint telemetry is missing, the interface labels the estimate as zero-angle and warns that it can drift during SONIC. Waist roll and pitch are not used by this 29-DoF setup; they may be locked on the specific G1 variant.

## Checks

`python3 g1/offline_waist_test.py` runs a synthetic -30° to +30° sweep with a fixed pelvis. `python3 g1/live_waist_check.py --output waist-test.csv --seconds 30` records read-only synchronized measurements during a live SONIC test. `g1/recalibrate_fallback_tags.py` can remeasure tags 0/1/2 from simultaneous stereo views of tag 5; inspect dry-run output before using `--write`.

The live test on 2026-10-09 confirmed that, during one ~23° left waist sweep, the measured-joint calculation changed estimated pelvis yaw by about 0.4° while the old fixed transform would have changed it by about 23°. Right-sweep visibility was insufficient for a complete validation. The one-point tape correction does not certify absolute distance accuracy at every robot pose.


## Integrated G1 Studio and trajectory experiments

The newer workstation source snapshot is in `outputs/G1-Studio`, with recording in `outputs/ball-recorder`, trajectory/collision/interception prediction in `outputs/ball-trajectory`, and simulation-only SONIC reach experiments in `outputs/sonic-reach-test`. Shared updated vision modules are in `vision/g1`; `experiments` contains historical offline analysis scripts. Existing root-level vision entry points remain available.

This snapshot preserves machine-specific absolute paths and calibration. It is not a portable installer: review the component README.txt files and configure existing Python environments, model locations, and SONIC installation paths before use. Model weights, environments, videos and runtime logs are excluded. `sonic` contains the local launch scripts; some explicitly start physical robot control, whereas `outputs/sonic-reach-test` is loopback simulation only. No services start merely by opening the Studio application.

Latest simulation validation: right/up 5 cm reaches passed twice each; left/down failed twice each, so this is not ready for ball interception. The English result table is in `outputs/sonic-reach-test/validation/results.html`. Offline validation before this snapshot: 31 trajectory tests and 5 reach-harness tests passed; Python and shell syntax checks passed. No new physical robot test was run for this publication.
