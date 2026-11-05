G1 Studio — Local Robot & Vision Console

START
Open G1 Studio from the desktop or application menu.
No controller or camera service starts automatically.

VISION
1. Connect the G1 Ethernet cable and power on the robot.
2. Keep Robot IP at 192.168.123.164 unless your setup has changed.
3. Click Connect / Start Vision.
4. If camera data does not arrive, click Start Camera Service.
   A separate SSH terminal opens for authentication. Enter the robot password.
   This is the only operation that requires a separate terminal.
5. Camera & Detection shows stereo frames and YOLO previews.
   Viser 3D View shows the robot, table and estimated ball position.
   It is a visualizer, not a physics simulator.

SONIC (REAL ROBOT)
Uses the existing locally installed SONIC model and TensorRT environment.
Select enp130s0 and click Start SONIC (Real Robot).
Check the log for initialization and robot state before enabling control.
The Enable Control button sends ]. Toggle Planner sends Enter.
Use the one-key input for existing keyboard controls.
R stops motion only in Planner mode; it differs in reference-motion mode.
Releasing a movement key does not immediately stop motion.
Exit SONIC sends O. Physical emergency stop and fall protection remain necessary.
The app does not close while its SONIC process is running.
An existing external SONIC controller prevents a duplicate launch.

RECORDING
Choose a duration (5–600 seconds), then Start Recording.
A five-second countdown gives time to prepare.
Stop & Save ends acquisition early and finalizes the files.
Wait for export completion before closing the app.
Open Recordings Folder to find review.mp4, review.avi, source JPEGs,
detections.jsonl, frames.jsonl, calibration.json and summary.json.
The MP4 is a convenience preview; source timing is preserved in frames.jsonl.

STATUS AND LIMITATIONS
Status labels use fresh incoming packets, not only process existence.
The current live 3D pipeline is the previous single-ball pipeline with contour
and YOLO fallback in Viser. Multi-ball optimization remains an offline replay.
World coordinates use the existing calibration and require visible tags.
Changing robot/camera pose still requires a valid current table transform.
Backend logs are available in Logs; original Viser backend text is unchanged.
Stop Vision stops only vision processes owned by this app.
Robot camera services remain available for other viewers.
All model/software dependencies use their existing local paths; this is not
an installer for another computer.

VALIDATION
Offline checks cover idle startup, packet display, ZMQ subscription, PTY key
transport and shutdown. Live robot/control operation requires a connected G1.

TRAJECTORY
Start Vision, confirm a valid 3D point in Viser, then open Trajectory and click
Start Trajectory. Displays velocity and gravity-based 50/100/200 ms forecasts.
Forecasts stop at the contact-height plane; bounce/spin prediction is not yet
implemented. Model acceleration is not measured acceleration. See the separate
ball-trajectory/README.txt for assumptions, timing and offline replay.

LATENCY UPDATE (2026-10-22)
Camera reception now uses one camera_relay.py process. Preview, YOLO, Viser and
recording subscribe to its local port rather than opening separate robot feeds.
The relay records desktop_received_monotonic; worker timing preserves this
original local receipt timestamp. This avoids repeated transfers over the
currently negotiated 100 Mbps robot link.
Live sampled camera-receipt-to-trajectory median: 48 ms; 95th percentile: 57 ms.
These are a short sample of desktop processing, excluding camera/network input
and screen rendering. They are not a guaranteed end-to-end latency bound.
Trajectory redraws at up to 30 Hz, flushes updates promptly, and writes snapshots
at 10 Hz after publishing. Normal application startup still starts no services.
