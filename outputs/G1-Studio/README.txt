G1 Studio — Local Robot & Vision Console

START
Open G1 Studio from the desktop or application menu.
No controller or camera service starts automatically.

TOKEN POLICY SIMULATION
Open Token Policy Sim, select Front table tennis (or an older curriculum), then Start Token
Simulation. These run entirely on this computer and send no robot commands.
Front table tennis shows a learned racket lift, front feed, table and net.
Face contact, net crossing and legal landing are reported separately.
This is an experimental simulation; no real-robot commands are sent.
Its viewer lets you compare controllers using Replay same ball.
Stop Token Simulation before switching stages. The contact exercise is side-fed
and below table height; it is not a trained table-tennis rally.

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

SONIC launch and software STOP
-----------------------------
Open the desktop icon "G1 SONIC Control", or run start_sonic_control.sh.
This opens the SONIC tab only; opening the program does not start motor control.
A second launch brings the existing window to that tab.

Start SONIC (Real Robot) starts the existing controller through a separate
heartbeat supervisor. Initialization itself moves toward the default pose.
The current forehand return policy remains simulation-only; this launcher does
not publish its tokens to the physical robot.

STOP SONIC (red button), Esc while this app has focus, and Exit SONIC / O all
request SONIC's global O stop. SONIC's normal Stop() path sends damping-only
commands. The app also stops owned token-live/token-publisher processes if any
are later connected. Stop latches: Reset Stop Latch is required before a new
start; resetting does not start the robot. App launch never enables control.

The supervisor checks a 150 ms GUI heartbeat. After 1.2 s without a fresh
heartbeat, it independently requests O. Its output is nonblocking so a frozen
GUI cannot stop this check by filling the logging pipe. It refuses startup if
there is no fresh heartbeat, and also requests stop on EOF or SIGTERM/SIGINT.
If SONIC does not exit within two seconds, the app/supervisor reports STOP
UNCONFIRMED and directs the operator to the physical emergency stop.

This is a SOFTWARE STOP, not a safety-rated emergency stop. Esc is not a
system-wide key. Computer power loss, a hung SONIC process, lost communication,
or controllers launched outside this application are not covered. Process exit
is not a measurement of motor state. Keep the physical emergency stop ready;
when supported, damping exit may let the arms and racket lower.

Offline validation: test_stop.py exercises real PTYs with harmless mock
controllers for Esc, the red button, the restart latch, a frozen GUI heartbeat,
and missing-heartbeat refusal. No physical token motion has been validated.

Slow-swing preparation (2026-11-20)
- SONIC Control > Check Robot for Slow Swing (Read Only) subscribes to rt/lowstate
  on the selected interface, compares all 29 measured joints against the settled
  simulation start, and writes runtime/session-*/swing-preflight.json. No motor
  publisher is constructed. A pose difference is not a motion command.
- Preview Trained Swing in Simulation starts the existing MuJoCo return viewer.
  Playback speed in that viewer changes display pacing, not physical motor speed.
- Play Physical Swing is disabled: no physical stroke publisher is wired. The
  real mount and transitions into/out of the stroke still need validation.
- The rebuilt SONIC receiver checks the token buffer's own update timestamp,
  requires 64 finite values (also under fast-math), and requests the normal
  damping exit if external tokens are missing/stale for 100 ms. Failure latches
  until process restart. An interface switch after external control also stops.
  External-only startup without a fresh token is refused.
- This watchdog measures the receiver buffer update age, not sensor-to-actuator
  latency. Repeated fresh packets or queued packets can still refresh it; source
  sequence/age checks remain necessary for a physical publisher. A hung control
  loop or hardware fault is not covered. Keep the physical emergency stop ready.
- token-safety/build-proof.json records the compiled binary hash and offline test
  status. token-safety/receiver-guard.patch preserves this change independently
  from the pre-existing modifications to the SONIC source checkout. A backup of
  the previous binary is in SONIC target/release/*pre-token-guard-20261120.
- Tests passed with harmless PTY processes and a standalone C++ guard. The normal
  STOP button was observed to cause SONIC's Stop() and normal exit on the robot;
  token loss has NOT been injected during physical motion.

Bounded trial command envelope (2026-11-20)
- An opt-in --guarded-upper-trial profile now checks feedback and command age,
  measured speed, displacement from captured startup, target displacement,
  tracking error, finite values, gains, and writer scheduling gaps in the 500 Hz
  command writer. It starts by holding measured joints rather than the default
  startup pose, and expires after 20 seconds. It does not publish Dex3 commands.
- Limits: right-arm speed 0.8 rad/s, target slew 0.25 rad/s, displacement 0.35 rad;
  other joints 0.35 rad/s, 0.05 rad/s, and 5 degrees. Age limits are 100 ms.
  A fault latches damping output (kp=0, kd=8) until process restart. Damping can
  allow the arm to lower; these software checks are not a safety-rated stop.
- This profile is NOT enabled by normal Start SONIC. It is not a validated
  physical swing path, a complete collision check, or an absolute joint-limit
  checker. Play Physical Swing remains disabled. No physical motion was sent.
- trial-safety/test_loopback.py tests the compiled controller with synthetic
  feedback on loopback NIC lo, DDS domain 77, and private ZMQ ports. Domain 77
  cannot be selected on a physical NIC using the new --sim-dds-domain option.
  Results are timestamped packet observations, not real actuator stop times.
- Run tests with GR00T-WholeBodyControl/.venv_sim/bin/python. The test refuses
  coexistence with another g1_deploy_onnx_ref process. Output is in
  trial-safety/loopback-results.json, including the binary SHA256.
- The --guarded-upper-trial profile holds all non-right-arm targets at the
  captured initial pose. All 29 measured joints remain monitored. This is not
  enabled by normal Start SONIC and has not been validated on physical hardware.
- Feet-on-ground recheck: the current support approximation still does not
  reproduce both feet contacting the floor. Left ankle speed trips during the
  initial hold, before a lift. Disabling stock finger collisions does not remove
  the issue. This is not proof that the real robot would move the same way.
  Latest read-only snapshot and diagnostics are in sonic-token-policy/forehand/
  runs/transition-20261120/feet-grounded-live.json and feet-grounded-recheck/.
- command-envelope.patch is incremental from the pre-existing receiver-watchdog
  changes. The existing source checkout also contains unrelated earlier edits.

External-token startup check (2026-11-20, subsequent repair)
- Found a real initialization mismatch: the first decoder inference had nine
  zero-padded history frames, with default gravity, instead of measured states.
  In --guarded-upper-trial, WAIT now gathers at least 12 measured hold samples
  and records the actual hold target as the previous action before permitting
  control. Early start is refused; ordinary SONIC startup is unchanged.
- In this opt-in right-arm trial, all other 22 targets stay at their captured
  values. Their measured speeds/displacements are still checked. Keeping a
  target fixed is not a guarantee that the physical joint cannot move.
- test_external_loopback.py sends a current-pose HOLD token into the actual
  compiled SONIC over ZMQ protocol v4, using lo/DDS77 synthetic feedback only.
  It asserts exact token arrival in decoder input and output, measured history,
  unchanged non-right targets, changed right-arm targets, and damping exit.
  external-result.json records 31 matching frames for the completed check.
  Synthetic joints intentionally do not move; tracking-error stop is expected.
  This is not a trained-swing playback or physical movement test.
- external-before-warmup.json/.csv preserve the failed startup; external-after-
  warmup.json/.csv preserve the intermediate result before the right-arm mask.
- Separately, trained-start-diagnostic.json evaluates the return-v4 initial
  reference token through the ONNX decoder using corrected measured history.
  The right-elbow target still differs by approximately 97 degrees. A gradual
  physical start/return transition is not yet validated; hardware_ready=false.
- startup-history.patch isolates the latest repair; command-envelope.patch
  includes the full opt-in envelope changes since the receiver-watchdog patch.

Slow preparation test (2026-11-20)
The SONIC Control tab has a separate "Slow Preparation · 5° out and back" button.
This opt-in direct joint test captures the measured pose, holds it, moves right
shoulder roll outward by 5 degrees in 3 seconds, holds 1 second, returns in
3 seconds and settles 1 second before damping exit. It does not execute a trained
SONIC token swing. All other joint targets remain at the captured pose.
STOP / Esc uses the same supervisor and stop latch; ordinary motion keys are
blocked for this test. A read-only joint/command log is saved in the GUI session.
The startup gate requires the tested binary and fresh joints within 3 degrees of
the checked preparation pose. It refuses an active/unconfirmed motion service.
The preparation shoulder uses Kp 40 with a 0.25 rad/s measured speed guard and
4 Nm estimated PD-effort guard. Other SONIC gains are unchanged. These software
checks and damping exit do not replace the physical emergency stop.
Two initial trials using the learned policy's softer Kp (~14.25) produced only
~0.3 degrees of measured shoulder movement despite a 5 degree command. The second
trial confirmed motor enable, valid command CRC, and ~1.3 Nm measured torque.
Those trials are not successful preparation motions. Results are in trial-safety.
