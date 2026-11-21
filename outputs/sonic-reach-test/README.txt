SONIC RIGHT-HAND REACH TEST — SIMULATION ONLY

Start: ./start.sh
Open: http://127.0.0.1:8090
Close an existing viewer on port 8090 before starting a new run.
The launcher enables experimental bounded position feedback.

Actual SONIC models control the MuJoCo G1 model. Blue links show simulated body
positions, green the desired target, orange the measured right-hand tracking
point. It includes SONIC's wrist offset, not a calibrated racket center.
Targets/errors are relative to the current pelvis, not the table.

Startup initializes the default pose, lowers support until both feet contact the
floor for 0.5 seconds, then fades support over 2 seconds. SONIC starts unsupported.
After 6 seconds of standing, hand/head targets are captured and held.
The reach button requires all three position errors within 2 cm for 1 second.
Click "Move right hand forward 5 cm": a 2-second smooth ramp, then a 5-second hold.
Low pelvis height stops the test instead of silently resetting the simulator.

Feedback adjusts transmitted targets: gain 0.5/s, correction speed capped at
1 cm/s per axis, correction magnitude capped at 4 cm per point. Errors are always
relative to ORIGINAL targets. For a baseline invoke sim_test.py without
--position-feedback, using the simulation Python environment in start.sh.

Use "Stop simulation" to stop its controller. Completed runs show a stopped
snapshot, not live tracking. Closing the launcher stops simulation and viewer.
Simulation exits after 5 minutes. Results: runs/<timestamp>/samples.jsonl,
controller.log, simulator.log, state.json, finished.json and report.json.

RESULTS (single trials, not repeatability validation)
grounded02 without feedback: unsupported hold mean errors left 2.53 cm, right
1.25 cm, head 0.61 cm. Readiness failed, so no reach was attempted.
grounded_feedback01: unsupported hold errors left 1.49 cm, right 1.59 cm, head
0.20 cm. Readiness passed. Right target moved forward 5 cm over 2 seconds.
Measured displacement: forward 4.59 cm, lateral 1.11 cm, vertical -0.01 cm.
Final-second position error: mean 1.14 cm, maximum 1.17 cm. Minimum pelvis height
during movement: 0.786 m. Passed the 2 cm final-second position criterion.
Some corrections approached the 4 cm cap; this remains an experimental aid.
Earlier supported01 failed (8.06 cm final mean error); grounded startup replaces
its suspended initialization procedure.

DDS is fixed to loopback (lo), domain 0, with local ZMQ ports 15576/15577.
The harness refuses to start if a SONIC controller or standard simulator exists.
The physical robot launcher is unchanged. No physical interface can be specified.
This test does not validate orientation accuracy, fast interception, racket-center
calibration, table-to-robot calibration or physical robot motion.
Next: repeat small reaches in several directions before connecting ball forecasts.

READY-POSE EXPERIMENTS (SIMULATION ONLY)
The optional --absolute-right-target X Y Z supplies a pelvis-relative tracking
point. --ramp-seconds and --hold-seconds set the timing. --stage-length 0.05
divides an absolute VR target into at most 5 cm waypoints, each requiring one
second below 2 cm position error before moving on. --right-joint-goal takes the
seven right-arm joint angles in MuJoCo order and conditions SONIC on the
17-joint upper-body target instead of the VR point during the reach. The VR
point remains the evaluation target. --initial-right-q seeds the simulated arm
with a measured starting pose. --hold-initial-upper-body tries to preserve that
pose from startup; --hold-initial-vr also provides the initial three tracking
points during startup. --startup-joint-feedback applies a bounded 0.6 rad
simulation-only correction during the standing interval set by
--standing-seconds. These options never select a physical robot interface.

Run capture_live_pose.py to save a fresh read-only pose snapshot and preliminary
position-only IK target. The model/observation discrepancy must be below 2 cm.
The resulting joint vector is an IK estimate, not a safe robot command.
Results and limitations are in READY_POSE_RESULTS.md.

DIRECTIONAL BATCH TEST
python3 batch_test.py --output /absolute/path/to/a/new/batch-directory
Runs left (+Y), right (-Y), up (+Z), down (-Z), twice each, relative to pelvis.
Each trial starts a fresh simulator/controller, waits for the same readiness gate,
then sends its 5 cm displacement automatically. No failed trial is silently retried.
The output directory must not already exist. Read results.html or results.json.
Individual run directories retain raw telemetry, configurations and controller logs.
Do not launch a second controller while this sequential batch runs.

DIRECTIONAL RESULTS: runs/directions01/results.html
Eight trials completed unsupported: right 0.61/0.68 cm and up 0.62/0.84 cm
passed; left 5.72/6.28 cm and down 6.47/6.56 cm failed (final-second mean errors).
Both feet maintained contact throughout all reach phases. This is directional
tracking failure, not proof of its cause; workspace feasibility and model tracking
from the current starting posture require further investigation. Do not connect
ball forecasts to physical motion based on these tests.

LATEST SHORT-REACH CHECK (SIMULATION ONLY)
READY_POSE_RESULTS.md records the current-pose collision finding and 10/10
repeatable 2 cm outward reaches. The old direct 2 cm forward path intersects
the right hip and is rejected before motion. These tests use simulation-only
arm gains and gravity compensation; they are not a physical G1 control path.
The concise aggregate is runs/ready-edge-outward-validation.json.
Use table_contact_audit.py /absolute/path/to/a/completed/run to replay a run
against the saved tabletop and net. This offline audit uses a saved pose, and
the racket is not yet installed or modeled.

DIRECT LATENT INTERFACE EXPERIMENT (SIMULATION ONLY)
The existing ball-proxy-02 policy outputs three normalized right-hand position
offsets. It does NOT output a SONIC latent. The configured SONIC decoder takes
64 latent values plus robot-state history. In the target-based route the SONIC
encoder creates those latents internally; the external-token route supplies
them directly through the pose topic (protocol v4, token_state [1,64]).

--record-token-state records encoder output to tokens.jsonl during the ordinary
loopback simulation. --token-replay /absolute/path/to/tokens.jsonl switches from
planner to streamed tokens at the start of the reach. Match the source delta,
ramp time and hold time. These options cannot select a physical NIC, cannot
combine replay with direct-arm servo, and do not train a new policy. The
replay result records whether SONIC acknowledged external tokens and whether
the controller's actual consumed tokens matched recently transmitted values.

A future ball-to-token policy needs training with the frozen SONIC decoder
inside the simulation loop. The 64 coordinates are not separate arm/leg
controls. Upper-body-only behavior needs explicit lower-body/torso constraints
and validation, plus the installed racket/gripper dynamics and collision model.
The current external-token receiver warns on missing tokens and may retain the
last token; it is not a validated fail-stop path for physical experiments.
