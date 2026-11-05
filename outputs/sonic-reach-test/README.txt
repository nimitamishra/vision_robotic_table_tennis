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
