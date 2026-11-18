Single-ball Trajectory — Version 1

Use in G1 Studio (English UI)
1. Restart G1 Studio to load the new Trajectory tab.
2. Connect the robot. Click Connect / Start Vision.
3. Confirm that Viser shows a valid 3D ball in table coordinates.
4. Open Trajectory and click Start Trajectory.
5. Blue = filtered recent positions. Orange = future position predictions.
   Velocity is in m/s. Model acceleration is in m/s^2.
SONIC does not need to be running. These scripts never send motor commands.

Standalone (existing measurement publisher on port 5563)
/home/jingxuan/Documents/Codex/2026-10-16/ni/outputs/ball-trajectory/start.sh
Open http://127.0.0.1:8082
G1 Studio uses private ports; start via its Trajectory button when using the app.

Offline synthetic demo (no robot required)
/home/jingxuan/Documents/Codex/2026-10-16/ni/outputs/ball-trajectory/start.sh --demo
The page is explicitly marked SYNTHETIC DEMO. This is not measured accuracy.

Replay
Run start.sh --replay /absolute/path/to/measurements.jsonl
Uses original capture timestamp intervals for filtering. Playback is paced at
60 messages/s for convenience, not original wall-clock speed.

Output
Each run writes config.json, measurements.jsonl, trajectory.jsonl, latest.json.
ZMQ publishes msgpack results, default tcp://127.0.0.1:5564.
Position, velocity, covariance, accepted/rejected measurements, events and
50/100/200 ms predictions are recorded. No forecast extends through the next
contact-height crossing. The crossing is of an infinite plane: table bounds
are not yet checked, so it must not be interpreted as a guaranteed table hit.

Model and assumptions
- Table frame: +x right, +y toward net, +z up. Units: meters and seconds.
- 6-state Kalman filter: position and velocity; gravity input [0,0,-9.81].
- Acceleration is a model assumption, not a measured acceleration signal.
- Contour and box observations have different configurable noise priors.
- Noise defaults are starting values, not empirically calibrated uncertainty.
- Five observations over >=50 ms initialize velocity. Motion changes reinitialize.
- Short losses coast; >200 ms without accepted measurements resets tracking.
- Suspected bounce or consistent rejected motion starts a new track segment.
  Bounce detection is heuristic; no impact coefficient or spin model is fitted.
- Stationary points are handled by a stationary constraint, not gravity.
- Air drag, spin, racket contact classification and actuator latency are absent.
- Timestamps come from capture_monotonic_ns in the existing camera service:
  robot frame receipt, not hardware exposure time. Predictions reference that
  sample, not the current display time. Desktop pipeline age is logged when
  available; it excludes camera and transport delay. Not ready for actuation.
- Valid live table localization is required. A drifting coordinate frame can
  look like ball velocity. Keep the camera/robot fixed for the initial test.
- Input selects the existing upstream single-ball candidate. It gates outliers
  but does not redo all left/right candidate matching using trajectory priors.
- Old image-only recordings need to be processed into valid world measurements
  first. They are not silently treated as 3D ground truth.

Initial robot validation
Keep the robot fixed. First hold a single ball still, then toss it gently, then
bounce it on the table. Check static speed, free-flight continuity, detection
loss behavior and bounce reset separately. Record new single-ball data before
tuning noise parameters or evaluating future errors against measured positions.

Tests
Run the vision Python interpreter on test_tracker.py.
Tests cover noisy ballistic forecasts, outliers, missing observations, timestamp
ordering, static points, contact cutoff, bounce reset and initialization gaps.
Synthetic test accuracy is not a claim about camera/robot accuracy.

DISPLAY UPDATE (2026-10-22)
Default maximum forecast is now 0.5 seconds, adjustable from 0.05 to 1.0 s
using Forecast horizon (s), or --horizon. It still ends at contact height.
A dense curve includes the final segment even when contact is <50 ms away.
Longer horizons are model extrapolation, not an increase in measured accuracy.
Blue historical segments stay visible for one second; they never join across
track resets. Orange is the current forecast; gray is the current raw 3D point.
Input status and left/right candidate counts explain missing measurements.
No-valid-3D and velocity warm-up are distinct states. Filter noise, rejection
thresholds and max coasting time have not been loosened to hide input losses.

COLLISION FORECAST UPDATE (supersedes the earlier plane-only description)
Enabled by default with the Collision forecast checkbox.
Uses the SAME finite table dimensions, position and net height as the viewer.
Orange: flight before contact. Red marker: first predicted contact ball center.
Purple: one estimated tabletop rebound. Blue: recent trajectory history.
Predicted contact shows type and time relative to the input sample, not the
current wall clock. Increase Forecast horizon to 0.8 s to inspect more of a
rebound; longer horizons have greater model error.

Restitution starts at 0.8; horizontal velocity retention starts at 0.9. These
are illustrative, adjustable values, NOT fitted to this table or these balls.
Record real bounce trajectories before treating rebound positions as accurate.
Contact calculations include a 0.02 m radius. For tabletop contact the stored
ball_center_m is 2 cm above the table and surface_point_m lies on the table.
Net and table-edge tests conservatively expand the geometry by the ball radius:
net_risk and table_edge_risk are possible contacts, not exact corner solutions.
Their forecasts stop there. No net deformation or edge rebound is invented.
A miss beyond the table continues toward the floor. The first floor contact or
second tabletop contact stops prediction; only one rebound is modeled.
Rackets, the robot body, spin, air drag and learned impact parameters are absent.

JSON output adds collision_events, collision_segments, collision_model,
collision_forecast_duration_s and collision_forecast_limit. Each rebound event
stores the actual coefficients used. Legacy predictions/forecast_path remain
pre-contact ballistic outputs; use collision_segments for the extended path.
Uncertainty after collision is not propagated/calibrated in this first version.
Disable Collision forecast to return to the previous plane-limited forecast.

Validation: 10 collision tests plus 8 existing tracker tests pass. They cover
finite bounds, edge risk, net travel in both directions, net clearance, contact
height/time, shifted tables, rebound parameters and forecast termination.
Viewer checked with synthetic data; real impact accuracy is not yet validated.

INTERCEPT TEST
In G1 Studio, restart only Trajectory (Stop, wait 2 seconds, Start).
Expand Intercept Test in the Trajectory sidebar. Default plane is 0.20 m from
the near table edge toward the net. It is a test plane, not a reachable racket
workspace. A translucent green rectangle marks it; green point is the prediction.

Keep the robot and camera fixed. Keep AprilTags visible. Before testing, hold the
ball still on the table and check 3D localization. Do not start SONIC for this test.
Choose Number of throws (default 20), then click Start Intercept Test once.
Each trial has 3 seconds GET READY followed by 5 seconds THROW ONE BALL.
Throw once from the far side toward the robot, with one table bounce. Wait for
the next trial before throwing again. Empty or missed trials are failures too.
Stop & Save Test ends early; an active unfinished trial is logged as cancelled.

A reference crossing needs two raw stereo points straddling the plane in the
incoming (-y) direction, no more than 100 ms apart and plausible speed <=25 m/s.
The finite window is table width, ball-center height 2 cm to 1.2 m above table.
Crossings outside it are reported separately. Long missing intervals are not
silently interpolated. The first valid crossing completes that trial.

For each observed crossing, select the latest saved prediction issued at least
100, 200, or 300 ms earlier (maximum extra age 50 ms). No future observations
are used to generate that prediction. Lead times refer to frame-receipt timestamps,
not actuator-available lead times; desktop pipeline age is saved separately.
No prediction in a lead window stays a missing result, not a zero error.

Results appear in runs/<run>/intercept-<session>/trials.jsonl and summary.json,
with config.json describing the plane and schedule. Means are calculated only
for evaluated in-window crossings; completed-attempt and evaluated counts are
both retained. Full measurement and trajectory logs remain in the parent run.
These are consistency errors against later camera measurements, NOT independent
physical accuracy. Optional slow-motion side video helps check physical errors.
The software does not verify that a throw occurred, or enforce exactly one
bounce: follow the timed protocol; skipped throw windows count as failed attempts.

EXPERIMENTAL OCT 30 IMPACT FIT
After restarting Trajectory, "Try session impact fit (0.90 / 0.60)" applies a
session-specific rebound estimate. Defaults are unchanged until clicked.
It was fitted from three impacts and needs fresh real-throw validation.
See analysis/REPORT.txt for replay comparisons, coverage, and limitations.
Bounce recovery now fits three post-impact samples spanning at least 50 ms;
it no longer includes a pre-impact sample in the same ballistic fit.

TABLE CONTACT BOUNDARY FIX (2026-10-30)
Collision forecasts now project up to 5 mm of estimated penetration onto the
ball-center contact plane, only within the expanded finite table footprint.
Descending or resting states at contact produce an immediate table/edge event;
ascending states retain their upward velocity and forecast the next descent.
Larger penetration remains invalid; states outside the table are not snapped.
This changes forecast state only, not raw observations or the tracking filter.
The applied adjustment and tolerance are included in collision_model metadata.

Frozen-state replay of intercept-20261030-160646-778080:
Trial 7 at 100 ms: position error 47.47 -> 0.94 cm.
100 ms mean: 12.05 -> 5.40 cm, evaluated count unchanged at 7.
200/300 ms metrics unchanged. The two earlier full test sessions have no changed
selected evaluations. These are camera-consistency replay results, not new live
accuracy measurements, and do not resolve missed observations.
31 unit tests pass, including exact contact, shallow/deep penetration, ascending
motion, table-edge risk, and outside-table trajectories.
Reproduce with check_contact_fix.py SESSION TEST --output REPORT.json.

OFFLINE PREDICTION BENCHMARK
See BENCHMARK.md for measurement replay, held-out uncertainty evaluation,
the virtual-plane protocol, and a reproducible synthetic example.
