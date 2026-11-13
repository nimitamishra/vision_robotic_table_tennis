# G1 right-hand ready pose: simulation results

The intended hand tracking point was at the near table edge, 20 cm above the
table. SONIC's point is offset from the wrist; it is not the racket center.
These are single MuJoCo loopback trials. No real robot motion was commanded.

| Input and start | Result | Final right-hand error | Both feet grounded |
| --- | --- | ---: | ---: |
| Direct 3D target, default simulated start, continuous feedback | Failed | 13.23 cm | 100% |
| Direct 3D target, default start, correction frozen during reach | Failed | 17.30 cm | 100% |
| First 5 cm waypoint, default start | Passed | 0.14 cm | 100% |
| 5 cm waypoints toward full target | Failed at waypoint 2 of 11 | 2.73 cm at failure | 100% |
| Upper-body joint target, default start, initial IK | Failed strict 2 cm criterion | 2.07 cm | 100% |
| Upper-body joint target, default start, adjusted IK | Failed strict 2 cm criterion | 2.21 cm | 100% |
| Upper-body joint target, default start, midpoint IK | Passed | 0.77 cm | 100% |
| Upper-body joint target, initialized with measured live arm joints | Failed | 3.71 cm | 100% |
| Measured arm joints commanded from SONIC startup | No reach: startup pose drifted and target exceeded the 60 cm guard | — | Grounded during startup |
| Measured arm joints and hand points commanded from startup | No reach: initial posture could not be held | — | Grounded during startup |
| Measured arm joints with bounded startup feedback | No reach: standing joint gap remained 0.22 rad after 12 s | — | Grounded during startup |
| Live arm pose pinned only while SONIC loaded; INIT preserves measured upper body | No reach: standing joint gap remained 0.47 rad | — | Grounded during startup |
| Same startup with bounded joint feedback | No reach: gap fell briefly to 0.12 rad, then rose to 0.25 rad | — | Grounded during startup |
| Same startup with initial hand-point targets | No reach: standing joint gap remained 0.38 rad | — | Grounded during startup |

Raw telemetry and reports are under `runs/ready-edge-*`. The main comparison
files are [direct 3D target](runs/ready-edge-trial01/report.json),
[first 5 cm waypoint](runs/ready-edge-first-step01/report.json),
[default-start joint target](runs/ready-edge-joint03/report.json), and
[live-start joint target](runs/ready-edge-from-live01/report.json).

Pass means the full final second stayed within 2 cm of the requested tracking
point. For the passing full-target trial, the simulated right arm moved over an
8-second ramp and held for 5 seconds without virtual support. The target was
`[0.4825, -0.0649, 0.1740]` m in the default simulated pelvis frame, and the
successful right-arm joint goal was
`[-0.6319, 0.00885, -0.01245, 0.4745, -0.00305, 0.0274, -0.03775]` rad.

After power was restored, the camera and joint-state services were restarted
and G1 Studio vision was restarted. A fresh live pose snapshot is saved in
`runs/ready-edge-live-snapshot.json`. The model-predicted current hand point
agreed with the observed point within 0.79 cm. In that new frame, the same
table-edge candidate became `[0.4942, -0.1191, 0.1869]` m relative to the
pelvis, roughly 70.0 cm from the actual right hand. Position-only IK found a
joint solution with 0.03 cm model error, but it requires up to 0.87 rad change
in one joint. That is not a tested movement trajectory.

The live-start simulation moved the arm away from its initial joint pose during
SONIC startup. Even when the initial upper-body joint angles and three tracking
points were transmitted from startup, the maximum joint gap was about 0.94 rad
during startup and remained about 0.49 rad at the standing checkpoint. A bounded
joint correction reduced the standing gap to 0.22 rad, still above the 0.15 rad
gate, with some correction channels at their limit. The live-start reach without
initial-pose conditioning then missed by 3.71 cm. The default-start pass therefore
does not validate switching on SONIC or moving the physical robot from its
current posture. The simulator has no table or racket collision geometry and
these trials do not validate orientation, repeatability, or hitting a ball.

The next checks are to hold the measured start pose and then verify that a
short hand movement actually occurs. Increasing external correction limits
alone is not justified by these trials.

## Startup isolation (12 November)

The controller's INIT state always ramps its joints toward default standing
angles. An opt-in `--sim-init-hold-upper-body` flag now captures the measured
waist and arm angles once during INIT. The flag is rejected unless DDS runs on
`lo` and CRC checking is disabled, and it is only passed by this simulation
runner when `--hold-initial-upper-body` is selected. The simulator also pins
its seeded upper-body pose while TensorRT loads; without control torques, that
pose had fallen by about 0.85 rad before SONIC started. The loading-stage
drift is now below 0.01 rad.

The remaining failure is at the controller handoff. During INIT, the right
shoulder pitch settled about 0.25 rad away from the preserved target with
Kp around 14.3 and roughly 3.6 Nm of shoulder torque. In the standing policy
phase, SONIC commanded shoulder pitch around 0.36 rad while the input target
was 0.107 rad. The upper-body planner input conditions the learned policy; it
does not directly set motor position. The right-arm gap at the standing gate
was 0.47 rad with only the joint target, 0.38 rad with both the joint and hand
point targets, and 0.25 rad with bounded joint feedback. The bounded feedback
briefly reached 0.12 rad, then lost it, so a momentary pass is not stable.

These runs are in `runs/ready-edge-init-hold03`,
`runs/ready-edge-init-hold-feedback04`,
`runs/ready-edge-init-hold-vr05`, and
`runs/ready-edge-init-diagnostic06`. No reach was issued because the
simulation-only 0.15 rad initial-pose gate failed. The original 0.77 cm pass
still applies only to SONIC's default simulated start. Table and racket
collision geometry is still missing. No physical motion was commanded.

The trace in `runs/ready-edge-target-trace07` confirmed that SONIC received
the upper-body target (`right_shoulder_pitch=0.107439` rad), so the persistent
error is not a lost ZMQ message.

## Simulation-only hybrid right-arm test

An additional opt-in loopback-only flag, `--sim-right-arm-direct-servo`, lets
SONIC control the body while the right-arm motors track the transmitted joint
targets with 2x the nominal position gain. It is rejected without the
loopback-only INIT hold flag. This is an experiment, not a real-robot control
path. It held the measured initial right arm within 0.023 rad at the standing
gate, without virtual support. The full ~70 cm table-edge move then missed by
15.34 cm; both feet were in contact for 99.6% of the reach. The interpolated
joint path also produced large intermediate Cartesian errors.

The offline position-only IK planner `plan_short_reach.py` generated a 2 cm
forward test goal from the stable hold; it required at most 0.035 rad of joint
change. This exposed an evaluation flaw: the old "within 2 cm" test can pass
even if a 2 cm command barely moves. Reports now include target-direction
movement completion and a stricter short-reach result (at least 70% movement,
no more than 1 cm error, and 99% two-foot contact). The tests were:

| Simulated ~2 cm move | Movement along target | Final hand error | Short-reach result |
| --- | ---: | ---: | --- |
| Direct right-arm servo, no feedback | 11% | 1.73 cm | Failed |
| Bounded joint feedback | 61% | 1.96 cm | Failed; lateral drift |
| Bounded Cartesian feedback | 129% | 1.90 cm | Failed; lateral/vertical drift and correction saturation |

The raw runs are `runs/ready-edge-hybrid09`,
`runs/ready-edge-hybrid-short10`,
`runs/ready-edge-hybrid-short-feedback11`, and
`runs/ready-edge-hybrid-short-cart12`. None validates a physical reach. A
controller that follows a bounded Cartesian hand path, an actual table/racket
collision model, and repeatable no-support simulation trials are still needed
before a real 2 cm movement. No physical motor command was sent.
The static plot `runs/ready-edge-hybrid-short-cart12/trajectory.png` shows the
forward overshoot and lateral/vertical drift without requiring browser WebGL.

## Short reach with self-collision screening (12 November)

The recorded right hand starts in light contact with the right hip. Replaying
the old 2 cm **forward** joint path through the full MuJoCo collision model
showed up to 16.76 mm of right-wrist/hip penetration. The new online IK planner
rejects that path before issuing a reach (`runs/ready-edge-forward-guard19`).
The original misses were therefore partly caused by an invalid path, not just
position-control error. A nominal 2 cm **outward** path was clear to within
0.2 mm in the same model, so it became the next controlled test.

The simulator now recomputes the right-arm IK goal after standing, logs actual
arm/body contacts and motor torques, and rejects a proposed joint path if its
modeled arm penetration exceeds 1 mm. An opt-in MuJoCo-only arm controller adds
60 Nm/rad of proportional gain, 3 Nms/rad of damping, and MuJoCo bias-torque
feedforward. A bounded Cartesian correction closes the remaining hand-point
error. None of these extra torques is present in the physical SONIC controller.

The **2 cm outward** move passed 10/10 fresh loopback simulation runs using
the stronger criterion of at least 90% target-direction movement, at most 1 cm
error throughout the final second, at least 99% two-foot contact, and no more
than 1 mm right-arm/body penetration. Observed movement was 98.5–101.5% of
the target; worst final-second error was 0.31 cm, worst modeled arm penetration
0.24 mm, and both feet remained in contact. The highest simulated right-arm
motor torque was 10.8% of its limit. See
`runs/ready-edge-outward-validation.json` and individual runs
`runs/ready-edge-outward-closed24` through `33`. The final run's static trace is
`runs/ready-edge-outward-closed33/trajectory.png`.

An offline replay against the measured table dimensions and net found no
right-arm/table or right-arm/net contact in these 10 short outward runs. It
uses the saved table-to-pelvis pose from `runs/ready-edge-live-snapshot.json`,
and the table did not
exert forces in the original simulation. Reports are saved as
`table_contact_audit.json` in each run. A fresh pose is needed after any robot
or table movement. There is no racket model because the racket is not mounted.
This result validates one small direction from one starting posture in the
simulator; it does not authorize a physical reach or ball strike.
