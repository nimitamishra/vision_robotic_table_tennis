# First policy: ball state to SONIC hand target

## Architecture

The first learned policy observes one ball and the current robot state. It
outputs a **bounded three-dimensional correction to the right-hand target**,
not seven joint commands or arbitrary bytes. A deterministic adapter converts
that correction to SONIC's existing VR three-point target packet. SONIC remains
the only whole-body controller in the simulator.

The policy changes only the right-hand target. The left-hand and head targets
and all three target orientations are held at their captured ready-pose values.
This means the POLICY commands only an upper-body task. The current SONIC binary
still controls the full G1 and may move the lower body for balance; it does not
provide a guarantee that legs remain stationary. If rigidly fixed legs are a
hard requirement, that is a different arm-only controller experiment and its
joint outputs cannot be sent directly through this SONIC VR interface.

## Model interface, version 1

Use robot/pelvis coordinates inside the policy. The table-frame ball state is
converted using the current robot pose, with timestamp alignment. Observation:

- Ball position and velocity (3 + 3), predicted intercept location (3), and
  time to intercept (1).
- Observation age/confidence and whether a reliable 3D estimate exists.
- Actual right-hand tracking point and its velocity (3 + 3), current pelvis
  pose/velocity, and the previous hand-target command.

Start with a small MLP. The action is `delta_right_m = (dx, dy, dz)` relative to
an independently chosen ready-pose or baseline intercept target. Clip actions
to a measured reachable region, apply speed/acceleration limits, and smoothly
update at a fixed rate (first test: 20 Hz). The adapter resends targets to SONIC
at the tested planner cadence (50 Hz). An old or invalid ball observation holds
or cancels the target; it never extrapolates indefinitely.

SONIC receives a 9-float VR position vector in order: left wrist tracking
point, right wrist tracking point, head/torso tracking point. It also receives
three relative quaternions (12 floats, wxyz order) for those points. These are
**pelvis-relative tracking targets**, not absolute table coordinates and not
the physical racket center. The existing right-wrist tracking offset is
`(0.18, +0.025, 0)` in its wrist-link frame; the racket transform must be
measured separately. The fixed left/head target values are captured from the
same ready pose as the right-hand baseline.

## Build and train in stages

1. Map reachable hand targets with the current SONIC simulator. The existing
   two-trial 5 cm tests passed right/up and failed left/down. Improve the ready
   pose and repeat several points in all directions. Track hand/orientation
   errors, response time, foot contact and pelvis motion.
2. Add table, ball and calibrated racket collision to a resettable simulator.
   Validate ball bounce and racket contact timestamps before using hit rewards.
3. Implement a deterministic baseline: convert a predicted ball crossing into
   a reachable right-hand target, command it through the same SONIC adapter,
   and measure contact and legal-return rates.
4. Train the small policy to correct that target. Begin with slow balls in
   reachable cells. Add launch diversity, observation noise, latency, dropout
   and physics variation. The actor may see only state available at runtime;
   a training critic may see full simulator state.
5. Evaluate against the baseline on identical held-out launches, counting
   misses and missing 3D estimates. Report attempts, valid observations,
   contact rate, legal-return rate, falls, target clipping and latency.

Reward legal returns more than simple contact; reward contact more than merely
reducing distance. Penalize falls, collisions, impossible commands and abrupt
target changes. Do not claim table-tennis success from a hand-target reward
before ball/racket contact physics is validated.

## Practical constraint

The current desktop runs a real-time MuJoCo + external SONIC controller test.
That is suitable for interface validation but likely too slow for large-scale
RL rollouts; measure episode throughput before choosing a training backend.
NVIDIA's Isaac Lab SONIC training environment is not installed locally and
upstream training preflight fails under system Python 3.12 (requires 3.11).
The workstation has an RTX 5080 Laptop GPU with 16 GB VRAM.

## Sources

- Current SONIC sender:
  `gear_sonic/utils/teleop/zmq/zmq_planner_sender.py`.
- Current C++ VR target normalization:
  `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`.
- Google DeepMind, *Achieving Human Level Competitive Robot Table Tennis*:
  https://deepmind.google/research/publications/107741/
- NVIDIA SONIC training setup:
  https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/getting_started/installation_training.md

## Training run completed (simulation only)

The code here includes a preliminary fixed-pelvis, right-arm MuJoCo training
environment. It removes every moving joint except the seven right-arm joints.
The policy's 3D right-hand target is executed through a damped-Jacobian IK and
joint PD surrogate. **This is not the full SONIC plant** and the ball flight is
analytic: it has no physical ball/racket contact, table bounce or legal-return
reward. The policy must not be described as a trained table-tennis hitter.

Run with the existing SONIC simulation Python environment:

```
<sim-python> train_ball_proxy.py --output runs/a-new-directory --updates 80 --episodes-per-update 32
<sim-python> -m unittest discover -s . -p 'test_*.py'
```

`runs/ball-proxy-02/policy.pt` is the ball-state checkpoint. The model receives
seven joint positions/velocities, current right tracking point, ball position
and velocity, time to crossing, and the predicted crossing point (27 floats in
total). It outputs a normalized 3D hand target within +/-5 cm of the ready pose.
The pure `policy_adapter.py` validates state and target bounds, rate-limits the
right-hand command, and serializes the existing SONIC 9-float position +
12-float quaternion VR packet without sending it.

PPO used a known ballistic-target warm start and auxiliary target loss. On 200
held-out analytic flights, the trained policy's mean wrist-to-ball-center proxy
error was 0.64 cm and 199/200 were within 2 cm. The deterministic predicted
crossing target was better by mean error (0.44 cm, 197/200 within 2 cm), so this
run is proof of a working training/inference pipeline, not an RL advantage.
Earlier ball-only-input training without the predicted crossing achieved only
3.06 cm mean error over 80 episodes; that checkpoint is not selected.

A separate 21-input static-target policy was tested through the real SONIC
binary in loopback MuJoCo on one 2.5 cm rightward reach: 1.20 cm final-second
mean error versus 0.83 cm for the deterministic target under the same setup.
The 27-input ball policy has been serialized to the SONIC VR message format,
but it has **not yet run with a physical ball or with SONIC responding to live
ball observations**. The current full-body SONIC harness needs a validated
racket/ball simulation before contact-based RL is meaningful.

## Open the ball-policy simulation demo

The demo records six deterministic, held-out ball flights through the trained
27-input policy and the MuJoCo fixed-pelvis right arm, then plays them in a
browser. The transparent table is only a visual reference; it is not a collider.
It does not start SONIC, connect to the robot, or simulate racket contact.

```
/home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl/.venv_sim/bin/python demo_rollouts.py
/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/work/vision-venv/bin/python demo_viewer.py runs/demo-rollouts.json --port 8091
```

Open `http://127.0.0.1:8091/`. The scene loops through all six balls. Use
**Pause** and **Next ball** to inspect one. The displayed final distance is the
simulated right-hand tracking-point distance to ball center, not a hit rate.

## SONIC connection status

`packet_preview.py` loads the 27-input checkpoint, reads one simulated arm and
ball state, converts them to pelvis coordinates, and serializes the proposed
right-hand target into SONIC's 9-position/12-orientation planner packet. It
opens no network socket and sends no command. Run it with the simulation Python
environment above; the JSON result is saved to `runs/packet-preview.json`.

The live Trajectory stream publishes ball position/velocity in the table frame.
The read-only G1 joint-state relay now publishes the seven right-arm positions
and velocities as well as waist yaw. The Viser process aligns those readings to
camera timestamps, computes a pelvis pose from Tag 5, and publishes the pelvis
transform and right wrist tracking point on localhost port 5565. It withholds
the pose when the tag is stale and marks whether both cameras see Tag 5.

`live_shadow.py` joins that pose feed with the live Trajectory feed by the
robot's capture clock. It converts the ball into pelvis coordinates and writes
a bounded model target preview only for an accepted, approaching ball crossing
the captured right-hand ready plane within 0.5 s. It also withholds previews
when a predicted table/net collision occurs first or when the actual right-arm
joint posture differs substantially from the policy's training ready pose.
It captures the physical ready wrist position only after the arm has stayed
near the training posture and nearly still for 0.3 s; moving away clears that
capture. Until then the process records diagnostics without running the policy.
The Trajectory filter uses stereo disparity to give far-away range measurements
less weight; its pixel-error assumptions are provisional, not a calibrated
accuracy estimate. During a short stereo dropout it may coast through one
predicted table bounce, but such a point remains unmeasured and cannot trigger
the policy preview until a new stereo observation is accepted. The shadow process opens **no SONIC socket**
and sends **no motion command**. Its `latest.json` and `preview.jsonl` show why
any frame was withheld. The G1 Studio trajectory port is allocated at launch;
pass the current port with `--trajectory tcp://127.0.0.1:PORT`.

The physical right arm is currently in a substantially different pose from
the fixed-arm training ready pose, so any live preview is an interface check,
not a validated action. The crossing plane is a wrist-plane proxy, not a
calibrated racket contact surface. The physical racket offset, reachability,
latency, and full-body SONIC response still need testing before any target can
be sent to the robot.
