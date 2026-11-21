# SONIC token policies: front-table simulation

This package trains and deploys a neural **64-dimensional SONIC token policy**
inside the existing G1 MuJoCo model. It does not contain a real-robot transport.

## Current task: front table tennis

Use **G1 Studio → Token Policy Sim → Front table tennis → Start Token Simulation**.
This is now the default when `front-deployment.json` exists. It is an experimental
simulation, not a reliable rally policy or a physical deployment.

The scene contains a 2.74 × 1.525 m table at 0.76 m and a 0.1525 m net. The robot
faces the table; the ball starts on the opponent's side, crosses the net, and
bounces on the robot's half before interception. Collisions with the actual
MuJoCo table, net, racket, handle and robot are scored. A legal return requires a
forced face contact, net clearance, and a physical landing on the opponent's half.

Every episode first runs a **learned lifting model** from the low initial pose.
It was trained by behavior cloning from SONIC-encoded motion demonstrations.
Its 132 observations produce an absolute 64D token. It raises the racket centre
from about 51 cm to 94.7 cm. The subsequent PPO policy uses 141 observations and
64D token residuals (scale 0.35), with a maximum residual change of 0.1 per 20 ms.
Both feed the frozen SONIC decoder; no direct IK or joint-target override is used
at runtime. IK only generated reference demonstrations for the lifting teacher.

Physics runs at 1,000 Hz, policy/SONIC at 50 Hz. Front-table contact parameters
are separate from the older side exercise. Ball-drop numerical checks and a
half-timestep evaluation are recorded in `audit.json`; they do not calibrate
real ball/rubber/table restitution. Mount geometry, mass and inertia are still
provisional, stock Dex3 remains, and SONIC can move all 29 body joints to balance.

The front viewer uses live inference and physics. **Lift and serve new ball**
starts a new attempt; **Replay same ball** repeats the feed; **Hold raised racket**
is the constant-token comparison. **Front camera** and **Overview camera** change
only the view. Playback speed affects display timing, not simulation dynamics.
All controls are in English. The separate **face contact**, **over net** and
**legal landing** indicators prevent a touch from being presented as a good return.

```bash
bash /home/jingxuan/Documents/Codex/2026-10-16/ni/outputs/sonic-token-policy/start_front.sh --port 8092 --autoplay
# In the simulation Python environment:
python train_front.py --output runs/new-front-run --updates 120
python train_front.py --output runs/new-front-refinement --init runs/new-front-run/policy.pt --updates 140 --return-shaping
python audit_front.py --checkpoint runs/new-front-refinement/policy.pt
python deploy_front.py runs/new-front-refinement
```

The deployment command only selects a checked **local experimental viewer**.
It rejects posture-limit failures and checkpoint mismatches. Even with zero
legal returns, the model may be inspected as an experimental checkpoint; the
viewer and report explicitly show the actual result. No hardware transport is
present. Old reach and side-contact tasks remain available for comparison.

## Archived side-contact curriculum

`impact_env.py` adds a mass-carrying racket and a free ball to MuJoCo. The
new actor takes 141 features and still outputs 64 token residuals through the
same frozen SONIC decoder. Its additional inputs are relative ball position,
ball velocity, the initial racket-plane normal, and predicted time to that
plane. The intercept is computed from current ball state and gravity, without
reading future simulator states. Physics runs at 500 Hz and control at 50 Hz.
The configuration's `incoming_speed_m_s` is a launch-construction parameter
(normal launch offset divided by flight time), not the measured speed at
contact. Gravity contributes to the actual approach velocity at the tilted
racket plane; contact events record the simulated pre-impact velocity.

The contact exercise uses lateral feeds at offsets of 4–12 cm from the initial
racket centre, with 0.65–0.95 seconds until interception. This is deliberately
not a table rally: the starting racket is approximately 51 cm high and faces
sideways. The provisional attachment uses 12 cm wrist-to-grip plus 15 cm
grip-to-face, a 15 cm face diameter, and an assumed total racket mass of 0.20 kg.
Its orientation, mass, inertia and contact properties have not been measured.
The stock Dex3 geometry is still present.

A successful return requires actual positive MuJoCo face-contact force,
incoming normal velocity, then outward velocity above 0.2 m/s. Rim/handle/body
contacts do not count. The score does not require landing on a table or in a
target zone. Robot height, tilt, translation and lower-body excursion checks
remain active; passing them is not physical safety certification.

Training and deployment are separate:

```bash
# Use the simulation virtual environment shown below.
python train_impact.py --output runs/new-contact-run --updates 100
python audit_impact.py runs/new-contact-run
python deploy_impact.py runs/new-contact-run
```

The deployment command rejects a policy that fails posture checks or does not
beat both constant-token and initial-policy test return rates. The prior reach
deployment is retained in `deployment.json`; contact has its own
`impact-deployment.json`. In **G1 Studio → Token Policy Sim**, select **Ball
contact**, then **Start Token Simulation**. Stop the simulation before switching
back to **Short reach**.

Standalone, after promotion:

```bash
bash /home/jingxuan/Documents/Codex/2026-10-16/ni/outputs/sonic-token-policy/start_contact.sh --port 8092 --autoplay
```

The viewer offers **Controller**, **New ball**, and **Replay same ball**. Change
the controller and replay the same ball for a fair visual comparison. Playback
defaults to half speed; this changes display timing, not physics. All UI text
is English. No robot connection is required.

`runs/impact-v1` is an interrupted development run with an obsolete scoring
version and is not deployable. The corrected environment freezes its intercept
estimate after any collision, preventing a floor bounce from changing the
miss penalty. Its checkpoint fingerprint differs from the old environment.

## Archived short-reach result — 2026-11-20

The selected run is `runs/reach-v2`, PPO update 70 (63,000 steps in this run;
117,000 PPO steps across both experiments). Selection used validation goals,
not the test set. The runtime reads `deployment.json` to find the checkpoint.

| Method | Test goals | Mean hand error | Below 1 cm |
|---|---:|---:|---:|
| Constant nominal token | 100 | 1.97 cm | 2% |
| Measured local-response rule | 100 | 1.83 cm | 29% |
| Neural warm start | 100 | 0.79 cm | 70% |
| Selected warm start + PPO | 100 | 0.74 cm | 73% |

The selected model's 95th-percentile error is 1.60 cm. A separate initial-state
recording gave 0.68 cm mean error across 40 additional goals. No episode triggered
the configured termination checks. The maximum leg/waist joint change in the
100-goal test was 0.052 rad (about 3 degrees); this is not a strictly immobilized
lower-body controller. These are ideal-state simulation results, not real-ball
or real-robot measurements.

## Run the archived short-reach model

In G1 Studio, select **Token Policy Sim → Start Token Simulation**.
The pane runs the policy and MuJoCo live; it is not a prerecorded animation.
Use **Run selected target**, **Try random target**, or **Pause / Resume**.
Target offsets are in centimetres, with total displacement limited to 1–3 cm.

Standalone:

```bash
bash /home/jingxuan/Documents/Codex/2026-10-16/ni/outputs/sonic-token-policy/start.sh --port 8092 --autoplay
```

Open http://127.0.0.1:8092/ on this computer. Stop with Ctrl+C.
The service binds only to loopback. It neither starts SONIC on the robot nor
publishes DDS/ZMQ motor commands. G1 Studio's physical SONIC controls are separate.

## What is trained

The first curriculum is **short right-hand target reaching**, starting from a
settled SONIC standing posture. There is no ball input, contact reward, or trained
return stroke at this stage.

The actor takes 131 features: desired hand offset (3), hand error (3), body joint
position offsets (29), joint velocities (29), projected gravity (3), and previous
token residual (64). Its two 128-unit hidden layers output 64 token residuals.
Each component is clipped to ±1, changes by at most 0.1 per 20 ms, and is scaled
by 0.15 before adding a nominal token recorded from SONIC.

The frozen released SONIC decoder consumes that token and ten frames of robot
history (994 inputs) and outputs all 29 body joint targets. The simulator uses
the same joint order, default positions, action scaling, and PD gains as the
C++ deployment controller. No inverse-kinematics targets or direct right-arm
servo overrides are used during these training rollouts.

Training uses a measured local token-response warm start, followed by PPO.
Checkpoint selection uses 24 validation goals. A separate 100-goal test set is
evaluated only after selection, against a constant-token baseline and the local
response rule. If PPO fails to improve validation performance, the warm-start
checkpoint is retained and `best_update` is zero; this must not be reported as a
PPO improvement.

## Artifacts

- `deployment.json`: selected local simulation deployment.
- `runs/reach-v2/policy.pt`: selected checkpoint and decoder fingerprint.
- `runs/reach-v2/report.json`: training history, held-out trials, limitations.
- `runs/reach-v2/rollouts.json`: recorded test trajectories for inspection.
- `runs/reach-v2/shifted_state_eval.json`: separate initial-state evaluation.
- `runs/reach-v1/`: preserved first-run baseline and PPO comparison.
- `runs/decoder-seed/decoder_parity.json`: numerical comparison against C++ SONIC.
- `runs/probe/response.npz`: experimentally measured local token response.
- `runs/decoder-validation/seed.npz`: independently collected simulation state.

Training:

```bash
/home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl/.venv_sim/bin/python \
  /home/jingxuan/Documents/Codex/2026-10-16/ni/outputs/sonic-token-policy/train.py \
  --output /path/to/new-run --updates 60 --episodes 12
```

Use a new output directory; existing training runs are not overwritten.
Runtime dependencies include MuJoCo, PyTorch, NumPy, SciPy, ONNX Runtime 1.23.2
and Viser 1.1.1 in the existing simulation virtual environment.

## Scope of validation

The floating base is not pinned. Leg, waist and left-arm disturbance is penalized,
but those joints are **not locked**. Episode termination checks height, root tilt,
root translation and leg/waist displacement. These checks are not a validated
collision or hardware safety system.

The stock model has Dex3 hands instead of the actual two-finger gripper. The
measured/provisional racket offsets have not been converted into a validated
mass, inertia and collision model. The orange marker is SONIC's hand tracking
point, **not** the physical racket centre. Do not interpret reaching error as a
ball-hit rate or physical positioning accuracy.

Before robot deployment: validate the actual gripper/racket model and poses,
full-state observations, token freshness and stop behavior, then test bounded
motions under supervision. The current checkpoint explicitly contains
`simulation_only: true` and `hardware_approved: false`.
