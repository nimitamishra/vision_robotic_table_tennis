# Physical forehand return through SONIC

This experiment adds a dynamic ball and changes the task from matching a pose
to returning a front-fed ball over the net onto the opponent's table. The old
motion-tracking experiment is retained separately in `README.md`.

## Evaluated result: return-v4

The completed refinement ran 24 generations and 3,840 training episodes, using
five feeds per candidate and four local simulation workers. Earlier exploratory
runs and their warm-start candidates are retained. The selected parameters were
fixed before the following test sets were evaluated.

| Evaluation | Face hits | Net crossings | Legal returns |
|---|---:|---:|---:|
| Forward-facing seed before RL, same 80 test feeds | 4/80 | 0/80 | 0/80 |
| Learned policy, narrow held-out feeds | 53/80 | 10/80 | 10/80 |
| Learned policy, wider held-out feeds | 20/40 | 1/40 | 1/40 |
| First 20 narrow feeds, 1 ms physics | See report | See report | 2/20 |
| Same first 20 feeds, 0.5 ms physics | 13/20 | 6/20 | 6/20 |

The narrow-test legal-return rate is **12.5%**. This is initial progress, not a
reliable return controller. The 80-feed failures were 37 wrong landings, 29 rim
contacts, two net contacts and two returns missing the table. No posture-limit
or racket/obstacle failures occurred in these held-out tests. Changing the
physics step changes outcomes materially, even though the finer-step count is
higher; contact sensitivity remains unresolved. All ten reported narrow-test
returns passed independent bounce/hit/net/landing sequence checks.

`return_test.mp4` contains the first six consecutive held-out serves; all six
failed. `return_selected_example.mp4` explicitly labels a selected successful
example, test serve #10. The latter is illustrative and must not be used to
estimate the success rate. Both videos use actual recorded dynamic rollouts.

The next controller should adapt to ball state during the stroke, target more
central racket contact, and be trained/evaluated across contact parameters and
timesteps. Simply continuing this fixed stroke search has shown limited gains.

## Controller and learning

The starting forehand is the public `table_tennis_forehand_y+000_z010` reference
from `table-tennis-motion-kit`, pinned at
`c6c4c061f8fcb24714fbe118a31e66c69adeb4bf`. A local inverse-kinematics warm start
turns the racket face toward the opponent while approximately preserving its
contact path. This warm start is not itself reinforcement learning.

Cross-entropy episodic policy search learns nine bounded stroke parameters:
seven right-arm reference offsets, phase lead and time scaling. Candidate
strokes are scored by their physical outcomes in MuJoCo. The refined run tests
each candidate on three randomly sampled feeds before updating the search
distribution. Selection uses a separate eight-feed validation set.

The execution path is:

```
learned stroke parameters
  -> time-varying reference poses
  -> frozen SONIC encoder (64-dimensional tokens)
  -> frozen SONIC decoder
  -> PD torques -> MuJoCo robot and ball dynamics
```

This is an open-loop stroke-parameter policy for a narrow feed curriculum. It
is not PPO and is not yet a neural controller that adapts continuously to live
ball observations. The previous learned four-input token prior is retained but
is not used by this return controller. No policy parameter is sent directly to
an actuator. Robot poses are only initialized at reset; subsequent movement is
produced by dynamics and the SONIC decoder.

## Reward and success definition

| Physical event | Reward |
|---|---:|
| First valid racket-face hit after one own-side bounce | +20 |
| Ball clears the net within table width | +80 |
| Ball then contacts the opponent's tabletop inside its boundaries | +200 |
| Miss, net contact, rim contact, wrong landing or out-of-bounds ball | -8 |
| Posture-limit violation or racket/obstacle collision | -80 |

Small regularizers penalize reference offsets and leg motion. Before a hit,
terminal proximity shaping is at most +2. After a hit, measured outgoing ball
velocity supplies landing shaping of at most +20. This prediction never counts
as an actual return: legal-return labels require physical table contact after
an observed net crossing. Net clearance includes the ball radius. Crossing
alone, a reference target marker, a rim hit and a selected animation do not
count as a legal return.

Events are detected at each physics substep. The ball is assigned an initial
state once at serve reset and thereafter follows gravity and MuJoCo contacts.
The control frequency is 50 Hz; the main physics step is 1 ms, with a separate
0.5 ms sensitivity audit.

## Feed curriculum and limits

The table near edge is X=0.36 m, net X=1.73 m and far edge X=3.10 m;
width is 1.525 m and tabletop height is 0.76 m. +X points toward the opponent.
The ball starts on the opponent half at X=2.05 m, Z=1.15 m, traveling toward
the robot. Narrow training feeds vary Y=-0.38 +/-0.018 m,
VX=-2.5 +/-0.06 m/s and VZ=2.086 +/-0.04 m/s. The wider test doubles these
three variation ranges. The ball first bounces on the robot's half.

The controller does not cover general serves, spin, perception errors or
camera/actuation latency. Racket mass, mount and contact restitution are
provisional. The simulated reference mount is not a calibration of the real
gripper. The full floating robot is simulated, with limits on leg motion and
pelvis drift; it starts from an initialized ready pose with an encoder warm-up.
There is no real-robot transport in this viewer or environment.

## Evaluation artifacts

The active run is selected in `../return-preview.json`. Its directory contains:

- `policy.json`: learned parameters, scales, encoder/decoder/reference hashes.
- `report.json`: all 80 held-out narrow feeds, the same feeds before RL, 40 wider
  feeds, validation and training history. RNG seeds 9917 and 9918 are reserved
  for these final evaluations; validation uses 811.
- `audit.json`: event-sequence checks, a 20-feed half-timestep comparison and
  isolated ball/table restitution diagnostics.
- `test_rollouts/`: every held-out physical trajectory, including failures.
- `return_test.mp4`: the first six consecutive test serves at half speed,
  including failures, rather than a selected success montage.

Read the audit alongside the headline return rate. Sensitivity to the physics
step is a limitation, not evidence of hardware reliability.

## Open and reproduce

In G1 Studio, select **Token Policy Sim -> Forehand return -> Start Token
Simulation**. Restart the app if that stage is not yet listed. Alternatively:

```bash
../start_return.sh
```

Open http://127.0.0.1:52175/. The viewer runs new live physical serves. Use
**Front camera**, **Table overview**, **Serve new ball** and **Replay same serve**.
The green line shows the racket face normal. **Before RL (forward seed)** uses
the same corrected reference without the learned offsets; it is not the old
45-degree reference. Change controller and replay the same serve to compare.

```bash
SIM_PY=/home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl/.venv_sim/bin/python
"$SIM_PY" -m unittest discover -s . -p 'test*.py'
"$SIM_PY" train_return.py --output runs/new-return-run \
  --resume runs/return-v3/refinement_start.json \
    --iterations 24 --population 32 --fitness-cases 5 --workers 4 --seed 773 --local
"$SIM_PY" audit_return.py --run runs/new-return-run
"$SIM_PY" render_return.py --run runs/new-return-run
```

Use a new output directory. Resume is a parameter warm start with a fresh RNG
and search distribution, not exact continuation. `select_return.py` can compare
successful training strokes using validation feeds only. Interrupted exploratory
runs are retained for provenance; only a completed, evaluated run is published
in the preview manifest.

## Measured-start transition diagnostics (2026-11-20)

`runs/transition-20261120/robot-start.json` records a fresh read-only DDS snapshot
of 29 joint positions/velocities and IMU orientation/angular velocity. No real
controller or publisher was started during these checks. `probe_transition.py`
feeds smooth reference transitions through the unchanged SONIC encoder and
decoder, with dynamic PD torque actuation, not animated joint poses.

Eight initial conditions/transition checks were run: holding position with two
history initializations; 2, 4 and 6 second transitions; a 0.5 ms timestep repeat;
a floating-base run; and a 10% reference excursion. None passed the diagnostic
limits. Slowing the full transition alone did not eliminate contact with the
training table. The table is NOT registered to the current hardware location.

`probe_small_lift.py` then tested a 3 cm upward reference with unchanged racket
normal, 4 seconds up and 4 seconds back. Results:

| Diagnostic model | Peak rise | Peak right-arm speed | Peak lower-body joint change | Final right-arm return error |
|---|---:|---:|---:|---:|
| Rigid pelvis support | 6.78 cm | 1.64 rad/s | 25.43 deg | 12.18 deg |
| Floating base | 1.39 cm | 1.23 rad/s | 16.25 deg | 6.69 deg |

Neither run is cleared for physical execution. The supported model approximates
the hoist as a rigid pelvis weld; the unsupported model assumes initial root
height 0.79 m. The mount remains provisional and the stock Dex3 hand remains in
the model. The encoder/decoder history is synthesized from the snapshot, with
both pose-holding and zero prior-action startup histories checked in the initial
hold trials. These limitations prevent attributing all error to the policy
alone. The observations nevertheless rule out claiming this replay is validated
for a slow upper-body-only hardware trial.

`small-lift-diagnostic.mp4` displays the supported run at 1x simulated speed and
labels the failed check. `small-lift-report.json` and the per-run JSON files retain
metrics, contacts, actual states and tokens. Tokens are saved locally only.


### Feet-on-ground clarification and guarded recheck

The operator confirmed hoist support with both feet on the floor. A new read-only
snapshot is saved as `runs/transition-20261120/feet-grounded-live.json`.
`probe_guarded_lift.py --snapshot <snapshot> --output <directory>` uses the compiled
C++ output envelope through a local shared library; it has no robot transport.
The optional hold-other-joints experiment masks non-right-arm targets, while
retaining all measured-state limits. This mode is not enabled in the controller.

The initialization only translates the root until the lowest modeled foot is
1 mm above the floor. In the new snapshot the other modeled foot is 8.31 mm above
it, so **both-foot support is not reproduced**. The report now records individual
foot heights and prevents this condition from passing the diagnostic. The model
also initially contains stock finger/thigh penetrations, although the robot has
a different end effector. A sensitivity run disabling finger collisions while
retaining their inertia still trips the left-ankle velocity check at 18 ms,
before the requested lift begins. It is not a physical gripper model.

`feet-grounded-recheck/guarded-report.json` preserves the results and exact fault
joint. Raise is measured from each run's own initial racket height; the earlier
height-offset artifact from comparing grounded and ungrounded baselines has been
fixed. These startup diagnostics do not validate or disprove a physical stroke.
No physical swing was sent. Next prerequisites are a consistent supported
initial state, measured end-effector geometry, and a start/return transition that
passes the unchanged command envelope.
