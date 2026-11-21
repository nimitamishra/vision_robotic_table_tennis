# Forehand motion prior and SONIC tracking

The newer physical ball-return experiment is documented in [RETURN.md](RETURN.md).
This page describes the earlier motion-only model, retained for comparison.

This replaces the improvised low-arm interception motion with the public
[table-tennis-motion-kit](https://github.com/dvij542/table-tennis-motion-kit)
forehand demonstrations, pinned at `c6c4c061f8fcb24714fbe118a31e66c69adeb4bf`.
The reference checkout and its examples remain unchanged.

## What is trained

1. `train_prior.py`: a 4 → 128 → 128 → 128 → 64 network learns
   `(contact XYZ relative to starting pelvis, time before contact) → SONIC token`.
   The frozen SONIC encoder supplies labels from ten future poses at 100 ms spacing.
   The frozen SONIC decoder produces all 29 body actuator targets in MuJoCo.
2. `train_tracking.py`: bounded 64-dimensional residual PPO adds current robot
   state and reference-pose error. Rewards track the forehand pose, racket path,
   and orientation; penalties discourage table contact and lower-body motion.
   This tracker additionally requires the reference trajectory. It is not a
   four-input-only online return controller.

The dataset has 20 motions, 126 frames each, at 50 Hz. Entire target trajectories
are split into 16 training, 2 validation and 2 test motions. No adjacent-frame
random split is used. Input targets are anchored at the pelvis at stroke start,
not continually reinterpreted in the moving pelvis frame. X is fixed at 0.43 m;
Y spans -0.53 to -0.33 m and relative Z approximately 0.078 to 0.228 m.
Out-of-range inference is rejected. Time spans +1.0 to -1.5 seconds, including
follow-through and recovery. There is no claim of extrapolation.

## Current result

The supervised model and a 40-update, 320-episode residual PPO experiment are
complete. Validation selected PPO update 30. The extra PPO training gives only
a small improvement over the supervised prior; it is not a major precision gain.

| Evaluation | Result |
|---|---:|
| Full grid, completed strokes | 20 / 20 |
| Full grid, posture-limit failures | 0 / 20 |
| Full grid, racket/obstacle contacts | 4 / 20 |
| Two held-out targets, table-frame contact-position error | 4.29 cm mean |
| Same held-out targets, 0.5 ms physics timestep | 4.30 cm mean |
| Same held-out targets, pelvis-aligned contact-position error | 2.23 cm mean |
| Full-grid table-frame contact-position error | 7.21 cm mean |

Errors are evaluated at the intended 1.00 s contact time. The world/table-frame
metric is the relevant one for a fixed incoming ball; pelvis alignment can hide
base motion. Two test trajectories are a small test, not a reliability estimate.
`runs/tracking-v1/final_audit.json` contains per-motion results and definitions.

This stage is **motion tracking**, not ball return. No physical ball is included
in the displayed stroke, and the green sphere is only a target marker. Some low
motions still touch the table. The robot starts in a reference pose and receives
one second of encoder warm-up; home-to-ready movement is not validated. The
stock G1 dynamics model retains its hands and uses estimated racket inertia.
Reference wrist-to-face offset is 23.65 cm plus a 6.2 mm face offset, which is
not a calibration of the user's installed gripper and racket. Hardware output
is absent and all checkpoints are marked `hardware_approved: false`.

## View

From G1 Studio, select **Forehand motion → Start Token Simulation** after restarting
the app. Alternatively run `../start_forehand.sh` and open http://127.0.0.1:52175/.
The local viewer has four views:

- **Reference · kinematic only**: supplied VICAR reference poses.
- **Encoded reference · MuJoCo replay**: reference encoder → frozen SONIC decoder.
- **Learned tokens · MuJoCo replay**: trained four-input model → frozen decoder.
- **PPO trial · MuJoCo replay**: prior plus the motion-tracking residual policy.

MuJoCo views replay recorded physical rollouts; they are not live control.
Use Y/Z offsets to select a saved target, **Show contact pose** to inspect the
1.00 s pose, and **Play** to see the complete stroke. Camera controls are not
reset while playing or switching targets. The panel uses fixed-height status
content. The table is 2.74 × 1.525 m, 0.76 m high, with a 0.1525 m net.

## Reproduce / continue

Use the existing simulation environment:

```bash
SIM_PY=/home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl/.venv_sim/bin/python
cd /home/jingxuan/Documents/Codex/2026-10-16/ni/outputs/sonic-token-policy/forehand
"$SIM_PY" test_forehand.py
"$SIM_PY" audit.py
"$SIM_PY" train_tracking.py --output runs/tracking-v2 --init runs/tracking-v1/policy.pt --updates 40
```

Warm-starting loads policy weights with a fresh optimizer; it is not an exact
optimizer/RNG resume. Existing tracking outputs are never overwritten. Dataset
provenance and encoder/decoder hashes are recorded in `runs/prior-v1/dataset.json`.
`finish_audit.py` adds fixed-world errors and the half-timestep check for v1.
`render_comparison.py` writes an H.264 comparison video using the existing local
`work/ffmpeg` executable. The reference side of that video uses the same stock
MuJoCo geometry for comparison; the interactive reference view uses the pinned
VICAR visual URDF.

## Research basis

[HITTER](https://arxiv.org/abs/2508.21043) motivates separating contact goals from
motion execution. [SMASH](https://arxiv.org/abs/2604.01158) motivates motion priors
and filtering references by actual tracking performance. This implementation
adapts those ideas to SONIC tokens; it is not a reproduction of either paper's
whole-body training system. Next work should reduce the measured tracking bias,
reject colliding low targets, and then add front-fed ball dynamics and return
objectives without sacrificing the forehand reference motion.
