# Hitting-plane inputs to SONIC in simulation

This experiment connects **[target X, target Y, target Z, time to hit] → learned 64D token → frozen SONIC decoder → torque-driven MuJoCo robot → physical ball contacts**. It has no robot transport or motor publisher.

## Inputs and model

Coordinates are metres in the pelvis frame **frozen at episode initialization**: +X forward, +Y left, +Z up. The root starts at world Z ≈ 0.782 m. Thus input Z = 0.178 means a world target height ≈ 0.960 m, not 0.178 m above the table.

- X is fixed at 0.43 m; it is not a freely movable hitting plane in this curriculum.
- The prior's supported Y/Z range is −0.53…−0.33 m / 0.078…0.228 m. That is an input domain, not a guarantee of successful returns.
- The episode accepts an initial time to hit of 0.8…1.0 seconds. Keep XYZ fixed; decrease time at 50 Hz. Negative time drives follow-through and recovery. Inference supports time 1.0…−1.5 s.
- Nominal demonstration: `[0.43, -0.38, 0.178, 1.0]`.

`runs/bracket-v4/policy.ts` is a self-contained TorchScript model with batch input `(N,4)` and output `(N,64)`. Its architecture includes a supervised 4–128–128–128–64 tanh prior, a learned 12-direction token correction, and a bounded phase lead. CEM episodic reinforcement learning optimizes 13 head parameters using simulated return rewards. This is not full-network PPO retraining. The frozen SONIC decoder still consumes current joint/IMU history in addition to the 64D token.

```python
import torch
policy = torch.jit.load('runs/bracket-v4/policy.ts')
token = policy(torch.tensor([[0.43, -0.38, 0.178, 1.0]], dtype=torch.float64))
assert token.shape == (1, 64)
```

The online policy needs no reference-motion encoder. Reference data initializes the ready pose only; after initialization the robot moves through SONIC-generated PD torques. The ball is assigned initial state once, at launch, then follows MuJoCo dynamics. No attachment to the racket, target snapping, scripted outgoing velocity, or teleporting during the flight is used.

## Direct wrist bracket

The public [table-tennis-motion-kit](https://github.com/dvij542/table-tennis-motion-kit), commit `c6c4c061f8fcb24714fbe118a31e66c69adeb4bf`, supplies **kinematic motion references and a visual URDF**, not a trained dynamic policy or measured attachment inertia. [HITTER](https://arxiv.org/html/2508.21043v1), Figure 2, describes a printed right-wrist racket connector. It does not establish that this experiment's physical bracket has identical inertia.

The simulation uses the reference wrist-to-blade-center translation `[0.2365,0,0]` m and local +Y face normal. It removes both stock Dex3 finger subtrees, actuators and palm geometry, and replaces the merged wrist/palm inertia with the bare-wrist inertia from the local G1 URDF. There are 29 actuated body joints. There is no left ball holder in this experiment.

The provisional attachment is 0.14 kg blade + 0.025 kg handle + 0.035 kg bracket. Blade/handle/bracket collision shapes are cylinder/capsule/box approximations; geometry and parameter provenance are in `mount.py` and each run's `mount.json`. These are simulation assumptions, not measured hardware properties. The torso and legs remain dynamic under SONIC stabilization; they are not welded in place.

## Start and inspect

Run `./run_app.sh` and open <http://127.0.0.1:52181/> for the browser-independent simulation UI. Enter Y/Z/time and click **Run strike**: it runs a fresh physical episode and renders the result automatically. This path works without WebGL. The server binds only to localhost, accepts one simulation job at a time, and saves new episodes under the selected run's `interactive/` folder.

Run `./run.sh` from this folder, then open <http://127.0.0.1:52180/>. Click **Run strike** after changing Y, Z or time. **Front camera** restores the front viewpoint; playback does not reset a manually selected camera.

After reopening the updated G1 Studio, it also exposes **Token Policy Sim → Hitting plane · Direct bracket → Start Token Simulation**. **Open Simulation Replay** starts the simulation/video UI on a separate local port. The stop controls in this simulation affect only simulation playback; no hardware is connected by these scripts.

The result page is `runs/bracket-v4/index.html`. It includes a training-case replay, six consecutive fresh test cases including failures, and links to the model and full audit. Videos are rendered from saved dynamic states, at half speed. Their camera is a front view across the table.

## Reproduce

Use the existing SONIC `.venv_sim/bin/python` identified in `run.sh`.

1. `train_model.py --output NEW_PRIOR_RUN` generates forward-facing motion-token labels and trains the supervised prior. Existing source clips are immutable. The historical `forehand/runs/forward-v1` prior is reused as initialization here; its labels are kinematic and did not train attachment dynamics.
2. `train_residual.py --output NEW_RL_RUN --resume runs/bracket-v2/policy.json --robust --workers 6` searches token-head parameters. The robust curriculum includes 1 ms and 0.5 ms physics, plus changed target/time combinations. Avoid excessive worker counts on this 32 GB workstation.
3. `export_policy.py NEW_RL_RUN` packages and checks the pure four-input model against the Python policy on 200 inputs.
4. `evaluate.py NEW_RL_RUN` checks the exported model, disables encoder construction, records actual contacts and writes `evaluation.json` plus per-trial states. The 20 fresh test combinations use seed 81114; they must not become training or model-selection data if still reported as held out.
5. `render.py NEW_RL_RUN` and `render.py NEW_RL_RUN --group fresh_test --count 6` produce H.264 MP4s. `publish_report.py NEW_RL_RUN` writes the local report and desktop preview manifest.

Run `test_pipeline.py` for attachment, name-based seed mapping, source FK, input domain, precise launch timing and persistent table-contact regression checks. The shared decoder environment retains backward-compatible initialization for the original 43-DOF model.

## Interpretation

A legal return requires one own-side bounce, a blade-face collision (not its rim), a forward crossing above the net with ball-radius clearance, and a first valid landing on the opponent's half. Posture failures, racket obstacles, net contacts and invalid earlier ball contacts invalidate success. Contact inspection happens at every physics substep.

Read `evaluation.json` for actual success counts; motion appearance and token prediction error alone are not success metrics. The nearby/wide groups were development checks; the separate fresh test is unseen target/time combinations. Sensitivity probes vary attachment mass and timestep, but do not constitute comprehensive domain randomization.

This is a ready-pose, single-ball, narrow-forehand simulation experiment with ideal target/time inputs. It does not include camera latency, spin, aerodynamic drag, variable incoming velocity families, full motion preparation from arbitrary postures, rallies or hardware validation. New bracket dimensions, mass and balance must be measured before any transfer to the robot.

Exploratory `bracket-v1` was interrupted after a repeated-contact bookkeeping bug was discovered; its old success flags are not final evidence. `bracket-v3` was interrupted to reduce training memory pressure. Completed exported-model audits, rather than partial training logs, are authoritative.
