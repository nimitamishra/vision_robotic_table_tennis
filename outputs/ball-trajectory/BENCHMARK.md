# Offline trajectory benchmark

`benchmark.py` replays the current single-ball tracker from saved table-frame
measurements. It uses the stereo-source selection and range-noise prior from
`tracker.py`, and the existing collision and virtual-plane code. The benchmark
runs in-process with NumPy and the Python standard library. Plotting also
requires Matplotlib.

Run from the repository root.

```sh
python outputs/ball-trajectory/benchmark.py --synthetic \
  --output /tmp/trajectory-synthetic.json --plot /tmp/trajectory-synthetic.png

python -m unittest discover -s outputs/ball-trajectory
```

For recordings, pass the `measurements.jsonl` from each run. Use separate capture
sessions for calibration and evaluation. Replaying a file under another name
or making overlapping clips from the same recording does not produce an
independent split. The evaluator rejects duplicate tracker inputs and shared
captured samples. Choose the split and settings before inspecting the evaluation
scores. Omit `--calibrate` to report unscaled uncertainty only.

```sh
python outputs/ball-trajectory/benchmark.py \
  --calibrate cal-run-1/measurements.jsonl \
  --calibrate cal-run-2/measurements.jsonl \
  --eval test-run-1/measurements.jsonl \
  --eval test-run-2/measurements.jsonl \
  --scene test-run-1/scene-calibration.json \
  --output /tmp/trajectory-recorded.json
```

Use runs that share a table frame, scene geometry and protocol. The checked-in
scene is the default. Inspect the saved run configuration and set the tracker
options explicitly when they differ from the defaults. Replay uses one fixed
configuration and a 500 ms model horizon, so it evaluates that configuration
even if the original live run changed settings. Input timestamps must be
strictly increasing integer nanoseconds. A packet with a timestamp and no
valid 3D position stays in the replay so the filter can coast or reset.

## Fixed horizons

- At each capture, the tracker sees only measurements up to that frame. The
  benchmark evaluates its published 100, 200 and 300 ms predictions. These
  are the tracker's pre-contact ballistic forecasts, and a missing forecast
  stays missing. Coasting forecasts are included.
- Real references use the later raw stereo point at the target timestamp, or
  linear interpolation between two later observations. Both endpoints must
  be strictly later than issuance. Interpolation requires a gap of at most
  40 ms by default and secant speed at most 25 m/s. Synthetic references use
  independently generated noiseless trajectories. Exact timestamp matches
  need no interpolation.
- JSON includes frame, forecast, reference and scored counts, plus distance
  RMSE, mean and 90th-percentile distance, and signed XYZ bias. The persistence
  comparison keeps the last point at the forecast origin and compares both
  predictors on the same scored rows with a valid origin measurement.
- The tracker reports per-axis standard deviations. The benchmark reports
  signed normalized residuals, their RMS, marginal Gaussian negative log
  likelihood and empirical central coverage at 50%, 80%, 90% and 95% for each
  axis. It makes no joint-coverage estimate.
- Each horizon gets one fitted multiplier on variance. It is the mean squared
  standardized residual, averaging axes and rows within a calibration run
  before averaging runs. At least two calibration runs with usable residuals
  at that horizon are required. Both the original and scaled forecasts are
  scored on the held-out runs. The scale leaves means unchanged and does not
  correct axis-specific bias. Pooled test metrics weight scored rows equally,
  and the report also gives metrics per recording. Overlapping time windows
  within a recording are correlated.

## Virtual plane

Pass `--plane-y` in table-frame metres when each file represents one
single-ball attempt. The evaluator uses the finite-table collision forecast,
including its modeled rebound. The reference is the first incoming crossing
of the plane interpolated from the reference positions under the existing
trial gap, speed and bounds rules. For each requested lead, the evaluator
picks the latest forecast issued between that lead and 50 ms earlier.
It reports attempt and crossing counts, position error, and signed and
absolute timing error. Plane predictions have no published uncertainty
to score. Lead times use camera frame receipt, which can precede delivery
to the predictor.

For stereo recordings, position and crossing errors include uncertainty in the
later camera observations and any interpolation error. Shared camera bias can
make a replay look more accurate than it physically is. Coverage and scaling
describe agreement with that reference. Use independent reference measurements
for claims about physical accuracy.

## Synthetic verification

The default seed generates six calibration and six held-out throws, balanced
between free flight and a single bounce. The input has position noise and
missing observations. The reference is generated separately and remains
available when a tracker input is missing. These trials verify the pipeline
and illustrate the metrics under the generator's assumptions.
