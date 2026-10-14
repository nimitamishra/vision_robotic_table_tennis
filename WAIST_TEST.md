# G1 waist rotation localization check

Purpose: verify that the **pelvis/waist rotation axis in the table frame** stays
approximately fixed while the upper body and head camera rotate. A single
41 cm tape correction at one pose does not validate this dynamic transform.

## Setup

1. Keep the table and reference tags 0, 1, 2, and 5 fixed. Prefer tag 5 visible;
   record which tag each pose actually uses.
2. Keep both feet planted and the pelvis supported. First test with stable
   support, then repeat under SONIC standing control. Standing can move the
   pelvis slightly; a moving pelvis invalidates the stationary-reference test.
3. Read **measured** `rt/lowstate` motor positions, not desired commands.
   On the G1 29-DoF mapping, waist yaw is motor 12. Motors 13/14 may be
   unavailable or locked on this variant; verify before using them.
4. Synchronize camera frames and joint samples using timestamps from the same
   robot clock. Record at least 2 seconds per stationary pose, at the center
   and at small left/right waist angles chosen by the robot operator. Do not
   infer angles from the command alone.

## Per-sample data

Record the image/frame ID and capture time, detected tag IDs and quality,
`T_world_camera`, measured waist yaw (and roll/pitch if active), and the
computed `T_world_pelvis`. Also note whether feet/pelvis moved physically.
Discard frames with no reliable reference tag or stale joint telemetry.

## Computation

Use the current URDF and the measured joint configuration for each sample:

```
T_world_pelvis = T_world_camera @ inverse(T_pelvis_camera(q_measured))
```

Keep the separately measured camera-mount correction explicit. For a robot
known to be upright, compare the horizontal pelvis position and yaw; do not
force the pelvis yaw to equal the camera yaw. At nonzero waist yaw, the camera
heading changes even when the pelvis heading does not.

## Pass/fail comparison

For each held pose, report the median pelvis `(x, y, yaw)` and the spread of
samples. Compare the left/right pose medians with the center baseline. The
corrected pelvis should remain within the measurement noise and actual
physical sway, while camera pose traces an arc. As an initial diagnostic,
investigate horizontal drift above roughly 2 cm or yaw drift above 2 degrees
under stable support; confirm these thresholds against measured tag and joint
noise rather than treating them as guaranteed accuracy.

If the corrected pelvis drifts: check the timestamp pairing, real joint index
and sign, URDF model/version, optical-to-link transform, and whether the camera
mount or pelvis actually moved. Re-measure camera extrinsics if needed. The
updated Viser uses measured waist yaw when the read-only relay is available.
A real rotation test is still required before trusting the result.

## Current readiness

The local G1 Viser code supports measured waist yaw and camera-frame
timestamps. A synthetic forward-kinematics sweep at -30, -15, 0, +15, and
+30 degrees passes. On 2026-10-09 the updated stereo server and read-only
joint relay were deployed and tested on the G1. One left-turn segment changed
the estimated pelvis yaw by about 0.4 degrees while the old fixed transform
would have changed it by about 23 degrees. The right-turn segment was not
conclusive because tag 5 left the camera view. A more controlled test with an
independent pelvis reference is still needed for absolute accuracy.
