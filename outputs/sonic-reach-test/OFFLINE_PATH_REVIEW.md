# Racket-mounted G1: offline first-motion review

This review used the saved read-only joint state, the user's 12 cm
wrist-cylinder-to-grip measurement and 15 cm racket-face width. The user
confirmed that the mounted racket stays stable and does not touch the right
leg. From the robot's front, the racket appears edge-on, so the path sweep
assumed the face normal is roughly lateral to the robot, with ±20° variation.
The unmeasured grip-to-face distance was swept from 15 to 20 cm.

The MuJoCo wrist frame's local +X axis points almost vertically down in this
saved posture, consistent with the photos. Both installed-racket length vectors
were **assumed** to follow that axis. This is not a calibrated mount transform.

| Hypothetical wrist target | Position error | Largest right-arm joint change | Modeled wrist/hip overlap at start → end | Lowest face-to-robot gap over tested lengths/angles |
| --- | ---: | ---: | ---: | ---: |
| 2 cm outward, away from right leg | 0.33 cm | 3.0° | 2.36 → 0 mm | 1.30 cm |
| 2 cm up | 0.16 cm | 34.9° | 2.36 → 3.63 mm | 1.25 cm |
| 2 cm outward and 2 cm up | 0.16 cm | 31.2° | 2.36 → 0 mm | 1.30 cm |

The upright lift is close to a kinematic singularity from this posture: the
small vertical wrist target calls for much larger joint changes than an
outward move. This is a result of one offline solver and one saved pose, not a
claim about all possible G1 poses or controllers.

The model already reports 2.36 mm wrist/hip overlap at the saved start pose,
while the user sees no physical contact. It also reports 0.83 cm overlap for
the crude cylindrical gripper envelope. Those mismatches show why these
numbers cannot certify real clearance. The silver bracket, table pose, added
tool mass and SONIC dynamics are absent. `racket_path_sweep.json` contains all
27 geometry/path scenarios; every result is marked `safe_for_physical_motion:
false`. No physical arm command was sent.

When the lab is available again:

1. Reconnect the G1 Ethernet cable and capture a fresh read-only arm pose and
   table reference. The wired interface was `DOWN` after the lab closed.
2. Check the grip remains stable and finish measuring blue-tape-center to
   racket-face-center distance. Calibrate the face normal and side bracket in
   the wrist frame.
3. Replace the broad gripper capsule with measured geometry. Re-run the
   table/robot collision audit on the fresh pose and a short outward route.
4. Only after the model and physical clearances agree should a supervised,
   very small, slow outward test be considered. Do not start with an upward
   lift or a ball strike.
