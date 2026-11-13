# Installed racket: photo and read-only pose review

The user confirmed that the racket remains stable without hand support and
does not physically touch the right leg. The two photos show a face-on and an
edge-on view, with the handle centered between the blue fingers and the racket
pointing approximately down the gripper axis. The hand visible in the photos
was supporting the face for the photographs; the stability statement comes
from the user's later direct check.

The attached `robot_joint_snapshot.json` is a read-only `rt/lowstate` sample.
The user subsequently measured 12 cm from the right-wrist black cylinder
center to the middle of the blue tape. This is a scalar distance; the exact
vector in the wrist frame is still uncalibrated.
In the MuJoCo G1 model, the current right-wrist local +X axis points nearly
straight down (world vector approximately [-0.08, -0.12, -0.99]). This is
consistent with the photographs. No arm or gripper command was sent for this
review.

A **hypothetical**, not calibrated mount was initially tested in the current joint pose:
10 cm from wrist to grip along local +X, 15 cm more to the face center, 15 cm
face diameter, 1 cm face thickness, and zero uncertainty margin. This is a
geometry sensitivity check, not a measured clearance certificate.

| Geometry assumption | Nearest face/right-leg distance |
| --- | ---: |
| Face represented by an orientation-free sphere | -1.45 cm (false-positive overlap is plausible) |
| Thin disc with normal along wrist local Y | +3.67 cm |
| Thin disc with normal along wrist local Z | -0.21 cm |

The gripper body was approximated by one 3.5 cm-radius capsule from wrist to
grip. It overlapped the modeled right hip by 0.83 cm in all three cases. The
physical observation says the assembly is clear, so this capsule is too crude
to validate that part. The side-mounted silver bracket is also not modeled.

The conclusion is limited: the racket's **orientation and gripper shape change
the collision answer**. A measured wrist-to-grip transform, face normal,
gripper/bracket envelope, and fresh table-to-robot pose are required before
auditing any arm path. The old 2 cm reach replay and old table pose cannot be
used as a current physical-motion clearance result.

## Updated length sweep after the 12 cm measurement

`photo_static_sweep.json` repeats a no-motion static check using the user's
12 cm wrist-to-grip scalar distance. It still assumes both length vectors
point along wrist local +X and tries 15, 17.5, and 20 cm grip-to-face lengths.
No uncertainty margin is included, because the purpose is to reveal how the
shape/orientation assumptions change the result rather than claim safety.

| Grip-to-face length | Face as sphere | Thin face normal: wrist Y | Thin face normal: wrist Z |
| --- | ---: | ---: | ---: |
| 15 cm | -1.17 cm | +3.38 cm | +0.04 cm |
| 17.5 cm | -0.81 cm | +2.84 cm | +0.36 cm |
| 20 cm | -1.17 cm | +2.47 cm | +0.67 cm |

Values are signed distance from the **face** to the nearest robot body. The
handle's modeled gap is 3.32 cm in all cases. The crude gripper capsule still
overlaps the modeled right hip by 0.83 cm. These values do not include the
silver side bracket or the table and cannot authorize robot motion.
