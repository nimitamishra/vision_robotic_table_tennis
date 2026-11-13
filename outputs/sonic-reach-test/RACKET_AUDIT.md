# Installed racket clearance check

`racket_clearance_audit.py` adds a zero-mass collision envelope for the right
gripper, handle, and racket face to the existing G1/table MuJoCo scene. It
checks the saved planned and replayed right-arm path against the table, net,
robot, and floor. It does not send robot commands. The old Dex3 hand in the
simulation is excluded from attachment-versus-robot distance checks because
the actual robot has a YAM two-finger gripper.

Without a measured face normal, the face is modeled as a sphere around its
center. This can falsely report a collision with the robot's right leg because
the real racket is thin. Once the face normal is calibrated in the wrist frame,
set `face_shape` to `disc` and fill `face_normal_wrist`; the model then uses a
thin cylinder. This is still **not** a dynamic collision or ball-impact test.

`racket_mount_template.json` contains the user's measured 15 cm face width,
2.4 cm handle thickness, and rough 15–20 cm grip-to-face distance. The missing
3D vectors and gripper dimensions are `null`, and the audit refuses to run
until they are measured on the installed hardware. The vectors use the G1
`right_wrist_yaw_link` frame in metres:

- `wrist_to_grip_center_m`: wrist-yaw frame origin to the middle of the handle
  section touched by the two blue fingers.
- `grip_to_face_center_m`: that grip center to the center of the racket face.
- `face_normal_wrist`: outward normal of the rubber surface, expressed in the
  same wrist frame. Only required for the thinner `disc` geometry.

To measure the second distance by hand, start at the center of the section
between the blue fingers, follow the handle, and stop at the middle of the
round face. A front and side image with a ruler can establish an initial
direction estimate; a calibrated mount transform is needed for a meaningful
1 cm clearance claim. The gripper envelope radius and face thickness must
also be filled in. Keep `measured_on_installed_racket` false until then.

After a new robot/table pose snapshot and a saved simulation run exist:

```bash
/home/jingxuan/Documents/Codex/2026-09-15/nih/work/GR00T-WholeBodyControl/.venv_sim/bin/python \
  /home/jingxuan/Documents/Codex/2026-10-16/ni/outputs/sonic-reach-test/racket_clearance_audit.py \
  /path/to/saved-simulation-run \
  --mount /path/to/measured-racket-mount.json \
  --snapshot /path/to/new-pose-snapshot.json
```

The output is `racket_clearance_audit.json` inside the run. A pose snapshot
older than 30 seconds cannot yield `preliminary_clear_at_1cm: true`. Even a
clear offline report does not validate physical arm control, balance under
the added mass, racket orientation, or contact with a moving ball.
