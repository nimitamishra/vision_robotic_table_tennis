# Live viser view of the G1 head camera and robot in the world frame defined by calibrate_world.py.
#
#   python g1/viz_viser.py      then open http://localhost:8080
#
# The robot is drawn in its nominal (zero-joint) pose, rigidly placed so its URDF d435_link matches
# the camera pose measured from the reference tag. Waist/leg joint angles are not read yet.
import argparse, json, time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import viser
import yourdfpy
from pupil_apriltags import Detector
from viser.extras import ViserUrdf

from g1_stream import (G1_HOST, G1CameraStream, T_D435LINK_OPTICAL, T_to_wxyz_pos, T_world_cam_from_tag,
                       make_T)

HERE = Path(__file__).parent
URDF_PATH = HERE / "g1_description" / "g1_29dof_rev_1_0.urdf"

ap = argparse.ArgumentParser()
ap.add_argument("--host", default=G1_HOST)
ap.add_argument("--calib", default=str(HERE / "calibration.json"))
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--smooth", type=float, default=0.3, help="EMA weight on new pose (1 = no smoothing)")
args = ap.parse_args()

calib = json.loads(Path(args.calib).read_text())
world = calib["world"]
TAG_ID, TAG_SIZE = world["tag_id"], world["tag_size_m"]
T_WORLD_TAG = np.array(world["T_world_tag"])

detector = Detector(families=world["tag_family"])
stream = G1CameraStream(args.host)
W, H = stream.intr["width"], stream.intr["height"]
fov_y = 2 * np.arctan2(H / 2, stream.intr["fy"])

urdf = yourdfpy.URDF.load(str(URDF_PATH))
T_PELVIS_OPTICAL = urdf.get_transform("d435_link", "pelvis") @ T_D435LINK_OPTICAL
T_OPTICAL_PELVIS = np.linalg.inv(T_PELVIS_OPTICAL)

# ---- scene ----
server = viser.ViserServer(port=args.port)
server.scene.set_up_direction("+z")
server.scene.add_grid("/grid", width=4, height=4, cell_size=0.1, section_size=0.5)
server.scene.add_frame("/world", axes_length=0.15, axes_radius=0.004)
server.scene.add_box("/world/ref_tag", dimensions=(TAG_SIZE, TAG_SIZE, 0.002), color=(20, 20, 20),
                     position=(0, 0, -0.001))
server.scene.add_label("/world/ref_tag/label", f"tag {TAG_ID} (world)", position=(0, 0, 0.02))

cam_frame = server.scene.add_frame("/camera", axes_length=0.08, axes_radius=0.003)
frustum = server.scene.add_camera_frustum("/camera/frustum", fov=fov_y, aspect=W / H, scale=0.15,
                                          color=(255, 120, 0))
robot_root = server.scene.add_frame("/robot", show_axes=True, axes_length=0.12, axes_radius=0.004)
robot = ViserUrdf(server, urdf, root_node_name="/robot")
robot.update_cfg(np.zeros(len(robot.get_actuated_joint_limits())))
trail_handle = None
other_tags = {}

with server.gui.add_folder("G1 camera"):
    gui_status = server.gui.add_markdown("waiting for frames...")
    gui_pos = server.gui.add_text("cam xyz (m)", initial_value="-", disabled=True)
    gui_pelvis = server.gui.add_text("pelvis xyz (m)", initial_value="-", disabled=True)
    gui_img = server.gui.add_checkbox("show image on frustum", True)
    gui_robot = server.gui.add_checkbox("show robot", True)
    gui_trail = server.gui.add_checkbox("show trail", True)
    gui_clear = server.gui.add_button("clear trail")

trail = deque(maxlen=600)
gui_clear.on_click(lambda _: trail.clear())
gui_robot.on_update(lambda _: setattr(robot_root, "visible", gui_robot.value))

print(f"viser at http://localhost:{args.port}  (world = tag {TAG_ID}, {TAG_SIZE * 100:.1f} cm)")

T_smooth = None
last_seen = 0.0
n, t_fps = 0, time.time()
while True:
    color, _ = stream.read()
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray, estimate_tag_pose=True, camera_params=stream.camera_params, tag_size=TAG_SIZE)
    ref = next((t for t in tags if t.tag_id == TAG_ID), None)

    if ref is not None:
        T = T_world_cam_from_tag(ref, T_WORLD_TAG)
        if T_smooth is None or args.smooth >= 1:
            T_smooth = T
        else:  # blend translation linearly, rotation via quaternion lerp
            from scipy.spatial.transform import Rotation, Slerp
            R = Slerp([0, 1], Rotation.from_matrix([T_smooth[:3, :3], T[:3, :3]]))(args.smooth).as_matrix()
            T_smooth = make_T(R, (1 - args.smooth) * T_smooth[:3, 3] + args.smooth * T[:3, 3])
        last_seen = time.time()

        wxyz, pos = T_to_wxyz_pos(T_smooth)
        cam_frame.wxyz, cam_frame.position = wxyz, pos
        T_world_pelvis = T_smooth @ T_OPTICAL_PELVIS
        robot_root.wxyz, robot_root.position = T_to_wxyz_pos(T_world_pelvis)
        gui_pos.value = " ".join(f"{v:+.3f}" for v in pos)
        gui_pelvis.value = " ".join(f"{v:+.3f}" for v in T_world_pelvis[:3, 3])

        trail.append(pos)
        if gui_trail.value and len(trail) > 1:
            pts = np.array(trail)
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            trail_handle = server.scene.add_line_segments("/trail", segs, colors=(255, 120, 0), line_width=2)
        elif trail_handle is not None:
            trail_handle.remove()
            trail_handle = None

        # other tags, placed in world through the camera
        for t in tags:
            if t.tag_id == TAG_ID:
                continue
            T_w_t = T_smooth @ make_T(t.pose_R, t.pose_t) @ np.diag([1.0, -1, -1, 1])
            q, p = T_to_wxyz_pos(T_w_t)
            if t.tag_id not in other_tags:
                other_tags[t.tag_id] = server.scene.add_frame(f"/tags/{t.tag_id}", axes_length=0.06,
                                                             axes_radius=0.003)
                server.scene.add_label(f"/tags/{t.tag_id}/label", f"tag {t.tag_id}")
            other_tags[t.tag_id].wxyz, other_tags[t.tag_id].position = q, p

    # draw detections on the image shown in the frustum
    for t in tags:
        c = t.corners.astype(int)
        cv2.polylines(color, [c], True, (0, 255, 0) if t.tag_id == TAG_ID else (255, 200, 0), 2)
        cv2.putText(color, f"ID {t.tag_id}", tuple(t.center.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
    n += 1
    if n % 3 == 0:  # ~10 Hz image updates is plenty
        frustum.image = cv2.cvtColor(cv2.resize(color, (W // 2, H // 2)), cv2.COLOR_BGR2RGB) if gui_img.value else None
    if n % 30 == 0:
        fps = 30 / (time.time() - t_fps)
        t_fps = time.time()
        age = time.time() - last_seen
        state = "**tracking**" if age < 0.5 else f"ref tag lost ({age:.1f}s)"
        gui_status.content = f"{state} · {fps:.0f} fps · tags: {sorted(t.tag_id for t in tags)}"
