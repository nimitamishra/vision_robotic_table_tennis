"""Live G1 stereo infrared AprilTag localization in a Viser scene."""

import argparse
import json
import time
from collections import defaultdict, deque
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import cv2
import msgpack
import numpy as np
import viser
import yourdfpy
import zmq
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation, Slerp
from viser.extras import ViserUrdf

from g1_stream import T_D435LINK_OPTICAL, T_to_wxyz_pos, average_poses, make_T
from joint_stream import JointStream
from stereo_stream import StereoStream
from table_scene import add_table


HERE = Path(__file__).parent
URDF_PATH = HERE / "g1_description" / "g1_29dof_rev_1_0.urdf"
NECK_PIVOT_TORSO = np.array([0.0, 0.0, 0.30])
TAG_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
# SONIC's right VR tracking point, expressed in the wrist-yaw link frame.
RIGHT_WRIST_TRACKING_OFFSET = np.array([0.18, 0.025, 0.0, 1.0])


def select_ball_measurement(result, prefer_contour):
    """Select a complete same-frame stereo estimate; never mix methods across eyes."""
    choices = [("ball_refined", "xyz_world_refined", "Contour center")] if prefer_contour else []
    choices.append(("ball", "xyz_world", "YOLO box fallback (lower precision)" if prefer_contour else "YOLO box center"))
    for pose_key, xyz_key, label in choices:
        pose = result.get(pose_key)
        xyz = result.get(xyz_key)
        if pose is None:
            continue
        if result.get("world_measurement"):
            if xyz is None or np.asarray(xyz).shape != (3,) or not np.all(np.isfinite(xyz)):
                continue
        return pose, xyz, label
    return None, None, "Lost: no valid stereo center"


def camera_yaw_pitch(T_world_opt):
    forward = T_world_opt[:3, 2]
    return np.arctan2(forward[1], forward[0]), np.arcsin(np.clip(-forward[2], -1, 1))


def estimate_upright_pelvis(T_world_camera, T_pelvis_camera, forward_correction_m=0.0):
    """Recover an upright pelvis using the *measured* joint-dependent camera FK."""
    camera_yaw = camera_yaw_pitch(T_world_camera)[0]
    relative_yaw = camera_yaw_pitch(T_pelvis_camera)[0]
    R_world_pelvis = Rotation.from_euler("z", camera_yaw - relative_yaw).as_matrix()
    position = (T_world_camera[:3, 3] - R_world_pelvis @ T_pelvis_camera[:3, 3]
                + R_world_pelvis @ np.array([forward_correction_m, 0.0, 0.0]))
    return make_T(R_world_pelvis, position)


def load_g1_with_head_tilt(tilt_rad):
    urdf = yourdfpy.URDF.load(str(URDF_PATH))
    pivot = make_T(np.eye(3), NECK_PIVOT_TORSO)
    tilt = pivot @ make_T(Rotation.from_euler("y", tilt_rad).as_matrix(), 0) @ np.linalg.inv(pivot)
    for name in ("head_joint", "d435_joint"):
        urdf.joint_map[name].origin = tilt @ urdf.joint_map[name].origin
    handler = partial(yourdfpy.filename_handler_magic, dir=str(URDF_PATH.parent))
    return yourdfpy.URDF(robot=urdf.robot, filename_handler=handler)


def detect_ir(detector, gray, intr, size, scale, top_fraction=1.0):
    region = gray[:int(gray.shape[0] * top_fraction)]
    enlarged = (region if scale == 1 else
                cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
    params = tuple(intr[key] * scale for key in ("fx", "fy", "cx", "cy"))
    detections = detector.detect(enlarged, estimate_tag_pose=True, camera_params=params, tag_size=size)
    return [SimpleNamespace(tag_id=tag.tag_id, pose_R=tag.pose_R, pose_t=tag.pose_t,
                            corners=tag.corners / scale, center=tag.center / scale)
            for tag in detections if tag.pose_R is not None and tag.pose_t is not None
            and np.all(np.isfinite(tag.pose_R)) and np.all(np.isfinite(tag.pose_t))
            and np.linalg.det(tag.pose_R) > 0.9
            and np.allclose(tag.pose_R.T @ tag.pose_R, np.eye(3), atol=0.01)]


def detect_camera_tags(detector, gray, intr, size, scale):
    # Distant tags occupy only a few pixels. Always search the far half at higher scale,
    # even when the large reference tag at the near edge is already visible.
    near = detect_ir(detector, gray, intr, size, scale)
    far = detect_ir(detector, gray, intr, size, max(3, scale), top_fraction=0.68)
    by_id = {tag.tag_id: tag for tag in near}
    by_id.update({tag.tag_id: tag for tag in far})
    return list(by_id.values())


def annotate(gray, tags):
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for tag in tags:
        corners = np.rint(tag.corners).astype(np.int32)
        cv2.polylines(image, [corners], True, (255, 255, 255), 2)
        center = tuple(np.rint(tag.center).astype(int))
        cv2.putText(image, f"ID {tag.tag_id}", center, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def smooth_pose(previous, current, weight):
    if previous is None or weight >= 1:
        return current
    rotations = Rotation.from_matrix([previous[:3, :3], current[:3, :3]])
    rotation = Slerp([0, 1], rotations)(weight).as_matrix()
    position = (1 - weight) * previous[:3, 3] + weight * current[:3, 3]
    return make_T(rotation, position)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.123.164")
    parser.add_argument("--camera-host", default=None)
    parser.add_argument("--stream-port", type=int, default=5560)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--calib", default=str(HERE / "calibration.json"))
    parser.add_argument("--head-tilt", type=float)
    parser.add_argument("--smooth", type=float, default=0.3)
    parser.add_argument("--detection-scale", type=int, default=1)
    parser.add_argument("--ball-port", type=int, default=5561)
    parser.add_argument("--joint-port", type=int, default=5562)
    parser.add_argument("--pose-port", type=int, default=5565,
                        help="localhost-only read-only robot pose stream")
    args = parser.parse_args()

    calibration = json.loads(Path(args.calib).read_text())
    world = calibration["world"]
    tag_id = world["tag_id"]
    tag_size = world["tag_size_m"]
    table_spec = world["table"]
    # A tape measurement from the waist-axis side plane to the near table edge
    # calibrates the camera-to-pelvis forward offset for this physical G1.
    waist_forward_correction = float(calibration.get("robot", {}).get(
        "waist_forward_correction_m", 0.0))
    T_world_tag = np.asarray(world["T_world_tag"], dtype=float)
    reference_poses = {tag_id: T_world_tag}
    if world.get("fallback_tags_valid", True):
        reference_poses.update({int(key): np.asarray(value, dtype=float)
                                for key, value in world.get("fallback_tags", {}).items()})
    play_min = np.asarray(world.get("play_volume_min_m", [-np.inf] * 3), dtype=float)
    play_max = np.asarray(world.get("play_volume_max_m", [np.inf] * 3), dtype=float)
    stream = StereoStream(args.camera_host or args.host, args.stream_port)
    joints = JointStream(args.host, args.joint_port)
    ball_context = zmq.Context()
    ball_socket = ball_context.socket(zmq.SUB)
    ball_socket.setsockopt(zmq.SUBSCRIBE, b"")
    ball_socket.setsockopt(zmq.CONFLATE, 1)
    ball_socket.setsockopt(zmq.LINGER, 0)
    ball_socket.connect(f"tcp://127.0.0.1:{args.ball_port}")
    pose_socket = ball_context.socket(zmq.PUB)
    pose_socket.setsockopt(zmq.SNDHWM, 1)
    pose_socket.setsockopt(zmq.LINGER, 0)
    pose_socket.bind(f"tcp://127.0.0.1:{args.pose_port}")
    left_intr, right_intr = stream.left_intr, stream.right_intr
    T_left_right, T_left_color = stream.T_left_right, stream.T_left_color
    detector = Detector(families=world["tag_family"], quad_decimate=1.0, refine_edges=1)

    stock_pose = (yourdfpy.URDF.load(str(URDF_PATH), load_meshes=False)
                  .get_transform("d435_link", "pelvis") @ T_D435LINK_OPTICAL)
    stock_pitch = camera_yaw_pitch(stock_pose)[1]
    measured_pitch = camera_yaw_pitch(np.asarray(calibration["calibration_pose"]["T_world_camera_optical"]))[1]
    tilt = np.radians(args.head_tilt) if args.head_tilt is not None else measured_pitch - stock_pitch
    urdf = load_g1_with_head_tilt(tilt)
    T_pelvis_color = urdf.get_transform("d435_link", "pelvis") @ T_D435LINK_OPTICAL

    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/floor", width=5, height=6, cell_size=0.1, section_size=0.5,
                          position=(0, table_spec["length_m"] / 2 - table_spec["tag_center_inset_m"],
                                    -table_spec["height_m"]))
    server.scene.add_frame("/world", axes_length=0.15, axes_radius=0.004)
    table_root, near_edge_y = add_table(server.scene, table_spec)
    server.scene.add_box("/world/ref_tag", dimensions=(tag_size, tag_size, 0.002),
                         color=(245, 245, 245), position=(0, 0, 0.003))
    server.scene.add_label("/world/ref_tag/label", f"tag {tag_id} (table origin)",
                           position=(0, 0, 0.04))

    cameras = {}
    for name, intr in (("left", left_intr), ("right", right_intr)):
        frame = server.scene.add_frame(f"/ir_{name}", axes_length=0.07, axes_radius=0.003)
        fov = 2 * np.arctan2(intr["height"] / 2, intr["fy"])
        frustum = server.scene.add_camera_frustum(
            f"/ir_{name}/frustum", fov=fov, aspect=intr["width"] / intr["height"],
            scale=0.18, color=(110, 110, 110), format="jpeg", jpeg_quality=70)
        server.scene.add_label(f"/ir_{name}/label", f"IR {name}", position=(0, -0.07, 0))
        cameras[name] = (frame, frustum)

    robot_root = server.scene.add_frame("/robot", show_axes=True, axes_length=0.12, axes_radius=0.004)
    robot = ViserUrdf(server, urdf, root_node_name="/robot")
    robot_cfg = np.zeros(len(robot.get_actuated_joint_limits()))
    cfg_names = robot.get_actuated_joint_names()
    waist_cfg_index = cfg_names.index("waist_yaw_joint")
    right_arm_cfg_indices = [cfg_names.index(name) for name in RIGHT_ARM_JOINTS]
    robot.update_cfg(robot_cfg)
    server.scene.add_frame("/measurements", show_axes=False)
    robot_floor_marker = server.scene.add_icosphere("/measurements/robot_floor", radius=0.03,
                                                     color=(245, 135, 40), visible=False)
    robot_floor_distance = None
    ball_sphere = server.scene.add_icosphere("/ball", radius=0.02, color=(245, 180, 30),
                                             visible=False)
    server.scene.add_label("/ball/label", "Ping-pong ball", position=(0, 0, 0.035))
    other_tags = {}
    tag_history = defaultdict(lambda: deque(maxlen=10))
    confirmed_tags = set()
    last_world_tags = {}
    last_tag_seen = {}
    trail = deque(maxlen=600)
    trail_handle = None

    with server.gui.add_folder("G1 Stereo IR Localization"):
        gui_status = server.gui.add_markdown("Waiting for stereo IR frames and a reference tag…")
        gui_left = server.gui.add_text("Left camera tag IDs", initial_value="-", disabled=True)
        gui_right = server.gui.add_text("Right camera tag IDs", initial_value="-", disabled=True)
        gui_positions = server.gui.add_markdown("Tag positions: waiting for detection")
        gui_ball = server.gui.add_markdown("Ping-pong ball: waiting for GPU model and stereo frames…")
        gui_ball_method = server.gui.add_dropdown("Ball center method", options=("Auto: contour + YOLO fallback", "Box center", "Contour center (experimental)"), initial_value="Auto: contour + YOLO fallback")
        gui_ball_comparison = server.gui.add_markdown("Contour comparison: waiting for frames")
        gui_pos = server.gui.add_text("Left camera xyz (m)", initial_value="-", disabled=True)
        gui_pelvis = server.gui.add_text("Waist axis xyz (m, tape-calibrated)", initial_value="-", disabled=True)
        gui_waist = server.gui.add_text("Measured waist yaw (°)", initial_value="Waiting for joint state", disabled=True)
        gui_relative = server.gui.add_markdown("Robot relative to table: waiting for localization")
        server.gui.add_markdown(
            f"Waist calibration: {waist_forward_correction * 100:+.1f} cm along robot forward axis; "
            "based on an approximately 41 cm tape measurement to the near table edge.")
        server.gui.add_text("Head pitch (°)", initial_value=f"{np.degrees(tilt):+.1f}", disabled=True)
        gui_img = server.gui.add_checkbox("Show stereo grayscale images", True)
        gui_robot = server.gui.add_checkbox("Show robot", True)
        gui_table = server.gui.add_checkbox("Show table", True)
        gui_trail = server.gui.add_checkbox("Show trajectory", True)
        gui_clear = server.gui.add_button("Clear trajectory")
        image_shape = (left_intr["height"], left_intr["width"], 3)
        gui_left_image = server.gui.add_image(np.zeros(image_shape, dtype=np.uint8),
                                              label="Left IR", format="jpeg", jpeg_quality=90)
        gui_right_image = server.gui.add_image(np.zeros(image_shape, dtype=np.uint8),
                                               label="Right IR", format="jpeg", jpeg_quality=90)
        gui_ball_left_image = server.gui.add_image(np.zeros(image_shape, dtype=np.uint8),
                                                   label="Ball detection · left", format="jpeg", jpeg_quality=80)
        gui_ball_right_image = server.gui.add_image(np.zeros(image_shape, dtype=np.uint8),
                                                    label="Ball detection · right", format="jpeg", jpeg_quality=80)

    gui_clear.on_click(lambda _: trail.clear())
    gui_robot.on_update(lambda _: setattr(robot_root, "visible", gui_robot.value))
    gui_table.on_update(lambda _: setattr(table_root, "visible", gui_table.value))

    @server.on_client_connect
    def set_view(client):
        client.camera.position = (2.1, -1.4, 1.7)
        client.camera.look_at = (0, table_spec["length_m"] / 2, -0.05)

    print(f"Stereo IR Viser ready: 127.0.0.1:{args.port}; world tag {tag_id}", flush=True)
    T_smooth = None
    last_seen = 0.0
    count = 0
    t_fps = time.time()
    ball_result = None
    ball_preview_id = None
    active_reference = None
    pelvis_pos = None
    active_waist_yaw = None
    active_right_arm_q = None
    try:
        while True:
            try:
                left, right, left_id, _ = stream.read()
            except TimeoutError:
                # Keep the HTTP/WebSocket server alive while the stream reconnects.
                gui_status.content = "Camera stream interrupted — waiting for frames to resume"
                gui_ball.content = "Ball position unavailable: camera stream interrupted"
                gui_ball_comparison.content = "No current measurements — waiting for camera"
                gui_relative.content = "Robot position unavailable: camera stream interrupted"
                ball_sphere.visible = False
                robot_floor_marker.visible = False
                if robot_floor_distance is not None:
                    robot_floor_distance.remove()
                    robot_floor_distance = None
                continue
            joint_sample = joints.nearest(stream.last_frame_monotonic_ns)
            waist_yaw = float(joint_sample["waist_yaw_rad"]) if joint_sample is not None else 0.0
            q = (np.asarray(joint_sample.get("right_arm_q_rad"), dtype=float)
                 if joint_sample is not None and joint_sample.get("right_arm_q_rad") is not None
                 else None)
            dq = (np.asarray(joint_sample.get("right_arm_dq_rad_s"), dtype=float)
                  if joint_sample is not None and joint_sample.get("right_arm_dq_rad_s") is not None
                  else None)
            arm_valid = (q is not None and dq is not None and q.shape == (7,)
                         and dq.shape == (7,) and np.all(np.isfinite(q))
                         and np.all(np.isfinite(dq)))
            if active_waist_yaw != waist_yaw or (arm_valid and
                    (active_right_arm_q is None or not np.array_equal(q, active_right_arm_q))):
                robot_cfg[waist_cfg_index] = waist_yaw
                if arm_valid:
                    robot_cfg[right_arm_cfg_indices] = q
                # The Viser robot and the FK calculation share this URDF object.
                robot.update_cfg(robot_cfg)
                active_waist_yaw = waist_yaw
                if arm_valid:
                    active_right_arm_q = q.copy()
            T_pelvis_color = urdf.get_transform("d435_link", "pelvis") @ T_D435LINK_OPTICAL
            try:
                ball_result = msgpack.unpackb(ball_socket.recv(zmq.NOBLOCK), raw=False)
            except zmq.Again:
                pass
            left_tags = detect_camera_tags(detector, left, left_intr, tag_size, args.detection_scale)
            right_tags = detect_camera_tags(detector, right, right_intr, tag_size, args.detection_scale)
            fused = {}
            for tags, T_left_camera in ((left_tags, np.eye(4)), (right_tags, T_left_right)):
                for tag in tags:
                    if tag.pose_R is not None and tag.pose_t is not None:
                        fused.setdefault(tag.tag_id, []).append(T_left_camera @ make_T(tag.pose_R, tag.pose_t))
            fused = {key: average_poses(poses) for key, poses in fused.items()}
            for key in set(tag_history) | (set(fused) - {tag_id}):
                tag_history[key].append(key in fused)
                if sum(tag_history[key]) >= 4:
                    confirmed_tags.add(key)
            visible_reference = next((key for key in reference_poses if key in fused), None)
            if visible_reference is not None:
                T_world_left = reference_poses[visible_reference] @ np.linalg.inv(fused[visible_reference])
                T_smooth = smooth_pose(T_smooth, T_world_left, args.smooth)
                last_seen = time.time()
                active_reference = visible_reference

            pose_fresh = T_smooth is not None and time.time() - last_seen < 0.5
            pose_packet = {
                "capture_monotonic_ns": stream.last_frame_monotonic_ns,
                "frame_id": left_id,
                "pose_fresh": bool(pose_fresh),
                "reference_tag_id": active_reference if pose_fresh else None,
                "reference_seen_both_cameras": (
                    tag_id in {tag.tag_id for tag in left_tags}
                    and tag_id in {tag.tag_id for tag in right_tags}),
                "joint_fresh": joint_sample is not None,
                "right_arm_fresh": bool(arm_valid),
            }
            if T_smooth is not None:
                left_frame, _ = cameras["left"]
                right_frame, _ = cameras["right"]
                left_frame.wxyz, left_frame.position = T_to_wxyz_pos(T_smooth)
                right_frame.wxyz, right_frame.position = T_to_wxyz_pos(T_smooth @ T_left_right)
                gui_waist.value = (f"{np.degrees(waist_yaw):+.1f}" if joint_sample is not None
                                   else "Unavailable; zero-angle estimate")
                if pose_fresh:
                    T_world_color = T_smooth @ T_left_color
                    T_world_pelvis = estimate_upright_pelvis(
                        T_world_color, T_pelvis_color, waist_forward_correction)
                    pelvis_pos = T_world_pelvis[:3, 3]
                    robot_root.wxyz, robot_root.position = T_to_wxyz_pos(T_world_pelvis)
                    gui_pos.value = " ".join(f"{v:+.3f}" for v in T_smooth[:3, 3])
                    gui_pelvis.value = " ".join(f"{v:+.3f}" for v in pelvis_pos)
                    robot_floor_marker.position = (pelvis_pos[0], pelvis_pos[1],
                                                   -table_spec["height_m"] + 0.015)
                    pose_packet["T_world_pelvis"] = T_world_pelvis.tolist()
                    if arm_valid:
                        T_pelvis_wrist = urdf.get_transform("right_wrist_yaw_link", "pelvis")
                        pose_packet["right_hand_pelvis_m"] = (
                            T_pelvis_wrist @ RIGHT_WRIST_TRACKING_OFFSET)[:3].tolist()
                        pose_packet["right_arm_q_rad"] = q.tolist()
                        pose_packet["right_arm_dq_rad_s"] = dq.tolist()
                robot_floor_marker.visible = pose_fresh

                if pose_fresh and count % 5 == 0:
                    if robot_floor_distance is not None:
                        robot_floor_distance.remove()
                    floor_z = -table_spec["height_m"] + 0.018
                    robot_floor_distance = server.scene.add_line_segments(
                        "/measurements/robot_to_near_edge",
                        np.asarray([[[pelvis_pos[0], pelvis_pos[1], floor_z],
                                     [np.clip(pelvis_pos[0], -table_spec["width_m"] / 2,
                                              table_spec["width_m"] / 2), near_edge_y, floor_z]]]),
                        colors=(245, 135, 40), line_width=3)
                elif not pose_fresh and robot_floor_distance is not None:
                    robot_floor_distance.remove()
                    robot_floor_distance = None

                if visible_reference is not None:
                    trail.append(T_smooth[:3, 3].copy())
                if gui_trail.value and len(trail) > 1 and count % 5 == 0:
                    points = np.asarray(trail)
                    segments = np.stack([points[:-1], points[1:]], axis=1)
                    trail_handle = server.scene.add_line_segments("/trail", segments,
                                                                  colors=(100, 100, 100), line_width=2)
                elif not gui_trail.value and trail_handle is not None:
                    trail_handle.remove()
                    trail_handle = None

                for key, T_left_tag in fused.items():
                    if key == tag_id or key not in confirmed_tags:
                        continue
                    T_world_other = T_smooth @ T_left_tag @ TAG_FLIP
                    last_world_tags[key] = T_world_other
                    last_tag_seen[key] = time.time()
                    if key not in other_tags:
                        frame = server.scene.add_frame(f"/tags/{key}", axes_length=0.08,
                                                       axes_radius=0.003)
                        server.scene.add_label(f"/tags/{key}/label", f"tag {key}",
                                               position=(0, 0, 0.03))
                        other_tags[key] = frame
                    other_tags[key].wxyz, other_tags[key].position = T_to_wxyz_pos(T_world_other)
                    other_tags[key].visible = True
                for key, frame in other_tags.items():
                    if time.time() - last_tag_seen.get(key, 0) > 2:
                        frame.visible = False

            # A timestamped, read-only local feed for policy shadow tests.
            # No pose or arm target is published when its source is stale.
            pose_socket.send(msgpack.packb(pose_packet, use_bin_type=True))

            ball_fresh = ball_result is not None and time.monotonic() - ball_result["timestamp"] < 0.4
            refined_mode = gui_ball_method.value != "Box center"
            ball_pose, selected_xyz, center_source = (select_ball_measurement(ball_result, refined_mode)
                                                      if ball_fresh else (None, None, "Lost: stale or missing data"))
            ball_in_play = False
            if ball_pose is not None and (ball_result.get("world_measurement") or
                                          (T_smooth is not None and time.time() - last_seen < 0.5)):
                if ball_result.get("world_measurement"):
                    point = np.asarray(selected_xyz if selected_xyz is not None else [np.nan, np.nan, np.nan])
                else:
                    point = (T_smooth @ np.r_[ball_pose["xyz_left"], 1])[:3]
                ball_in_play = bool(np.all(point >= play_min) and np.all(point <= play_max))
                if ball_in_play:
                    ball_sphere.position = point
            ball_sphere.visible = ball_in_play
            if ball_result is not None and ball_result.get("preview_jpg") and ball_result["frame_id"] != ball_preview_id:
                preview = cv2.imdecode(np.frombuffer(ball_result["preview_jpg"], np.uint8), cv2.IMREAD_COLOR)
                if preview is not None:
                    w = preview.shape[1] // 2
                    gui_ball_left_image.image = cv2.cvtColor(preview[:, :w], cv2.COLOR_BGR2RGB)
                    gui_ball_right_image.image = cv2.cvtColor(preview[:, w:], cv2.COLOR_BGR2RGB)
                    ball_preview_id = ball_result["frame_id"]

            count += 1
            if count % 2 == 0:
                for name, image, tags in (("left", left, left_tags), ("right", right, right_tags)):
                    _, frustum = cameras[name]
                    annotated = annotate(image, tags)
                    frustum.image = annotated if gui_img.value else None
                    (gui_left_image if name == "left" else gui_right_image).image = annotated
            if count % 5 == 0:
                elapsed = time.time() - t_fps
                fps = 5 / elapsed if elapsed else 0
                t_fps = time.time()
                age = time.time() - last_seen
                state = (f"**Reference tag {active_reference} localized**" if age < 0.5 else
                         "**Reference tag not visible; pose update paused**")
                joint_state = ("live waist encoder" if joint_sample is not None else
                               "NO live waist encoder — pelvis pose may drift during SONIC")
                gui_status.content = f"{state} · {fps:.1f} fps · stereo IR · {joint_state} · frame {left_id}"
                gui_left.value = ", ".join(map(str, sorted(t.tag_id for t in left_tags))) or "None detected"
                gui_right.value = ", ".join(map(str, sorted(t.tag_id for t in right_tags))) or "None detected"
                lines = [f"tag {tag_id}: (0.000, 0.000, 0.000) m — world origin"]
                if T_smooth is not None:
                    for key, T_world_other in sorted(last_world_tags.items()):
                        if time.time() - last_tag_seen[key] <= 2:
                            pos = T_world_other[:3, 3]
                            lines.append(f"tag {key}: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m")
                gui_positions.content = "<br>".join(lines)
                if pelvis_pos is not None and age < 0.5:
                    gap = near_edge_y - pelvis_pos[1]
                    gui_relative.content = (f"**Robot relative to table (waist axis, tape-calibrated)** · "
                                            f"left/right x {pelvis_pos[0]:+.3f} m · "
                                            f"toward net y {pelvis_pos[1]:+.3f} m · "
                                            f"near-edge gap {gap:+.3f} m")
                else:
                    gui_relative.content = "**No fixed reference tag visible — distance unavailable**"
                if ball_fresh:
                    comparisons = []
                    for method, key in (("Box", "xyz_world"), ("Contour", "xyz_world_refined")):
                        value = ball_result.get(key)
                        comparisons.append(method + ": " + (" / ".join(f"{v*100:+.2f}" for v in value) + " cm (x/y/z)" if value is not None else "unavailable"))
                    gui_ball_comparison.content = "<br>".join(comparisons) + "<br>Contour is experimental; lower jitter does not guarantee lower bias."
                    left_count, right_count = map(len, ball_result["boxes"])
                    location = ("Stereo 3D position " + " ".join(f"{v:+.3f}" for v in ball_sphere.position) + " m"
                                if ball_sphere.visible else "No reliable 3D position in the table area")
                    gui_ball.content = (f"**Ball model {ball_result['inference_fps']:.1f} pairs/s** · "
                                        f"left {left_count} / right {right_count} candidates · "
                                        f"inference {ball_result['model_ms']:.1f} ms · **{center_source}** · {location}")
                else:
                    gui_ball.content = "Ball model: waiting for results or video stream disconnected"
    finally:
        joints.close()
        stream.close()
        ball_socket.close()
        pose_socket.close()
        ball_context.term()


if __name__ == "__main__":
    main()
