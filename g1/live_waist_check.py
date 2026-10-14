"""Record live, read-only waist/localization data for the SONIC rotation test."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yourdfpy
from pupil_apriltags import Detector

from g1_stream import T_D435LINK_OPTICAL, average_poses, make_T
from joint_stream import JointStream
from stereo_stream import StereoStream
from stereo_viser import (URDF_PATH, camera_yaw_pitch, detect_camera_tags,
                          estimate_upright_pelvis, load_g1_with_head_tilt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.123.164")
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads((Path(__file__).parent / "calibration.json").read_text())
    world = calibration["world"]
    references = {world["tag_id"]: np.asarray(world["T_world_tag"])}
    if world.get("fallback_tags_valid", True):
        references.update({int(k): np.asarray(v) for k, v in world["fallback_tags"].items()})
    correction = calibration["robot"]["waist_forward_correction_m"]
    stock = (yourdfpy.URDF.load(str(URDF_PATH), load_meshes=False)
             .get_transform("d435_link", "pelvis") @ T_D435LINK_OPTICAL)
    measured_pitch = camera_yaw_pitch(np.asarray(
        calibration["calibration_pose"]["T_world_camera_optical"]))[1]
    urdf = load_g1_with_head_tilt(measured_pitch - camera_yaw_pitch(stock)[1])
    urdf.update_cfg({"waist_yaw_joint": np.asarray(0.0)})
    zero_fk = urdf.get_transform("d435_link", "pelvis") @ T_D435LINK_OPTICAL
    detector = Detector(families=world["tag_family"], quad_decimate=1.0, refine_edges=1)
    stream = StereoStream(args.host)
    joints = JointStream(args.host)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["host_time", "frame_id", "frame_robot_ns", "reference_tag", "joint_robot_ns",
              "joint_age_ms", "waist_yaw_deg", "camera_yaw_deg", "pelvis_x_m", "pelvis_y_m",
              "pelvis_yaw_deg", "edge_gap_m", "fixed_fk_pelvis_y_m", "fixed_fk_pelvis_yaw_deg"]
    count = 0
    missing_joint = 0
    missing_tag = 0
    try:
        with args.output.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            end = time.monotonic() + args.seconds
            while time.monotonic() < end:
                left, right, frame_id, _ = stream.read()
                stamp = stream.last_frame_monotonic_ns
                sample = joints.nearest(stamp)
                if sample is None:
                    missing_joint += 1
                    continue
                found = {}
                for image, intr, T_left_camera in (
                    (left, stream.left_intr, np.eye(4)),
                    (right, stream.right_intr, stream.T_left_right),
                ):
                    for tag in detect_camera_tags(detector, image, intr, world["tag_size_m"], 1):
                        if tag.tag_id in references and tag.pose_R is not None:
                            found.setdefault(tag.tag_id, []).append(
                                T_left_camera @ make_T(tag.pose_R, tag.pose_t))
                reference = next((key for key in references if key in found), None)
                if reference is None:
                    missing_tag += 1
                    continue
                T_world_camera = (references[reference] @
                                  np.linalg.inv(average_poses(found[reference])) @
                                  stream.T_left_color)
                q = float(sample["waist_yaw_rad"])
                urdf.update_cfg({"waist_yaw_joint": np.asarray(q)})
                fk = urdf.get_transform("d435_link", "pelvis") @ T_D435LINK_OPTICAL
                pelvis = estimate_upright_pelvis(T_world_camera, fk, correction)
                fixed = estimate_upright_pelvis(T_world_camera, zero_fk, correction)
                yaw = lambda T: np.degrees(np.arctan2(T[1, 0], T[0, 0]))
                writer.writerow({
                    "host_time": time.time(), "frame_id": frame_id,
                    "frame_robot_ns": stamp, "reference_tag": reference,
                    "joint_robot_ns": sample["sample_monotonic_ns"],
                    "joint_age_ms": abs(stamp - sample["sample_monotonic_ns"]) / 1e6,
                    "waist_yaw_deg": np.degrees(q),
                    "camera_yaw_deg": np.degrees(camera_yaw_pitch(T_world_camera)[0]),
                    "pelvis_x_m": pelvis[0, 3], "pelvis_y_m": pelvis[1, 3],
                    "pelvis_yaw_deg": yaw(pelvis),
                    "edge_gap_m": -world["table"]["tag_center_inset_m"] - pelvis[1, 3],
                    "fixed_fk_pelvis_y_m": fixed[1, 3],
                    "fixed_fk_pelvis_yaw_deg": yaw(fixed),
                })
                count += 1
    finally:
        joints.close()
        stream.close()
    print(f"Recorded {count} localized frames to {args.output}; "
          f"missing joint {missing_joint}, missing tag {missing_tag}")


if __name__ == "__main__":
    main()
