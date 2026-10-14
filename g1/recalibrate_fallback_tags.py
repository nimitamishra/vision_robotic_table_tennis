"""Re-measure fixed table tags from simultaneous stereo views of tag 5."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation

from g1_stream import average_poses, make_T
from stereo_stream import StereoStream
from stereo_viser import detect_camera_tags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.123.164")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--calib", type=Path, default=Path(__file__).parent / "calibration.json")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    calibration = json.loads(args.calib.read_text())
    world = calibration["world"]
    origin = world["tag_id"]
    T_world_origin = np.asarray(world["T_world_tag"])
    detector = Detector(families=world["tag_family"], quad_decimate=1.0, refine_edges=1)
    samples = defaultdict(list)
    stream = StereoStream(args.host)
    try:
        for _ in range(args.frames):
            left, right, _, _ = stream.read()
            fused = defaultdict(list)
            for image, intr, T_left_camera in (
                (left, stream.left_intr, np.eye(4)),
                (right, stream.right_intr, stream.T_left_right),
            ):
                for tag in detect_camera_tags(detector, image, intr, world["tag_size_m"], 1):
                    if tag.pose_R is not None and tag.tag_id in (origin, 0, 1, 2):
                        fused[tag.tag_id].append(T_left_camera @ make_T(tag.pose_R, tag.pose_t))
            if origin not in fused:
                continue
            T_world_left = T_world_origin @ np.linalg.inv(average_poses(fused[origin]))
            for key in (0, 1, 2):
                if key in fused:
                    samples[key].append(T_world_left @ average_poses(fused[key]))
    finally:
        stream.close()

    valid = True
    new_poses = {}
    for key in (0, 1, 2):
        poses = samples[key]
        if len(poses) < 20:
            print(f"tag {key}: only {len(poses)} simultaneous samples")
            valid = False
            continue
        positions = np.asarray([T[:3, 3] for T in poses])
        center = np.median(positions, axis=0)
        error = np.linalg.norm(positions - center, axis=1)
        p95_mm = np.percentile(error, 95) * 1000
        rotation = Rotation.from_matrix([T[:3, :3] for T in poses]).mean().as_matrix()
        T_world_other = make_T(rotation, center)
        new_poses[str(key)] = T_world_other.tolist()
        old = np.asarray(world["fallback_tags"][str(key)])[:3, 3]
        print(f"tag {key}: n={len(poses)} xyz={np.round(center, 4)} m; "
              f"p95 scatter={p95_mm:.1f} mm; old-to-new={np.linalg.norm(center-old)*100:.1f} cm")
        if p95_mm > 20:
            valid = False
    if args.write:
        if not valid:
            raise SystemExit("Calibration not written: insufficient or inconsistent tag samples")
        world["fallback_tags"] = new_poses
        world["fallback_tags_valid"] = True
        world["fallback_note"] = (
            "Re-measured from simultaneous stereo observations of tag 5 on 2026-10-09 "
            f"using {args.frames} frames after tags were moved; do not move fixed tags without recalibrating.")
        args.calib.write_text(json.dumps(calibration, indent=2, ensure_ascii=False) + "\n")
        print(f"Wrote {args.calib}")


if __name__ == "__main__":
    main()
