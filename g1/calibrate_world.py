# Defines the global (world) frame from a reference AprilTag and saves it to calibration.json.
#
# - Measures the printed tag size from RealSense depth (or use --tag-size to set it).
# - Averages the camera pose over many frames and records it as the calibration pose.
#
#   python g1/calibrate_world.py                 # tag 0, measure size from depth
#   python g1/calibrate_world.py --tag-size 0.16 # known tag size in meters
import argparse, json, time
from pathlib import Path
import cv2
import numpy as np
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation

from g1_stream import (G1_HOST, G1CameraStream, T_WORLD_TAG, T_world_cam_from_tag, average_poses,
                       measure_tag_size)

CALIB_PATH = Path(__file__).parent / "calibration.json"

ap = argparse.ArgumentParser()
ap.add_argument("--host", default=G1_HOST)
ap.add_argument("--tag-id", type=int, default=0)
ap.add_argument("--tag-size", type=float, default=None, help="black-border edge length in meters")
ap.add_argument("--frames", type=int, default=60)
args = ap.parse_args()

detector = Detector(families="tag36h11")
stream = G1CameraStream(args.host)
print(f"connected, intrinsics fx={stream.intr['fx']:.1f} fy={stream.intr['fy']:.1f}")


def collect(n, tag_size=None):
    out, t0 = [], time.time()
    while len(out) < n:
        if time.time() - t0 > 30:
            raise SystemExit(f"tag {args.tag_id} not seen in enough frames ({len(out)}/{n}) in 30s")
        color, depth = stream.read()
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        kw = dict(estimate_tag_pose=True, camera_params=stream.camera_params, tag_size=tag_size) if tag_size else {}
        tags = [t for t in detector.detect(gray, **kw) if t.tag_id == args.tag_id]
        if tags:
            out.append((tags[0], depth))
    return out


tag_size = args.tag_size
if tag_size is None:
    print("measuring tag size from depth...")
    sizes = [s for tag, depth in collect(args.frames) if (s := measure_tag_size(tag, depth, stream.K))]
    if len(sizes) < 10:
        raise SystemExit("couldn't get valid depth on the tag; pass --tag-size")
    tag_size = float(np.median(sizes))
    print(f"  tag size = {tag_size * 100:.2f} cm  (std {np.std(sizes) * 100:.2f} cm over {len(sizes)} frames)")

print("estimating camera pose...")
samples = collect(args.frames, tag_size)
Ts = [T_world_cam_from_tag(tag) for tag, _ in samples]
T = average_poses(Ts)
spread_mm = np.std([Ti[:3, 3] for Ti in Ts], axis=0) * 1000
dist = np.linalg.norm(T[:3, 3])
rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
print(f"  camera position in world: {np.round(T[:3, 3], 3)} m  (|d| = {dist:.3f} m)")
print(f"  camera orientation (xyz euler, deg): {np.round(rpy, 1)}")
print(f"  jitter (std, mm): {np.round(spread_mm, 1)}")

calib = {
    "world": {
        "definition": "frame of the reference tag; x = tag right, y = tag up, z = out of tag face",
        "tag_family": "tag36h11",
        "tag_id": args.tag_id,
        "tag_size_m": tag_size,
        "T_world_tag": T_WORLD_TAG.tolist(),
    },
    "camera": {"name": "G1 head D435i", "intrinsics": stream.intr},
    "calibration_pose": {"T_world_camera_optical": T.tolist(), "jitter_mm": spread_mm.tolist()},
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
CALIB_PATH.write_text(json.dumps(calib, indent=2))
print(f"saved {CALIB_PATH}")
stream.close()
