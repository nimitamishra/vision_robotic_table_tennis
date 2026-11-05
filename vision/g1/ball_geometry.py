"""Triangulate an object seen by the G1's calibrated infrared camera pair."""

import cv2
import numpy as np


def box_center(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=float)


def camera_ray(pixel, intrinsics):
    matrix = np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                       [0, intrinsics["fy"], intrinsics["cy"]], [0, 0, 1]], dtype=float)
    distortion = np.asarray(intrinsics.get("distortion", [0] * 5), dtype=float)
    xy = cv2.undistortPoints(np.asarray(pixel, dtype=float).reshape(1, 1, 2),
                             matrix, distortion).reshape(2)
    ray = np.array([xy[0], xy[1], 1.0])
    return ray / np.linalg.norm(ray)


def triangulate_pair(left_box, right_box, left_intr, right_intr, T_left_right):
    """Return (xyz in left optical frame, ray separation), or None if implausible."""
    left_ray = camera_ray(box_center(left_box), left_intr)
    right_ray = T_left_right[:3, :3] @ camera_ray(box_center(right_box), right_intr)
    right_origin = T_left_right[:3, 3]
    if not np.all(np.isfinite([left_ray, right_ray])) or np.linalg.norm(np.cross(left_ray, right_ray)) < 1e-5:
        return None
    distances = np.linalg.lstsq(np.column_stack((left_ray, -right_ray)),
                                right_origin, rcond=None)[0]
    if min(distances) <= 0:
        return None
    left_point = left_ray * distances[0]
    right_point = right_origin + right_ray * distances[1]
    separation = float(np.linalg.norm(left_point - right_point))
    point = (left_point + right_point) / 2
    if not 0.25 < point[2] < 10 or separation > 0.08:
        return None
    return point, separation


def best_stereo_ball(left_boxes, right_boxes, calibration):
    """Match high-confidence ball boxes using calibrated ray agreement."""
    transform = np.asarray(calibration["T_left_right"], dtype=float)
    options = []
    for left in left_boxes[:5]:
        for right in right_boxes[:5]:
            match = triangulate_pair(left, right, calibration["left_intrinsics"],
                                     calibration["right_intrinsics"], transform)
            if match is None:
                continue
            point, error = match
            right_point = transform[:3, :3].T @ (point - transform[:3, 3])
            if right_point[2] <= 0:
                continue
            reprojection = max(np.linalg.norm(project_point(point, calibration["left_intrinsics"]) - box_center(left)),
                               np.linalg.norm(project_point(right_point, calibration["right_intrinsics"]) - box_center(right)))
            # Initial pixel gate; validate against real captures before tuning.
            if not np.isfinite(reprojection) or reprojection > 2.0:
                continue
            score = float(left[4] * right[4]) / (0.5 + reprojection)
            options.append((score, point, error, reprojection, left, right))
    if not options:
        return None
    _, point, error, reprojection, left, right = max(options, key=lambda value: value[0])
    return {"xyz_left": point.tolist(), "ray_error_m": error,
            "reprojection_error_px": float(reprojection),
            "left_center_px": box_center(left).tolist(), "right_center_px": box_center(right).tolist(),
            "confidence": [float(left[4]), float(right[4])]}


def project_point(point, intr):
    matrix = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], float)
    pixels, _ = cv2.projectPoints(np.asarray(point, float).reshape(1, 3), np.zeros(3), np.zeros(3),
                                  matrix, np.asarray(intr.get("distortion", [0]*5), float))
    return pixels.reshape(2)
