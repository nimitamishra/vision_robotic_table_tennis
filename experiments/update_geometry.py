from pathlib import Path
p=Path('/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/repo/g1/ball_geometry.py')
s=p.read_text().replace('    distances = np.linalg.lstsq', '    if not np.all(np.isfinite([left_ray, right_ray])) or np.linalg.norm(np.cross(left_ray, right_ray)) < 1e-5:\n        return None\n    distances = np.linalg.lstsq')
s=s.replace('            score = float(left[4] * right[4]) / (0.01 + error)\n            options.append((score, point, error))', '''            right_point = transform[:3, :3].T @ (point - transform[:3, 3])
            if right_point[2] <= 0:
                continue
            reprojection = max(np.linalg.norm(project_point(point, calibration["left_intrinsics"]) - box_center(left)),
                               np.linalg.norm(project_point(right_point, calibration["right_intrinsics"]) - box_center(right)))
            # Initial pixel gate; validate against real captures before tuning.
            if not np.isfinite(reprojection) or reprojection > 2.0:
                continue
            score = float(left[4] * right[4]) / (0.5 + reprojection)
            options.append((score, point, error, reprojection, left, right))''')
s=s.replace('    _, point, error = max(options, key=lambda value: value[0])\n    return {"xyz_left": point.tolist(), "ray_error_m": error}', '''    _, point, error, reprojection, left, right = max(options, key=lambda value: value[0])
    return {"xyz_left": point.tolist(), "ray_error_m": error,
            "reprojection_error_px": float(reprojection),
            "left_center_px": box_center(left).tolist(), "right_center_px": box_center(right).tolist(),
            "confidence": [float(left[4]), float(right[4])]}


def project_point(point, intr):
    matrix = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], float)
    pixels, _ = cv2.projectPoints(np.asarray(point, float).reshape(1, 3), np.zeros(3), np.zeros(3),
                                  matrix, np.asarray(intr.get("distortion", [0]*5), float))
    return pixels.reshape(2)''')
p.write_text(s)
