"""Offline regression check for G1 waist-aware camera-to-pelvis localization.

Run with the same Python environment as G1 Viser. No robot or network needed.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from g1_stream import T_D435LINK_OPTICAL, make_T
from stereo_viser import (URDF_PATH, camera_yaw_pitch,
                          estimate_upright_pelvis, load_g1_with_head_tilt)
import yourdfpy


def pelvis_yaw(T):
    return np.arctan2(T[1, 0], T[0, 0])


def run():
    calibration = json.loads((Path(__file__).parent / "calibration.json").read_text())
    stock = (yourdfpy.URDF.load(str(URDF_PATH), load_meshes=False)
             .get_transform("d435_link", "pelvis") @ T_D435LINK_OPTICAL)
    measured_pitch = camera_yaw_pitch(np.asarray(
        calibration["calibration_pose"]["T_world_camera_optical"]))[1]
    tilt = measured_pitch - camera_yaw_pitch(stock)[1]
    urdf = load_g1_with_head_tilt(tilt)
    correction = calibration["robot"]["waist_forward_correction_m"]

    # Ground truth pelvis stays fixed; only the robot's waist yaw changes.
    true_pelvis = make_T(Rotation.from_euler("z", np.deg2rad(92)).as_matrix(),
                         [0.12, -0.51, 0.0])
    reference = true_pelvis[:3, 3] + true_pelvis[:3, :3] @ [correction, 0, 0]
    urdf.update_cfg({"waist_yaw_joint": np.asarray(0.0)})
    zero_camera_fk = urdf.get_transform("d435_link", "pelvis") @ T_D435LINK_OPTICAL

    print("Waist yaw | camera heading | corrected pelvis drift | corrected yaw drift | fixed-FK yaw error")
    for degrees in (-30, -15, 0, 15, 30):
        urdf.update_cfg({"waist_yaw_joint": np.asarray(np.deg2rad(degrees))})
        camera_fk = urdf.get_transform("d435_link", "pelvis") @ T_D435LINK_OPTICAL
        observed_camera = true_pelvis @ camera_fk
        estimated = estimate_upright_pelvis(observed_camera, camera_fk, correction)
        fixed_estimate = estimate_upright_pelvis(observed_camera, zero_camera_fk, correction)

        position_error = np.linalg.norm(estimated[:3, 3] - reference)
        yaw_error = abs(pelvis_yaw(estimated) - pelvis_yaw(true_pelvis))
        fixed_yaw_error = abs(pelvis_yaw(fixed_estimate) - pelvis_yaw(true_pelvis))
        heading = np.rad2deg(camera_yaw_pitch(observed_camera)[0])
        print(f"{degrees:+9.0f}° | {heading:+14.1f}° | {position_error*1000:19.6f} mm | "
              f"{np.rad2deg(yaw_error):19.6f}° | {np.rad2deg(fixed_yaw_error):18.1f}°")
        assert position_error < 1e-8, "pelvis drifted while only the waist moved"
        assert yaw_error < 1e-8, "pelvis yaw followed the camera"
        if degrees:
            assert fixed_yaw_error > np.deg2rad(10), "test did not exercise moving waist"
    print("PASS: waist-aware FK holds the pelvis fixed over the simulated sweep.")


if __name__ == "__main__":
    run()
