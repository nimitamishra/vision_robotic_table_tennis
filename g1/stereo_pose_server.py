"""Publish G1 D435i stereo IR frames and factory camera calibration over ZMQ.

Runs on the G1 Jetson. The color and depth streams are not started.
"""

import argparse
import time

import cv2
import msgpack
import numpy as np
import pyrealsense2 as rs
import zmq


def transform_matrix(source, target):
    """Return target <- source using librealsense's own point transform."""
    extrinsics = source.get_extrinsics_to(target)
    origin = np.asarray(rs.rs2_transform_point_to_point(extrinsics, [0, 0, 0]))
    columns = [np.asarray(rs.rs2_transform_point_to_point(extrinsics, axis)) - origin
               for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack(columns)
    transform[:3, 3] = origin
    return transform.tolist()


def intrinsics(profile):
    camera = profile.as_video_stream_profile().get_intrinsics()
    return {
        "width": camera.width,
        "height": camera.height,
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.ppx,
        "cy": camera.ppy,
        "distortion": list(camera.coeffs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, choices=[6, 10, 30, 60, 90], default=30)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--port", type=int, default=5560)
    args = parser.parse_args()
    width = args.width or (480 if args.fps == 60 else 640)
    height = args.height or (270 if args.fps == 60 else 480)

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.SNDHWM, 1)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.bind(f"tcp://0.0.0.0:{args.port}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device("254322074777")
    for index in (1, 2):
        config.enable_stream(rs.stream.infrared, index, width, height, rs.format.y8, args.fps)

    profile = pipeline.start(config)
    try:
        streams = profile.get_streams()
        left_profile = next(p for p in streams if p.stream_type() == rs.stream.infrared and p.stream_index() == 1)
        right_profile = next(p for p in streams if p.stream_type() == rs.stream.infrared and p.stream_index() == 2)
        device = profile.get_device()
        color_profile = next(
            p for sensor in device.query_sensors() for p in sensor.get_stream_profiles()
            if p.stream_type() == rs.stream.color and p.is_video_stream_profile()
            and p.as_video_stream_profile().width() == 640
            and p.as_video_stream_profile().height() == 480
            and p.fps() == 30 and p.format() == rs.format.bgr8
        )
        calibration = {
            "left_intrinsics": intrinsics(left_profile),
            "right_intrinsics": intrinsics(right_profile),
            "T_left_right": transform_matrix(right_profile, left_profile),
            "T_left_color": transform_matrix(color_profile, left_profile),
        }
        print(f"Stereo IR ready on {args.port}: {width}x{height} @ {args.fps} FPS", flush=True)
        print(f"baseline {abs(calibration['T_left_right'][0][3]) * 1000:.1f} mm", flush=True)

        count = 0
        started = time.monotonic()
        while True:
            frames = pipeline.wait_for_frames(5000)
            capture_monotonic_ns = time.monotonic_ns()
            left = frames.get_infrared_frame(1)
            right = frames.get_infrared_frame(2)
            if not left or not right:
                continue
            pair = np.hstack((np.asanyarray(left.get_data()), np.asanyarray(right.get_data())))
            ok, encoded = cv2.imencode(".jpg", pair, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                continue
            packet = {
                "left_id": left.get_frame_number(),
                "right_id": right.get_frame_number(),
                "capture_monotonic_ns": capture_monotonic_ns,
                "jpg": encoded.tobytes(),
                "calibration": calibration,
            }
            publisher.send(msgpack.packb(packet, use_bin_type=True))
            count += 1
            if time.monotonic() - started >= 5:
                elapsed = time.monotonic() - started
                print(f"Published {count / elapsed:.1f} FPS", flush=True)
                count = 0
                started = time.monotonic()
    finally:
        pipeline.stop()
        publisher.close()
        context.term()


if __name__ == "__main__":
    main()
