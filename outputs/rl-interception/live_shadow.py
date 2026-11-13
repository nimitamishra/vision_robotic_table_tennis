"""Read-only live ball-policy preview. This process never opens a SONIC socket.

It joins the trajectory and G1 pose streams by the robot's capture clock. A
moving ball must approach the captured right-hand ready plane before inference
is attempted. Outputs are diagnostics only; they are not robot commands.
"""

import argparse
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import msgpack
import numpy as np
import zmq

from policy_adapter import load_policy, right_hand_target


TRAINING_READY_Q = np.array([-0.2, 0.0, 0.0, 0.45, 0.0, 0.0, 0.0])


def joined_observation(trajectory, pose, ready, policy, previous=None, dt=None,
                       ready_valid=True):
    """Make one preview from two timestamp-aligned, validated live packets."""
    result = {"ball_frame_id": trajectory.get("frame_id"),
              "robot_frame_id": pose.get("frame_id"), "transmitted": False}
    ball_stamp = trajectory.get("timestamp_s")
    pose_stamp = pose.get("capture_monotonic_ns")
    if not isinstance(ball_stamp, (int, float)) or not isinstance(pose_stamp, int):
        return {**result, "status": "missing_capture_time"}
    skew = ball_stamp - pose_stamp * 1e-9
    result["sensor_skew_ms"] = round(skew * 1000, 2)
    if abs(skew) > 0.12:
        return {**result, "status": "sensor_clock_mismatch"}
    if (not pose.get("pose_fresh") or not pose.get("right_arm_fresh")
            or not pose.get("reference_seen_both_cameras")):
        return {**result, "status": "pose_or_arm_unavailable"}
    T = np.asarray(pose.get("T_world_pelvis"), dtype=float)
    q = np.asarray(pose.get("right_arm_q_rad"), dtype=float)
    dq = np.asarray(pose.get("right_arm_dq_rad_s"), dtype=float)
    wrist = np.asarray(pose.get("right_hand_pelvis_m"), dtype=float)
    if (T.shape != (4, 4) or q.shape != (7,) or dq.shape != (7,)
            or wrist.shape != (3,) or not all(np.all(np.isfinite(v)) for v in (T, q, dq, wrist))
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=0.02)):
        return {**result, "status": "invalid_robot_state"}
    result["right_hand_pelvis_m"] = wrist.tolist()
    result["right_arm_q_rad"] = q.tolist()
    result["right_arm_dq_rad_s"] = dq.tolist()
    result["training_ready_max_joint_gap_rad"] = round(float(np.max(np.abs(q - TRAINING_READY_Q))), 3)
    result["training_pose_mismatch"] = result["training_ready_max_joint_gap_rad"] > 0.5
    if trajectory.get("status") == "stationary":
        return {**result, "status": "stationary_ball"}
    if trajectory.get("status") != "tracking" or not trajectory.get("measurement_accepted"):
        return {**result, "status": "waiting_for_moving_ball"}
    if trajectory.get("measurement_age_s", 1.0) > 0.08:
        return {**result, "status": "stale_ball"}
    ball_world = np.asarray(trajectory.get("position_m"), dtype=float)
    velocity_world = np.asarray(trajectory.get("velocity_m_s"), dtype=float)
    if (ball_world.shape != (3,) or velocity_world.shape != (3,)
            or not np.all(np.isfinite(ball_world)) or not np.all(np.isfinite(velocity_world))):
        return {**result, "status": "invalid_ball_state"}
    R = T[:3, :3]
    ball = R.T @ (ball_world - T[:3, 3])
    velocity = R.T @ velocity_world
    result["ball_pelvis_m"] = ball.tolist()
    result["ball_velocity_pelvis_m_s"] = velocity.tolist()
    if velocity[0] >= -0.1:
        return {**result, "status": "ball_not_approaching_hand_plane"}
    intercept_time = (ready[0] - ball[0]) / velocity[0]
    result["time_to_intercept_s"] = float(intercept_time)
    # The proxy policy was trained on uninterrupted free flight. A table or net
    # contact before the hand plane invalidates that prediction; merely widening
    # the time horizon would make a physically impossible preview look feasible.
    events = trajectory.get("collision_events") or []
    if events and isinstance(events[0], dict):
        first = events[0]
        collision_time = first.get("time_s")
        if (isinstance(collision_time, (int, float)) and np.isfinite(collision_time)
                and 0 <= collision_time <= intercept_time):
            return {**result, "status": "collision_before_hand_plane",
                    "first_collision_type": first.get("type"),
                    "time_to_collision_s": float(collision_time)}
    if not 0.02 <= intercept_time <= 0.5:
        return {**result, "status": "intercept_outside_horizon"}
    crossing = ball + velocity * intercept_time + np.array([0, 0, -4.905 * intercept_time ** 2])
    result["crossing_pelvis_m"] = crossing.tolist()
    if np.max(np.abs(crossing - ready)) > 0.08:
        return {**result, "status": "intercept_outside_ready_region"}
    if result["training_pose_mismatch"]:
        return {**result, "status": "training_pose_mismatch"}
    if not ready_valid:
        return {**result, "status": "waiting_for_ready_pose"}
    target = right_hand_target(policy, q, dq, wrist, ready, ball, velocity,
                               intercept_time, previous, dt)
    return {**result, "status": "preview", "ready_right_pelvis_m": ready.tolist(),
            "proposed_right_pelvis_m": target.tolist(),
            "target_delta_cm": ((target - ready) * 100).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default="tcp://127.0.0.1:58213")
    ap.add_argument("--pose", default="tcp://127.0.0.1:5565")
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(__file__).parent / "runs/ball-proxy-02/policy.pt")
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).parent / "runs" /
                    ("live-shadow-" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    args = ap.parse_args()
    policy = load_policy(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps({
        "trajectory": args.trajectory, "pose": args.pose,
        "checkpoint": str(args.checkpoint.resolve()), "transmitted": False,
        "note": "read-only diagnostics; trained fixed-pelvis proxy; no SONIC control",
    }, indent=2))
    ctx = zmq.Context()
    poller = zmq.Poller()
    sockets = {}
    for name, endpoint in (("trajectory", args.trajectory), ("pose", args.pose)):
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(endpoint)
        sockets[name] = sock
        poller.register(sock, zmq.POLLIN)
    poses = deque(maxlen=60)
    ready = None
    ready_candidate_since = None
    ready_candidate_point = None
    previous = None
    previous_stamp = None
    last_epoch = None
    last_report = 0.0
    print("Read-only live shadow started; no SONIC socket is opened", flush=True)
    try:
        with (args.output / "preview.jsonl").open("w") as log:
            while True:
                events = dict(poller.poll(1000))
                if sockets["pose"] in events:
                    packet = msgpack.unpackb(sockets["pose"].recv(), raw=False)
                    poses.append(packet)
                    q = np.asarray(packet.get("right_arm_q_rad", []), dtype=float)
                    dq = np.asarray(packet.get("right_arm_dq_rad_s", []), dtype=float)
                    point = np.asarray(packet.get("right_hand_pelvis_m", []), dtype=float)
                    stamp_ns = packet.get("capture_monotonic_ns")
                    candidate_ok = (packet.get("pose_fresh") and packet.get("right_arm_fresh")
                                    and packet.get("reference_seen_both_cameras")
                                    and isinstance(stamp_ns, int)
                                    and q.shape == (7,) and dq.shape == (7,) and point.shape == (3,)
                                    and np.all(np.isfinite(q)) and np.all(np.isfinite(dq))
                                    and np.all(np.isfinite(point))
                                    and np.max(np.abs(q - TRAINING_READY_Q)) <= 0.5
                                    and np.max(np.abs(dq)) <= 0.1)
                    if not candidate_ok:
                        ready = None
                        ready_candidate_since = None
                        ready_candidate_point = None
                    elif (ready_candidate_since is None or
                          np.linalg.norm(point - ready_candidate_point) > 0.025):
                        ready_candidate_since = stamp_ns * 1e-9
                        ready_candidate_point = point.copy()
                        ready = None
                    elif stamp_ns * 1e-9 - ready_candidate_since >= 0.3 and ready is None:
                        ready = point.copy()
                if sockets["trajectory"] not in events:
                    continue
                trajectory = msgpack.unpackb(sockets["trajectory"].recv(), raw=False)
                stamp = trajectory.get("timestamp_s")
                if not poses or not isinstance(stamp, (int, float)):
                    preview = {"status": "waiting_for_pose", "transmitted": False}
                else:
                    pose = min(poses, key=lambda p: abs(p["capture_monotonic_ns"] * 1e-9 - stamp))
                    epoch = trajectory.get("track_epoch")
                    if epoch != last_epoch:
                        previous = None
                        previous_stamp = None
                        last_epoch = epoch
                    dt = stamp - previous_stamp if previous_stamp is not None else None
                    if dt is not None and not 0 < dt <= 0.2:
                        previous = None
                        dt = None
                    diagnostic_ready = (ready if ready is not None else
                                        np.asarray(pose.get("right_hand_pelvis_m", []), dtype=float))
                    preview = joined_observation(trajectory, pose, diagnostic_ready, policy,
                                                 previous, dt, ready_valid=ready is not None)
                    preview["ready_pose_captured"] = ready is not None
                    if preview["status"] == "preview":
                        previous = np.asarray(preview["proposed_right_pelvis_m"])
                        previous_stamp = stamp
                    else:
                        previous = None
                        previous_stamp = None
                preview["trajectory_status"] = trajectory.get("status")
                preview["time_utc"] = datetime.utcnow().isoformat() + "Z"
                log.write(json.dumps(preview) + "\n")
                now = time.monotonic()
                if now - last_report > 0.5:
                    log.flush()
                    temp = args.output / "latest.tmp"
                    temp.write_text(json.dumps(preview, indent=2))
                    temp.replace(args.output / "latest.json")
                    if preview["status"] != "waiting_for_moving_ball" or now - last_report > 3:
                        print(preview["status"], flush=True)
                    last_report = now
    finally:
        for sock in sockets.values():
            sock.close(0)
        ctx.term()


if __name__ == "__main__":
    main()
