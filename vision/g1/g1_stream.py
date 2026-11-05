# Client for rs_stream_server.py running on the G1, plus the frame math shared by calibration and viz.
import json, socket, struct
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

G1_HOST = "192.168.123.164"
PORT = 5005

# World frame = frame of the reference tag, re-oriented so +z points out of the tag face:
#   x = tag right, y = tag up (as printed), z = out of the tag toward the viewer.
# pupil_apriltags tag frame is x right, y down, z into the tag, so this is a 180 deg rotation about x.
T_WORLD_TAG = np.diag([1.0, -1.0, -1.0, 1.0])

# D435 optical frame (x right, y down, z forward) expressed in URDF d435_link (x forward, y left, z up).
T_D435LINK_OPTICAL = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=float)


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("stream closed")
        buf += chunk
    return bytes(buf)


class G1CameraStream:
    """Yields (color_bgr, depth_m) from the G1 head RealSense. `self.intr` holds intrinsics."""

    def __init__(self, host=G1_HOST, port=PORT):
        self.sock = socket.create_connection((host, port), timeout=5)
        try:
            (n,) = struct.unpack(">I", recv_exact(self.sock, 4))
        except socket.timeout:
            raise ConnectionError(f"no header from {host}:{port} in 5s — is rs_stream_server.py running on the G1?")
        self.intr = json.loads(recv_exact(self.sock, n))
        i = self.intr
        self.K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]])

    @property
    def camera_params(self):
        i = self.intr
        return (i["fx"], i["fy"], i["cx"], i["cy"])

    def read(self):
        n_jpg, n_png = struct.unpack(">II", recv_exact(self.sock, 8))
        color = cv2.imdecode(np.frombuffer(recv_exact(self.sock, n_jpg), np.uint8), cv2.IMREAD_COLOR)
        depth = cv2.imdecode(np.frombuffer(recv_exact(self.sock, n_png), np.uint8), cv2.IMREAD_UNCHANGED)
        return color, depth.astype(np.float32) * self.intr["depth_scale"]

    def close(self):
        self.sock.close()


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def tag_to_T_cam_tag(tag):
    return make_T(tag.pose_R, tag.pose_t)


def T_world_cam_from_tag(tag, T_world_ref=T_WORLD_TAG):
    """Camera (optical) pose in world given a detection of the reference tag."""
    return T_world_ref @ np.linalg.inv(tag_to_T_cam_tag(tag))


def T_to_wxyz_pos(T):
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]]), T[:3, 3].copy()


def average_poses(Ts):
    R = Rotation.from_matrix(np.stack([T[:3, :3] for T in Ts])).mean().as_matrix()
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    return make_T(R, t)


def _depth_at(depth, uv, r=3):
    u, v = int(round(uv[0])), int(round(uv[1]))
    patch = depth[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1]
    patch = patch[patch > 0]
    return float(np.median(patch)) if patch.size else None


def measure_tag_size(tag, depth, K, inset=0.2):
    """Metric edge length of the tag's black border, from aligned depth.

    Depth right at the tag border is noisy, so sample points pulled `inset` of the way toward the
    center, deproject them, and scale the measured side back up.
    """
    Kinv = np.linalg.inv(K)
    pts = []
    for c in tag.corners:
        uv = c + inset * (tag.center - c)
        z = _depth_at(depth, uv)
        if z is None:
            return None
        pts.append(z * (Kinv @ np.array([uv[0], uv[1], 1.0])))
    pts = np.array(pts)
    sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    return float(np.mean(sides) / (1.0 - inset))
