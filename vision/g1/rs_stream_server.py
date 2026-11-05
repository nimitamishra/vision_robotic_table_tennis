# Runs on the G1 (Jetson). Streams RealSense D435i color (JPEG) + color-aligned depth (16-bit PNG) over TCP.
# Any number of clients can connect at once; each gets the latest frame (slow clients skip frames).
#
# Protocol: on connect, one length-prefixed JSON header (intrinsics, depth scale).
# Then per frame: >II (jpeg_len, depth_png_len) followed by the two payloads.
import json, socket, struct, threading
import cv2
import numpy as np
import pyrealsense2 as rs

PORT = 5005
W, H, FPS = 640, 480, 30

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
profile = pipe.start(cfg)
align = rs.align(rs.stream.color)

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
header = json.dumps({
    "width": intr.width, "height": intr.height,
    "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
    "dist": list(intr.coeffs),
    "depth_scale": profile.get_device().first_depth_sensor().get_depth_scale(),
}).encode()

latest = None  # (seq, packet bytes)
cond = threading.Condition()


def capture_loop():
    global latest
    seq = 0
    while True:
        frames = align.process(pipe.wait_for_frames())
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data())
        _, jpg = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, png = cv2.imencode(".png", depth, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        jpg, png = jpg.tobytes(), png.tobytes()
        seq += 1
        with cond:
            latest = (seq, struct.pack(">II", len(jpg), len(png)) + jpg + png)
            cond.notify_all()


def serve_client(conn, addr):
    print("client", addr, flush=True)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    last = 0
    try:
        conn.sendall(struct.pack(">I", len(header)) + header)
        while True:
            with cond:
                cond.wait_for(lambda: latest is not None and latest[0] != last)
                last, packet = latest
            conn.sendall(packet)
    except OSError:
        pass
    finally:
        print("client disconnected", addr, flush=True)
        conn.close()


srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", PORT))
srv.listen(8)
threading.Thread(target=capture_loop, daemon=True).start()
print(f"streaming on :{PORT}", flush=True)

try:
    while True:
        conn, addr = srv.accept()
        threading.Thread(target=serve_client, args=(conn, addr), daemon=True).start()
finally:
    pipe.stop()
