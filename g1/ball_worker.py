"""Run stereo ping-pong detection at camera rate and publish only the newest result."""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import msgpack
import numpy as np
import torch
import zmq
from ultralytics import YOLO

from ball_geometry import best_stereo_ball


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="tcp://192.168.123.164:5560")
    parser.add_argument("--publish", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--ready-file")
    args = parser.parse_args()
    if not Path(args.model).is_file():
        raise FileNotFoundError(args.model)
    if not torch.cuda.is_available():
        raise RuntimeError("The ping-pong ball model requires a CUDA GPU on this computer")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    model = YOLO(args.model)
    if model.task != "detect":
        raise ValueError("The selected model must be an object detection model")
    options = dict(imgsz=args.imgsz, device=0, quantize=16, verbose=False,
                   max_det=5, rect=True, conf=args.conf)
    blank = np.zeros((270, 480, 3), dtype=np.uint8)
    for _ in range(5):
        model.predict([blank, blank], **options)
    torch.cuda.synchronize()

    ctx = zmq.Context()
    source = ctx.socket(zmq.SUB)
    source.setsockopt(zmq.SUBSCRIBE, b"")
    source.setsockopt(zmq.CONFLATE, 1)
    source.setsockopt(zmq.LINGER, 0)
    source.connect(args.camera)
    publisher = ctx.socket(zmq.PUB)
    publisher.setsockopt(zmq.SNDHWM, 1)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.bind(args.publish)
    if args.ready_file:
        Path(args.ready_file).write_text(str(os.getpid()))
    print(f"Ball model ready: {args.model}; camera {args.camera}; results {args.publish}", flush=True)
    count = 0
    interval_start = time.monotonic()
    latest_preview = None
    try:
        while True:
            if not source.poll(1000):
                continue
            packet = msgpack.unpackb(source.recv(), raw=False)
            if packet.get("left_id") != packet.get("right_id"):
                continue
            received = time.monotonic()
            pair = cv2.imdecode(np.frombuffer(packet["jpg"], np.uint8), cv2.IMREAD_COLOR)
            if pair is None or pair.shape[1] % 2:
                continue
            width = pair.shape[1] // 2
            predictions = model.predict([pair[:, :width], pair[:, width:]], **options)
            boxes = [prediction.boxes.data.cpu().numpy().tolist() for prediction in predictions]
            position = best_stereo_ball(boxes[0], boxes[1], packet["calibration"])
            finished = time.monotonic()
            count += 1
            elapsed = finished - interval_start
            rate = count / elapsed if elapsed else 0
            result = {"frame_id": packet["left_id"], "boxes": boxes,
                      "ball": position, "camera_shape": [pair.shape[0], width],
                      "model_ms": (finished - received) * 1000,
                      "inference_fps": rate, "timestamp": finished}
            if count % 3 == 0:
                for side, rows in enumerate(boxes):
                    for x1, y1, x2, y2, confidence, _ in rows:
                        corner1 = (round(x1) + side * width, round(y1))
                        corner2 = (round(x2) + side * width, round(y2))
                        cv2.rectangle(pair, corner1, corner2, (0, 255, 0), 1)
                        cv2.putText(pair, f"{confidence:.2f}", (corner1[0], max(12, corner1[1] - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                ok, preview = cv2.imencode(".jpg", pair, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    latest_preview = preview.tobytes()
            if latest_preview is not None:
                result["preview_jpg"] = latest_preview
            publisher.send(msgpack.packb(result, use_bin_type=True))
            if elapsed >= 5:
                print(json.dumps({"inference_fps": round(rate, 1),
                                  "frame_id": result["frame_id"],
                                  "ball": position}), flush=True)
                interval_start, count = finished, 0
    finally:
        source.close()
        publisher.close()
        ctx.term()


if __name__ == "__main__":
    main()
