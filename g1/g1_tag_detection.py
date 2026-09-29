# Runs on the Mac. Pulls frames from the G1 head RealSense and runs AprilTag (tag36h11) detection.
import sys
import cv2
from pupil_apriltags import Detector

from g1_stream import G1CameraStream

HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.123.164"
HEADLESS = "--headless" in sys.argv

detector = Detector(families="tag36h11")


stream = G1CameraStream(HOST)
print(f"connected to {HOST}")

while True:
    frame, _ = stream.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray)

    for tag in tags:
        print("Detected tag:", tag.tag_id, "center", tag.center.round(1))

        corners = tag.corners.astype(int)
        for i in range(4):
            p1 = tuple(corners[i])
            p2 = tuple(corners[(i + 1) % 4])
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

        center = tuple(tag.center.astype(int))
        cv2.putText(frame, f"ID {tag.tag_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if HEADLESS:
        continue
    cv2.imshow("G1 AprilTag", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

stream.close()
cv2.destroyAllWindows()
