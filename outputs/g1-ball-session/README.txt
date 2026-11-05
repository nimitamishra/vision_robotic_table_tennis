G1 static-ball measurement session — 2026-10-16

Status at preparation: robot reachable; camera TCP 5560 closed; SSH key login unavailable.
Local worker: V1 model, original stereo JPEG included in results.
Recorder: same-source-frame tag 5 pose and stereo ball, no temporal smoothing.
Recorder stops after 900 seconds; restart it for a later session. Worker remains idle if camera is absent.

Files:
  measure_ball.py: read-only recorder and same-frame world-coordinate publisher (5563).
  measurements-*.csv: raw coordinates and quality metrics, including rejected samples.
  frames-*.msgpack: concatenated MessagePack packets (read with msgpack.Unpacker), with source JPEG/calibration.
  control.json: sample label and optional surveyed ball-center [x,y,z] in metres.
  latest.json / latest.jpg: last processed diagnostics and detection image.

Important interpretation:
  Valid rate is over processed samples, not all 60 FPS camera frames.
  local_pipeline_ms starts at desktop receipt, not sensor exposure, and excludes network/camera time.
  capture_monotonic_ns from the existing robot server is sampled after wait_for_frames returns;
  it is a host-side receipt timestamp, not a measured exposure timestamp.
  Matching the source JPEG avoids mixing different image frames for world reconstruction.
  2 px reprojection gate is provisional, not a demonstrated accuracy guarantee.
  Tag size/orientation and stereo factory calibration remain subject to physical validation.
  Static-ball Z is the ball-center height above the tag-defined tabletop plane.

Initial physical test:
  Robot stationary; tag 5 visible and unchanged. Place ball on table near the centerline,
  roughly 30 cm toward the net from tag 5; keep hands clear. First verify detection before
  entering measured ground truth. Do not enter approximate placement as accurate truth.
