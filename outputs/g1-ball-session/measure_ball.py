"""Read-only, same-frame stereo-ball measurement. No motor commands.

Subscribes to ball_worker --include-frame; records replayable packets and raw
unfiltered world positions. Latest-only processing: coverage is over processed
samples, not all camera frames. Control file: {"label":"near_center", "truth_m":[x,y,z]}.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import msgpack
import numpy as np
import zmq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--g1-dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--input', default='tcp://127.0.0.1:5561')
    ap.add_argument('--publish', default='tcp://127.0.0.1:5563')
    ap.add_argument('--seconds', type=float, default=0)
    args = ap.parse_args()
    sys.path.insert(0, args.g1_dir)
    from stereo_viser import Detector, detect_camera_tags, average_poses, make_T
    from ball_geometry import best_stereo_ball
    from ball_center import refine_boxes
    cfg = json.loads((Path(args.g1_dir)/'calibration.json').read_text())
    world = cfg['world']
    detector = Detector(families=world['tag_family'], quad_decimate=1.0, refine_edges=1)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out/'session-calibration.json').write_text(json.dumps(cfg, indent=2))
    (out/('session-calibration-'+time.strftime('%Y%m%d-%H%M%S')+'.json')).write_text(json.dumps(cfg, indent=2))
    control = out/'control.json'
    if not control.exists(): control.write_text(json.dumps({'label':'setup', 'truth_m':None}))
    context = zmq.Context()
    source = context.socket(zmq.SUB); source.setsockopt(zmq.SUBSCRIBE, b''); source.setsockopt(zmq.CONFLATE, 1)
    source.connect(args.input)
    pub = context.socket(zmq.PUB); pub.setsockopt(zmq.SNDHWM, 1); pub.bind(args.publish)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    csv_file = (out/f'measurements-{stamp}.csv').open('w', newline='')
    raw = (out/f'frames-{stamp}.msgpack').open('wb')
    fields = ['unix_time','label','frame_id','capture_monotonic_ns','left_count','right_count',
              'tag5_views','status','x_m','y_m','z_m','ray_error_m','reprojection_error_px',
              'local_pipeline_ms','refined_status','refined_x_m','refined_y_m','refined_z_m','truth_x_m','truth_y_m','truth_z_m','position_error_m']
    writer = csv.DictWriter(csv_file, fieldnames=fields); writer.writeheader()
    start = report_time = time.monotonic(); count = valid = 0
    try:
        while not args.seconds or time.monotonic()-start < args.seconds:
            if not source.poll(500): continue
            packet = msgpack.unpackb(source.recv(), raw=False)
            if 'source_jpg' not in packet: raise RuntimeError('Start ball_worker with --include-frame')
            try: settings = json.loads(control.read_text())
            except (ValueError, OSError): settings = {'label':'control_error','truth_m':None}
            pair = cv2.imdecode(np.frombuffer(packet['source_jpg'], np.uint8), cv2.IMREAD_GRAYSCALE)
            if pair is None: continue
            cal = packet['calibration']; width = cal['left_intrinsics']['width']
            poses = []; tag_ids = []
            for side, key, T in [(0,'left_intrinsics',np.eye(4)), (1,'right_intrinsics',np.array(cal['T_left_right']))]:
                intr = cal[key]; gray = pair[:,side*width:(side+1)*width]
                K = np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
                # Pupil pose estimation assumes undistorted pinhole pixels.
                dist = np.asarray(intr.get('distortion',[0]*5), float)
                if np.any(dist): gray = cv2.undistort(gray,K,dist)
                tags = detect_camera_tags(detector, gray, intr, world['tag_size_m'], 1)
                tag_ids.append([int(t.tag_id) for t in tags])
                for tag in tags:
                    if tag.tag_id == world['tag_id']:
                        poses.append(T @ make_T(tag.pose_R,tag.pose_t))
            ball = packet.get('ball'); xyz = None
            status = 'no_stereo_ball' if ball is None else 'no_tag5'
            T_world_left = None
            if poses:
                T_world_left = np.array(world['T_world_tag']) @ np.linalg.inv(average_poses(poses))
                # Select among all stereo candidates AFTER the table-volume check.
                candidates = []
                for lb in packet['boxes'][0]:
                    for rb in packet['boxes'][1]:
                        candidate = best_stereo_ball([lb], [rb], cal)
                        if candidate is None: continue
                        candidate_xyz = (T_world_left @ np.r_[candidate['xyz_left'],1])[:3]
                        if (np.all(candidate_xyz >= world['play_volume_min_m']) and
                                np.all(candidate_xyz <= world['play_volume_max_m'])):
                            score = np.prod(candidate['confidence']) / (0.5 + candidate['reprojection_error_px'])
                            candidates.append((score, candidate))
                if candidates:
                    ball = max(candidates, key=lambda v:v[0])[1]
                    packet['ball'] = ball
                if ball:
                    xyz = (T_world_left @ np.r_[ball['xyz_left'],1])[:3]
                    status = 'valid' if np.all(xyz >= world['play_volume_min_m']) and np.all(xyz <= world['play_volume_max_m']) else 'outside_volume'
            refined_xyz = None
            refined_ball = None
            refined_boxes = [refine_boxes(pair[:,i*width:(i+1)*width], packet['boxes'][i])[0] for i in (0,1)]
            if T_world_left is not None:
                refined_candidates = []
                for lb in refined_boxes[0]:
                    for rb in refined_boxes[1]:
                        candidate = best_stereo_ball([lb], [rb], cal)
                        if candidate is None: continue
                        cp = (T_world_left @ np.r_[candidate['xyz_left'],1])[:3]
                        if np.all(cp >= world['play_volume_min_m']) and np.all(cp <= world['play_volume_max_m']):
                            score = np.prod(candidate['confidence'])/(.5+candidate['reprojection_error_px'])
                            refined_candidates.append((score,candidate,cp))
                if refined_candidates:
                    _,refined_ball,refined_xyz = max(refined_candidates,key=lambda v:v[0])
            row = dict(unix_time=time.time(), label=settings.get('label','setup'),frame_id=packet['frame_id'],
                       capture_monotonic_ns=packet.get('capture_monotonic_ns'),
                       left_count=len(packet['boxes'][0]),right_count=len(packet['boxes'][1]),
                       tag5_views=len(poses),status=status,
                       local_pipeline_ms=1000*(time.monotonic()-packet['source_received_monotonic']))
            row['refined_status'] = 'valid' if refined_xyz is not None else 'unavailable'
            if refined_xyz is not None:
                row.update(zip(['refined_x_m','refined_y_m','refined_z_m'],refined_xyz.tolist()))
            if ball:
                row.update(ray_error_m=ball['ray_error_m'],reprojection_error_px=ball.get('reprojection_error_px'))
            if xyz is not None: row.update(zip(['x_m','y_m','z_m'],xyz.tolist()))
            truth = settings.get('truth_m')
            if truth is not None:
                row.update(zip(['truth_x_m','truth_y_m','truth_z_m'],truth))
                if xyz is not None: row['position_error_m']=float(np.linalg.norm(xyz-np.array(truth)))
            writer.writerow(row)
            packet.update(xyz_world_refined=refined_xyz.tolist() if refined_xyz is not None else None,
                          ball_refined=refined_ball, world_measurement=True, xyz_world=xyz.tolist() if status=='valid' else None,
                          measurement_status=status, tag_ids=tag_ids,
                          T_world_left=T_world_left.tolist() if T_world_left is not None else None,
                          measurement_label=row['label'],truth_m=truth)
            if row['label'].startswith('static_'):
                raw.write(msgpack.packb(packet,use_bin_type=True))
            pub.send(msgpack.packb({k:v for k,v in packet.items() if k not in ('source_jpg','calibration')},use_bin_type=True))
            count += 1; valid += status=='valid'
            now=time.monotonic()
            if now-report_time >= 1:
                summary={**row,'tag_ids':tag_ids,'processed_samples':count,'valid_samples':valid,
                         'processed_hz':count/(now-start),'coverage_scope':'processed_samples_only',
                         'latency_scope':'desktop_receive_to_measurement; excludes camera/transport'}
                tmp=out/'latest.tmp';tmp.write_text(json.dumps(summary,indent=2));tmp.replace(out/'latest.json')
                preview=cv2.cvtColor(pair,cv2.COLOR_GRAY2BGR)
                for side,boxes in enumerate(packet['boxes']):
                    for x1,y1,x2,y2,confidence,_ in boxes:
                        cv2.rectangle(preview,(round(x1)+side*width,round(y1)),(round(x2)+side*width,round(y2)),(0,255,0),1)
                cv2.imwrite(str(out/'latest.jpg'),preview)
                csv_file.flush();raw.flush();report_time=now
                print(json.dumps(summary),flush=True)
    finally:
        csv_file.close();raw.close();source.close(0);pub.close(0);context.term()

if __name__=='__main__': main()
