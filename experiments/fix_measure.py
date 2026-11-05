from pathlib import Path
p=Path('outputs/g1-ball-session/measure_ball.py');s=p.read_text();s=s.replace('    from stereo_viser import Detector, detect_camera_tags, average_poses, make_T','    from stereo_viser import Detector, detect_camera_tags, average_poses, make_T\n    from ball_geometry import best_stereo_ball')
s=s.replace("                if ball:\n                    xyz =",'''                # Select among all stereo candidates AFTER the table-volume check.
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
                    xyz =''')
s=s.replace("            raw.write(msgpack.packb(packet,use_bin_type=True))","            if row['label'].startswith('static_'):\n                raw.write(msgpack.packb(packet,use_bin_type=True))")
p.write_text(s)
p=Path('/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/repo/g1/stereo_viser.py');s=p.read_text();s=s.replace('            for tag in detections]','            for tag in detections if tag.pose_R is not None and tag.pose_t is not None\n            and np.all(np.isfinite(tag.pose_R)) and np.all(np.isfinite(tag.pose_t))\n            and np.linalg.det(tag.pose_R) > 0.9\n            and np.allclose(tag.pose_R.T @ tag.pose_R, np.eye(3), atol=0.01)]');p.write_text(s)
