from pathlib import Path
p=Path('outputs/g1-ball-session/measure_ball.py');s=p.read_text().replace('    from ball_geometry import best_stereo_ball','    from ball_geometry import best_stereo_ball\n    from ball_center import refine_boxes')
s=s.replace("'local_pipeline_ms','truth_x_m'","'local_pipeline_ms','refined_status','refined_x_m','refined_y_m','refined_z_m','truth_x_m'")
s=s.replace("            row = dict(unix_time=",'''            refined_xyz = None
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
            row = dict(unix_time=''')
s=s.replace("            if ball:\n                row.update", "            row['refined_status'] = 'valid' if refined_xyz is not None else 'unavailable'\n            if refined_xyz is not None:\n                row.update(zip(['refined_x_m','refined_y_m','refined_z_m'],refined_xyz.tolist()))\n            if ball:\n                row.update")
s=s.replace('            packet.update(world_measurement=True,','            packet.update(xyz_world_refined=refined_xyz.tolist() if refined_xyz is not None else None,\n                          ball_refined=refined_ball, world_measurement=True,')
p.write_text(s)
p=Path('/home/jingxuan/Documents/Codex/2026-10-02/zhzeze/outputs/G1-Viser/repo/g1/stereo_viser.py');s=p.read_text()
s=s.replace('        gui_pos = server.gui.add_text', '''        gui_ball_method = server.gui.add_dropdown("Ball center method", options=("Box center", "Contour center (experimental)"), initial_value="Box center")
        gui_ball_comparison = server.gui.add_markdown("Contour comparison: waiting for frames")
        gui_pos = server.gui.add_text''')
s=s.replace('            ball_pose = ball_result.get("ball") if ball_fresh else None','            refined_mode = gui_ball_method.value == "Contour center (experimental)"\n            ball_pose = ball_result.get("ball_refined" if refined_mode else "ball") if ball_fresh else None')
s=s.replace('                    point = np.asarray(ball_result.get("xyz_world") if ball_result.get("xyz_world") is not None\n                                       else [np.nan, np.nan, np.nan])','                    selected_xyz = ball_result.get("xyz_world_refined" if refined_mode else "xyz_world")\n                    point = np.asarray(selected_xyz if selected_xyz is not None else [np.nan, np.nan, np.nan])')
s=s.replace('                    left_count, right_count = map(len, ball_result["boxes"])','''                    comparisons = []
                    for method, key in (("Box", "xyz_world"), ("Contour", "xyz_world_refined")):
                        value = ball_result.get(key)
                        comparisons.append(method + ": " + (" / ".join(f"{v*100:+.2f}" for v in value) + " cm (x/y/z)" if value is not None else "unavailable"))
                    gui_ball_comparison.content = "<br>".join(comparisons) + "<br>Contour is experimental; lower jitter does not guarantee lower bias."
                    left_count, right_count = map(len, ball_result["boxes"])''')
p.write_text(s)
