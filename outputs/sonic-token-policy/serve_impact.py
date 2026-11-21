"""Live side-feed contact demonstration using the exact training physics."""
import argparse
import hashlib
import json
import threading
import time
from pathlib import Path
import numpy as np
import torch
import trimesh
import viser
from scipy.spatial.transform import Rotation
from decoder_env import ROOT
from impact_env import ImpactEnv, CONFIG, sample_cases
from impact_runtime import load_policy, predict
from train_impact import initialize

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, default=8092)
    ap.add_argument('--checkpoint', type=Path); ap.add_argument('--autoplay', action='store_true'); a = ap.parse_args()
    if a.checkpoint is None:
        manifest = json.loads((ROOT/'impact-deployment.json').read_text())
        a.checkpoint = Path(manifest['checkpoint'])
        if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() != manifest['checkpoint_sha256']:
            raise ValueError('Deployed checkpoint changed')
    torch.set_num_threads(1)
    metadata = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    env = ImpactEnv(residual_scale=metadata['residual_scale']); policy, _ = load_policy(a.checkpoint, env)
    previous = initialize(ROOT/'runs/reach-v2/policy.pt').eval()
    report = json.loads((a.checkpoint.parent/'report.json').read_text())
    server = viser.ViserServer(host='127.0.0.1', port=a.port)
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/floor', plane='xy', width=3, height=3, cell_size=.25)
    server.gui.add_markdown('## SONIC Token Policy — Ball Contact\n'
        '**141 observations → 64D token → frozen SONIC → MuJoCo**\n\n'
        'Side-fed ball, 4–12 cm target offsets. Red: racket; yellow: ball; green: predicted intercept.\n\n'
        'This is a low, side-facing contact exercise. **No table rally or real-robot commands.**')
    r = report['test_policy']; z = report['test_constant_token']
    old = report.get('test_previous_reaching_policy', report['test_initial_policy'])
    server.gui.add_markdown(f"Held-out returns: **{r['successful_returns']}/{r['episodes']} ({r['return_fraction']:.1%})**\n\n"
        f"Constant token: {z['return_fraction']:.1%} · Previous reaching policy: {old['return_fraction']:.1%}\n\n"
        'Success requires a forced face contact and outward ball velocity. Mount and impact properties are provisional.')
    mode = server.gui.add_dropdown('Controller', options=('Trained contact policy', 'Previous reaching policy', 'Constant token'))
    new = server.gui.add_button('New ball'); replay = server.gui.add_button('Replay same ball')
    pause = server.gui.add_checkbox('Paused', initial_value=False)
    repeat = server.gui.add_checkbox('Repeat new balls', initial_value=a.autoplay)
    speed = server.gui.add_slider('Playback speed', min=.25, max=1., step=.25, initial_value=.5)
    status = server.gui.add_html('<div style="height:190px">Ready.</div>')
    lock = threading.Lock(); state = {'request': 'new' if a.autoplay else None}
    @new.on_click
    def new_ball(_):
        with lock:state['request'] = 'new'
    @replay.on_click
    def same_ball(_):
        with lock:state['request'] = 'same'
    @server.on_client_connect
    def camera(client):
        client.camera.position = (.95, -1.9, 1.25); client.camera.look_at = (.4, -.3, .85)
    # Render the training MuJoCo geometry, including its mass-carrying racket.
    handles = []; m = env.model
    for i in range(m.ngeom):
        if m.geom_type[i] == 7 and m.geom_contype[i] == 0:
            mesh = int(m.geom_dataid[i]); va = int(m.mesh_vertadr[mesh]); vn = int(m.mesh_vertnum[mesh])
            fa = int(m.mesh_faceadr[mesh]); fn = int(m.mesh_facenum[mesh])
            vertices, faces = m.mesh_vert[va:va+vn].copy(), m.mesh_face[fa:fa+fn].copy()
        elif m.geom(i).name == 'training_face':
            mesh = trimesh.creation.cylinder(radius=m.geom_size[i, 0], height=2*m.geom_size[i, 1], sections=48)
            vertices, faces = mesh.vertices, mesh.faces
        elif m.geom(i).name == 'training_handle':
            mesh = trimesh.creation.cylinder(radius=m.geom_size[i, 0], height=2*m.geom_size[i, 1])
            vertices, faces = mesh.vertices, mesh.faces
        else:continue
        h = server.scene.add_mesh_simple(f'/robot/geom_{i}', vertices=vertices, faces=faces,
            color=tuple((m.geom_rgba[i, :3]*255).astype(int)))
        handles.append((i, h))
    ball = server.scene.add_icosphere('/ball', radius=CONFIG['ball_radius_m'], color=(255, 220, 40))
    target = server.scene.add_icosphere('/target', radius=.009, color=(30, 220, 120))
    trail = server.scene.add_line_segments('/ball_path', points=np.zeros((1, 2, 3)), colors=(255, 180, 25), line_width=3)
    rng = np.random.default_rng(4172); case = sample_cases(rng, 1)[0]
    obs = env.reset(case); active = False; finished = time.monotonic(); info = None
    path = [env.ball_position()]; count = 0; next_frame = time.monotonic(); selected = mode.value
    while True:
        with lock:request = state['request']; state['request'] = None
        if request:
            if request == 'new':case = sample_cases(rng, 1)[0]
            obs = env.reset(case); active = True; info = None; path = [env.ball_position()]; selected = mode.value
        if active and not pause.value:
            action = (np.zeros(64) if selected == 'Constant token' else
                      predict(previous if selected == 'Previous reaching policy' else policy, obs))
            if selected == 'Previous reaching policy':action = action*.15/env.base.residual_scale
            obs, _, done, info = env.step(action); path.append(env.ball_position())
            if done:active = False; finished = time.monotonic()
        elif repeat.value and not pause.value and time.monotonic()-finished > 1.2:new_ball(None)
        if count % 2 == 0:
            with server.atomic():
                for i, h in handles:
                    h.position = env.data.geom_xpos[i]
                    q = Rotation.from_matrix(env.data.geom_xmat[i].reshape(3, 3)).as_quat(); h.wxyz = q[[3, 0, 1, 2]]
                ball.position = env.ball_position(); target.position = env.target
                if len(path) > 1:trail.points = np.stack((path[:-1], path[1:]), axis=1)
                else:trail.points = np.array([[path[0], path[0]]])
                result = ('RETURN' if info and info['hit'] else 'MISS') if info and not active else 'In flight'
                uv = np.asarray(case['offset_uv_m'])*100
                status.content = (f'<div style="height:190px">{"Paused" if pause.value else result}<br>'
                    f'Simulation time: {env.data.time:.2f} s<br>Target offset: {uv[0]:+.1f}, {uv[1]:+.1f} cm<br>'
                    f'Controller: {selected}<br>Face contact: {bool(info and info["face_contact"])}<br>'
                    f'Outward ball speed: {env.outgoing_speed:.2f} m/s<br>Hardware output: disabled</div>')
            server.flush()
        count += 1; next_frame += env.dt/speed.value
        time.sleep(max(0, next_frame-time.monotonic()))
        if time.monotonic()-next_frame > .1:next_frame = time.monotonic()

if __name__ == '__main__':main()
