"""Read-only Viser playback of MuJoCo ball-policy rollouts."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import viser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('rollouts', type=Path)
    parser.add_argument('--port', type=int, default=8091)
    args = parser.parse_args()
    run = json.loads(args.rollouts.read_text())
    episodes = run['episodes']
    if not episodes or not all(len(e['frames']) == 11 for e in episodes):
        raise ValueError('Expected recorded 0.5-second episodes with 11 frames each')

    server = viser.ViserServer(host='127.0.0.1', port=args.port)
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/floor', plane='xy', width=3, height=3, cell_size=.25)
    # Table is a visual reference only. The trained environment has no table collider.
    server.scene.add_box('/reference/table', color=(32, 75, 145),
                         dimensions=(1.37, 1.525, .025), opacity=.35,
                         position=(1.02, 0, .745))
    server.scene.add_line_segments('/reference/net',
                                   points=np.array([[[1.02, -.76, .76],
                                                     [1.02, .76, .76]]]),
                                   colors=(240, 240, 240), thickness=.008)
    server.gui.add_markdown('**G1 ball-policy demo — simulation only**\n\n'
                            'Blue: fixed-body G1 skeleton · Yellow: analytic ball · '
                            'Orange: right-hand tracking point · Green: policy hand target.\n\n'
                            '**No ball/racket collision, table physics, SONIC control, or real-robot commands.**')
    pause = server.gui.add_button('Pause')
    next_episode = server.gui.add_button('Next ball')
    status = server.gui.add_markdown('Loading…')
    ball = server.scene.add_icosphere('/ball', radius=.02, color=(250, 220, 30))
    hand = server.scene.add_icosphere('/hand', radius=.025, color=(255, 110, 30))
    target = server.scene.add_icosphere('/policy_target', radius=.02, color=(45, 235, 105))
    crossing = server.scene.add_icosphere('/crossing', radius=.016, color=(230, 60, 195))
    state = {'episode': 0, 'step': 0, 'paused': False}

    @pause.on_click
    def toggle_pause(_):
        state['paused'] = not state['paused']
        pause.label = 'Resume' if state['paused'] else 'Pause'

    @next_episode.on_click
    def advance(_):
        state['episode'] = (state['episode'] + 1) % len(episodes)
        state['step'] = 0

    @server.on_client_connect
    def setup_camera(client):
        client.camera.position = (1.1, -1.4, 1.32)
        client.camera.look_at = (.42, 0, .84)

    parents = np.asarray(run['parents'])
    names = run['body_names']
    links = [(i, int(parents[i])) for i in range(2, len(parents))
             if names[i] and names[int(parents[i])] and names[i] != 'world']
    last = time.monotonic()
    while True:
        episode = episodes[state['episode']]
        frame = episode['frames'][state['step']]
        positions = np.asarray(frame['body_positions_m'])
        segments = np.asarray([[positions[i], positions[p]] for i, p in links])
        server.scene.add_line_segments('/robot', points=segments,
                                       colors=(35, 130, 235), thickness=.013)
        ball.position = frame['ball_m']
        hand.position = frame['hand_m']
        target.position = frame['target_m']
        crossing.position = frame['crossing_m']
        trajectory = np.asarray([f['ball_m'] for f in episode['frames']])
        server.scene.add_line_segments('/ball_path',
                                       points=np.stack([trajectory[:-1], trajectory[1:]], axis=1),
                                       colors=(235, 195, 65), thickness=.003)
        hand_path = np.asarray([f['hand_m'] for f in episode['frames'][:state['step'] + 1]])
        if len(hand_path) > 1:
            server.scene.add_line_segments('/hand_path',
                                           points=np.stack([hand_path[:-1], hand_path[1:]], axis=1),
                                           colors=(255, 110, 30), thickness=.005)
        status.content = (f'**Ball {state["episode"] + 1}/{len(episodes)}** · '
                          f't = {frame["time_s"]:.2f} s · '
                          f'final wrist-to-ball proxy error = '
                          f'{episode["final_proxy_error_cm"]:.2f} cm\n\n'
                          'The final distance measures the orange point to ball center. '
                          'It is **not** a racket hit or return rate.')
        server.flush()
        now = time.monotonic()
        if not state['paused'] and now - last >= .05:
            if state['step'] < 10:
                state['step'] += 1
            else:
                time.sleep(.5)
                state['episode'] = (state['episode'] + 1) % len(episodes)
                state['step'] = 0
            last = time.monotonic()
        time.sleep(.03)


if __name__ == '__main__':
    main()
