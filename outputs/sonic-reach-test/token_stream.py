"""Validated SONIC latent packets and recorded-token replay; no sockets or motors."""
import json
from pathlib import Path

import numpy as np


def validate_token(value):
    token = np.asarray(value, dtype=np.float32)
    if token.shape != (64,) or not np.all(np.isfinite(token)):
        raise ValueError('SONIC requires exactly 64 finite latent values, not a 3D target.')
    return token


def token_packet(value, frame_index):
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message
    if not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError('A nonnegative integer frame index is required.')
    return pack_pose_message({
        'token_state': validate_token(value)[None, :],
        'frame_index': np.asarray([frame_index], dtype=np.int64),
    }, topic='pose', version=4)


class TokenReplay:
    """Only replays a contiguous, timed sequence recorded from this simulator."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        rows = [r for r in rows if r.get('phase') in ('reach', 'complete')]
        if len(rows) < 2 or not all(r.get('simulation_only') is True for r in rows):
            raise ValueError('A recorded simulation reach with at least two tokens is required.')
        self.times = np.asarray([r['trial_elapsed_s'] for r in rows], dtype=float)
        if (not np.all(np.isfinite(self.times)) or not 0 <= self.times[0] <= .1
                or np.any(np.diff(self.times) <= 0) or np.max(np.diff(self.times)) > .1):
            raise ValueError('Token timeline is missing, nonmonotonic or has a gap over 100 ms.')
        self.tokens = np.stack([validate_token(r['token_state']) for r in rows])
        indices = np.asarray([r['controller_index'] for r in rows], dtype=np.int64)
        if np.any(np.diff(indices) <= 0):
            raise ValueError('Recorded controller indices must advance.')

    def at(self, elapsed):
        if not np.isfinite(elapsed) or elapsed < 0 or elapsed > self.times[-1] + .1:
            raise ValueError('Token replay time is outside the recorded interval.')
        index = max(0, int(np.searchsorted(self.times, elapsed, side='right')) - 1)
        return self.tokens[index].copy()
