import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from token_stream import TokenReplay, validate_token


class TokenStreamTests(unittest.TestCase):
    def test_rejects_target_vector_and_nonfinite_latent(self):
        for value in ([0, 0, 0], [0] * 63, [float('nan')] * 64,
                      [float('inf')] * 64, [[0] * 64]):
            with self.assertRaises(ValueError):
                validate_token(value)

    def test_replay_enforces_time_and_provenance(self):
        rows = [dict(simulation_only=True, phase='reach', trial_elapsed_s=t,
                     controller_index=i, token_state=[float(i)] * 64)
                for i, t in enumerate([.01, .03, .05])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tokens.jsonl'
            def save(data):
                path.write_text('\n'.join(map(json.dumps, data)))
            save(rows)
            replay = TokenReplay(path)
            np.testing.assert_array_equal(replay.at(.04), np.ones(64))
            for t in (-.01, .2, float('nan')):
                with self.assertRaises(ValueError): replay.at(t)
            for change in ({'trial_elapsed_s': .3}, {'trial_elapsed_s': .03},
                           {'simulation_only': False}, {'controller_index': 0}):
                save(rows[:-1] + [dict(rows[-1], **change)])
                with self.assertRaises(ValueError): TokenReplay(path)


if __name__ == '__main__':
    unittest.main()
