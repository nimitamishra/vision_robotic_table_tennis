import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmark import (
    CAMERA_REFERENCE, DEFAULT_SCENE, Frame, References, Run, Score, evaluate,
    fit_variance_scale, load_run, main, marginal_metrics, replay, synthetic_runs,
)
from tracker import Tracker


TABLE = json.loads(DEFAULT_SCENE.read_text())["world"]["table"]


class ReferenceTests(unittest.TestCase):
    def test_exact_reference_and_future_only_brackets(self):
        samples = [(10, np.zeros(3)), (30, np.array([2e-7, 0., 0.]))]
        ref = References(samples, max_gap_ns=20)
        np.testing.assert_allclose(ref.at(0, 20), [1e-7, 0., 0.])
        np.testing.assert_array_equal(ref.at(10, 30), samples[1][1])
        for issued, target in [(10, 20), (30, 30), (0, 8), (0, 31)]:
            self.assertIsNone(ref.at(issued, target))
        self.assertIsNone(References(samples, 19).at(0, 20))
        self.assertIsNone(References([(10, np.zeros(3)), (30, np.ones(3))], 20).at(0, 20))

    def test_recorded_packet_schema_preserves_missing_samples_and_stereo_priors(self):
        contour = dict(xyz_left=[0, 0, 3], left_center_px=[100, 50], right_center_px=[90, 50])
        rows = [
            dict(capture_monotonic_ns=1, world_measurement=True, ball_refined=contour,
                 xyz_world_refined=[.1, .2, .3], ball={}, xyz_world=[1, 2, 3]),
            dict(capture_monotonic_ns=2, world_measurement=False, ball={}, xyz_world=[1, 2, 3]),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "measurements.jsonl"
            path.write_text("\n".join(map(json.dumps, rows)))
            run = load_run(path)
            self.assertEqual(len(run.frames), 2)
            self.assertEqual(len(run.reference), 1)
            self.assertEqual(run.frames[0].source, "contour")
            self.assertGreater(run.frames[0].range_sigma, .024)
            np.testing.assert_allclose(run.reference[0][1], [.1, .2, .3])
            self.assertIsNone(run.frames[1].position)
            path.write_text("\n".join(map(json.dumps, reversed(rows))))
            with self.assertRaisesRegex(ValueError, "measurements.jsonl:2:.*strictly increasing"):
                load_run(path)

    def test_recorded_demo_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "measurements.jsonl"
            path.with_name("config.json").write_text('{"demo": true}')
            with self.assertRaisesRegex(ValueError, "recorded demo"):
                load_run(path)


class EvaluationTests(unittest.TestCase):
    def test_tracker_is_called_once_per_frame_in_order_and_sees_only_inputs(self):
        run = next(synthetic_runs(3, count=1))
        calls = []

        def record_step(tracker, t, p, source, range_sigma_m):
            calls.append((t, p, source, range_sigma_m))
            return original_step(tracker, t, p, source, range_sigma_m)

        original_step = Tracker.step
        with patch.object(Tracker, "step", record_step):
            scores, counts, _ = replay(run, TABLE, {}, 40_000_000)
        self.assertEqual(len(calls), len(run.frames))
        for (t, p, source, sigma), frame in zip(calls, run.frames):
            self.assertEqual(t, frame.ns / 1e9)
            self.assertIs(p, frame.position)
            self.assertEqual((source, sigma), (frame.source, frame.range_sigma))
        for horizon, count in counts.items():
            self.assertGreater(count["scored"], 0)
            self.assertEqual(count["scored"], len(scores[horizon]))
            self.assertLessEqual(count["scored"], min(count["reference"], count["forecast"]))

    def test_marginal_coverage_and_equal_weight_variance_fit(self):
        scores = [Score(np.full(3, z), np.ones(3), None) for z in (1., 3.)]
        self.assertEqual(fit_variance_scale([{100: [scores[0]] * 3}, {100: [scores[1]]}], 100), 5.)
        self.assertIsNone(fit_variance_scale([{100: scores}], 100))
        raw = marginal_metrics(scores, 1.)
        scaled = marginal_metrics(scores, 5.)
        self.assertEqual(raw["central_coverage_xyz"]["0.9"]["x"], .5)
        self.assertEqual(scaled["central_coverage_xyz"]["0.9"]["x"], 1.)
        self.assertLess(scaled["mean_marginal_gaussian_nll"], raw["mean_marginal_gaussian_nll"])
        self.assertIsNone(marginal_metrics([Score(np.ones(3), None, None)], 1.))

    def test_evaluation_reference_cannot_change_fitted_scales(self):
        runs = list(synthetic_runs(7, count=4))
        normal = evaluate(runs[:2], [runs[2]], TABLE, {}, "synthetic")
        shifted = Run(runs[2].name, runs[2].frames,
                      [(ns, p + .3) for ns, p in runs[2].reference])
        changed = evaluate(runs[:2], [shifted], TABLE, {}, "synthetic")
        self.assertEqual(normal["variance_scale_by_horizon_ms"], changed["variance_scale_by_horizon_ms"])
        self.assertNotEqual(normal["fixed_horizon"]["100"]["point"], changed["fixed_horizon"]["100"]["point"])
        duplicate = Run("different-name", runs[0].frames, runs[0].reference)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate(runs[:2], [duplicate], TABLE, {}, "synthetic")
        overlap = Run("overlapping-clip", runs[0].frames[5:], runs[0].reference[5:])
        with self.assertRaisesRegex(ValueError, "overlapping"):
            evaluate(runs[:2], [overlap], TABLE, {}, "synthetic")

    def test_missing_reference_and_plane_are_reported_with_denominators(self):
        run = Run("empty-attempt", [Frame(0, None, "unavailable")], [])
        report = evaluate([], [run], TABLE, {}, "camera", plane_y=.2)
        row = report["fixed_horizon"]["300"]
        self.assertEqual(report["reference"], CAMERA_REFERENCE)
        self.assertEqual((row["frames"], row["forecast"], row["reference"], row["scored"]), (1, 0, 0, 0))
        self.assertIsNone(row["point"])
        self.assertIsNone(row["uncertainty_raw"])
        self.assertEqual(report["virtual_plane"]["attempts"], 1)
        self.assertEqual(report["virtual_plane"]["status_counts"], {"no_valid_3d": 1})

    def test_crossing_selection_is_causal_at_all_requested_leads(self):
        run = next(synthetic_runs(23, count=1))
        _, _, crossing = replay(run, TABLE, {}, 40_000_000, plane_y=.2)
        self.assertEqual(crossing["status"], "observed_crossing")
        for row in crossing["lead_metrics"]:
            self.assertEqual(row["status"], "evaluated")
            self.assertGreaterEqual(row["actual_lead_s"] + 1e-12, row["requested_lead_s"])
            self.assertLessEqual(row["actual_lead_s"], row["requested_lead_s"] + .05 + 1e-12)

    def test_virtual_plane_replays_a_rebound(self):
        bounce = list(synthetic_runs(23, count=2))[1]
        heights = [p[2] for _, p in bounce.reference]
        impact_index = int(np.argmin(heights))
        self.assertGreater(heights[impact_index + 5], heights[impact_index])
        _, _, crossing = replay(bounce, TABLE, {}, 40_000_000, plane_y=.2)
        self.assertEqual(crossing["status"], "observed_crossing")
        self.assertGreater(crossing["reference_time_s"], bounce.reference[impact_index][0] / 1e9)
        self.assertEqual(crossing["lead_metrics"][-1]["status"], "evaluated")

    def test_cli_synthetic_reproducibility_and_logged_packet_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "report.json"
            args = ["--synthetic", "--output", str(output)]
            main(args)
            first = output.read_bytes()
            main(args)
            self.assertEqual(output.read_bytes(), first)
            report = json.loads(first)
            self.assertEqual(report["reference_kind"], "synthetic")
            self.assertEqual(len(report["calibration_runs"]), 6)
            self.assertEqual(report["virtual_plane"]["observed_in_window"], 6)
            for row in report["fixed_horizon"].values():
                paired = row["paired_persistence"]
                self.assertGreater(paired["n"], 0)
                self.assertLess(paired["tracker"]["rmse_m"], paired["persistence"]["rmse_m"])
                self.assertIsNotNone(row["uncertainty_scaled"])

            packet_log = Path(folder) / "measurements.jsonl"
            run = next(synthetic_runs(3, count=1))
            packets = [
                dict(capture_monotonic_ns=frame.ns, world_measurement=frame.position is not None,
                     ball={}, xyz_world=None if frame.position is None else frame.position.tolist())
                for frame in run.frames
            ]
            packet_log.write_text("\n".join(map(json.dumps, packets)))
            main(["--eval", str(packet_log), "--output", str(output)])
            report = json.loads(output.read_bytes())
            self.assertEqual(report["reference_kind"], "camera")
            self.assertGreater(report["fixed_horizon"]["100"]["scored"], 0)
            self.assertIsNone(report["fixed_horizon"]["100"]["uncertainty_scaled"])


if __name__ == "__main__":
    unittest.main()
