"""Offline replay and held-out evaluation of the single-ball tracker."""
import argparse
from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

from collision import forecast as collision_forecast
from intercept import Trial
from tracker import Tracker, measurement, stereo_range_sigma

HORIZONS_MS = (100, 200, 300)
LEVELS = (0.50, 0.80, 0.90, 0.95)
Z_CUTOFFS = {str(level): NormalDist().inv_cdf((1 + level) / 2) for level in LEVELS}
NS_PER_SECOND = 1_000_000_000
AXES = ("x", "y", "z")
CAMERA_REFERENCE = "later raw stereo observations, interpolated where needed"
SYNTHETIC_REFERENCE = "separately generated, noiseless synthetic positions"
DEFAULT_SCENE = Path(__file__).resolve().parents[2] / "vision/g1/calibration.json"


@dataclass
class Frame:
    ns: int
    position: np.ndarray | None
    source: str
    range_sigma: float | None = None


@dataclass
class Run:
    name: str
    frames: list[Frame]
    reference: list[tuple[int, np.ndarray]]

    def fingerprint(self):
        """Identify identical tracker inputs even if file names or JSON formatting differ."""
        first_ns = self.frames[0].ns
        rows = [
            [f.ns - first_ns, None if f.position is None else f.position.tolist(),
             f.source, f.range_sigma]
            for f in self.frames
        ]
        return hashlib.sha256(json.dumps(rows, allow_nan=False).encode()).hexdigest()


@dataclass
class Score:
    error: np.ndarray
    sigma: np.ndarray | None
    persistence_error: np.ndarray | None


def load_run(path):
    """Read the slim measurements.jsonl emitted by run.py in capture order."""
    config_path = path.with_name("config.json")
    if config_path.is_file() and json.loads(config_path.read_text()).get("demo"):
        raise ValueError(f"{path}: recorded demo data cannot be scored as a camera recording")
    frames = []
    with path.open() as stream:
        for line, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            try:
                packet = json.loads(raw)
                if not isinstance(packet, dict):
                    raise ValueError("expected a measurement object")
                ns = packet.get("capture_monotonic_ns")
                if type(ns) is not int or ns < 0 or (frames and ns <= frames[-1].ns):
                    raise ValueError("expected strictly increasing integer capture_monotonic_ns")
                position, source = measurement(packet)
                sigma = stereo_range_sigma(packet, source) if position is not None else None
                frames.append(Frame(ns, position, source, sigma))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line}: {exc}") from exc
    if not frames:
        raise ValueError(f"{path}: no measurement packets")
    reference = [(f.ns, f.position) for f in frames if f.position is not None]
    return Run(str(path), frames, reference)


def synthetic_runs(seed, count=12):
    """Separate noisy tracker input and exact fixed-horizon truth for each throw."""
    for index, sequence in enumerate(np.random.SeedSequence(seed).spawn(count)):
        rng = np.random.default_rng(sequence)
        bounce = index % 2 == 1
        height = rng.uniform(.36, .44) if bounce else rng.uniform(.9, .98)
        position = np.array([rng.uniform(-.25, .25), 1.85, height])
        velocity = np.array([rng.uniform(-.3, .3), rng.uniform(-3.5, -2.9), .6 if bounce else 2.45])
        gravity = np.array([0., 0., -9.81])
        if bounce:
            impact_s = (velocity[2] + np.sqrt(velocity[2]**2 + 2 * 9.81 * (height - .02))) / 9.81
            impact = position + velocity * impact_s + .5 * gravity * impact_s**2
            retention = rng.uniform(.87, .95)
            restitution = rng.uniform(.78, .88)
            rebound = (velocity + gravity * impact_s) * [retention, retention, -restitution]
        frames, reference = [], []
        for step in range(38):
            t = step * .02
            ns = step * 20_000_000
            if bounce and t >= impact_s:
                dt = t - impact_s
                truth = impact + rebound * dt + .5 * gravity * dt**2
            else:
                truth = position + velocity * t + .5 * gravity * t**2
            reference.append((ns, truth))
            noisy = truth + rng.normal(0., [.008, .024, .012])
            if rng.random() < .12:
                frames.append(Frame(ns, None, "unavailable"))
            else:
                frames.append(Frame(ns, noisy, "contour", .024))
        scenario = "bounce" if bounce else "flight"
        yield Run(f"synthetic-{scenario}-{index + 1:02d}", frames, reference)


class References:
    def __init__(self, samples, max_gap_ns):
        self.times = [ns for ns, _ in samples]
        self.points = [point for _, point in samples]
        self.max_gap_ns = max_gap_ns

    def at(self, issued_ns, target_ns):
        """An interpolant may only use samples strictly after issuance."""
        if target_ns <= issued_ns:
            return None
        right = bisect_left(self.times, target_ns)
        if right == len(self.times):
            return None
        after = self.times[right]
        if after == target_ns:
            return self.points[right]
        if right == 0:
            return None
        before = self.times[right - 1]
        gap = after - before
        if before <= issued_ns or gap > self.max_gap_ns:
            return None
        p, q = self.points[right - 1], self.points[right]
        if np.linalg.norm(q - p) > 25 * (gap / NS_PER_SECOND):
            return None
        return p + (q - p) * ((target_ns - before) / gap)


def replay(run, table, tracker_options, max_reference_gap_ns, plane_y=None):
    half_width = table["width_m"] / 2
    near = -table["tag_center_inset_m"]
    bounds = (-half_width, half_width, near, near + table["length_m"])
    tracker = Tracker(table_bounds=bounds, **tracker_options)
    references = References(run.reference, max_reference_gap_ns)
    exact_references = dict(run.reference) if plane_y is not None else {}
    score = {h: [] for h in HORIZONS_MS}
    counts = {h: dict(frames=0, forecast=0, reference=0, scored=0) for h in HORIZONS_MS}
    trial = None if plane_y is None else Trial(1, plane_y, table["width_m"],
                                               tracker_options.get("table_z", 0.))
    origin = run.frames[0].ns
    crossing = None
    for frame in run.frames:
        t = (frame.ns - origin) / NS_PER_SECOND
        result = tracker.step(t, frame.position, frame.source, range_sigma_m=frame.range_sigma)
        predictions = {round(p["horizon_s"] * 1000): p for p in result["predictions"]}
        for h in HORIZONS_MS:
            ref = references.at(frame.ns, frame.ns + h * 1_000_000)
            prediction = predictions.get(h)
            count = counts[h]
            count["frames"] += 1
            count["forecast"] += prediction is not None
            count["reference"] += ref is not None
            if prediction is None or ref is None:
                continue
            count["scored"] += 1
            sigma = np.asarray(prediction["position_std_m"], dtype=float)
            if not np.all(np.isfinite(sigma) & (sigma > 0)):
                sigma = None
            persistence_error = None if frame.position is None else frame.position - ref
            score[h].append(Score(np.asarray(prediction["position_m"]) - ref, sigma, persistence_error))
        if trial is not None and not trial.done:
            if result.get("velocity_m_s") is not None and tracker.mode != "stationary":
                result.update(collision_forecast(
                    result["position_m"], result["velocity_m_s"], table, horizon=tracker.horizon,
                    table_z=tracker_options.get("table_z", 0.), restitution=tracker.restitution,
                    tangent_retention=tracker.tangent_retention,
                ))
            crossing = trial.observe(t, exact_references.get(frame.ns), result)
    if trial is not None and crossing is None:
        crossing = trial.finish()
    return score, counts, crossing


def point_metrics(errors):
    if not len(errors):
        return None
    values = np.asarray(errors)
    distance = np.linalg.norm(values, axis=1)
    return {
        "mean_distance_m": float(np.mean(distance)),
        "rmse_m": float(np.sqrt(np.mean(distance**2))),
        "p90_distance_m": float(np.quantile(distance, .90)),
        "bias_xyz_m": dict(zip(AXES, np.mean(values, axis=0).tolist())),
    }


def marginal_metrics(scores, variance_scale):
    """Use only marginal sigmas published by the tracker."""
    valid = [s for s in scores if s.sigma is not None]
    if not valid or variance_scale is None:
        return None
    error = np.asarray([s.error for s in valid])
    sigma = np.asarray([s.sigma for s in valid]) * np.sqrt(variance_scale)
    z = error / sigma
    return {
        "n": len(valid),
        "z_mean_xyz": dict(zip(AXES, np.mean(z, axis=0).tolist())),
        "z_rms_xyz": dict(zip(AXES, np.sqrt(np.mean(z**2, axis=0)).tolist())),
        "mean_marginal_gaussian_nll": float(np.mean(.5 * z**2 + np.log(sigma) + .5 * np.log(2 * np.pi))),
        "central_coverage_xyz": {
            level: dict(zip(AXES, np.mean(np.abs(z) <= cutoff, axis=0).tolist()))
            for level, cutoff in Z_CUTOFFS.items()
        },
    }


def fit_variance_scale(run_scores, h):
    valid_runs = [[s for s in scores[h] if s.sigma is not None] for scores in run_scores]
    valid_runs = [scores for scores in valid_runs if scores]
    if len(valid_runs) < 2:
        return None
    # Equal-weight runs, with one multiplicative variance parameter per horizon.
    scale = float(np.mean([
        np.mean([(s.error / s.sigma)**2 for s in scores]) for scores in valid_runs
    ]))
    return scale if np.isfinite(scale) and scale > 0 else None


def summarize(scores, counts, scales):
    result = {}
    for h in HORIZONS_MS:
        paired = [s for s in scores[h] if s.persistence_error is not None]
        result[str(h)] = {
            **counts[h],
            "point": point_metrics([s.error for s in scores[h]]),
            "paired_persistence": {
                "n": len(paired),
                "tracker": point_metrics([s.error for s in paired]),
                "persistence": point_metrics([s.persistence_error for s in paired]),
            },
            "uncertainty_raw": marginal_metrics(scores[h], 1.),
            "uncertainty_scaled": marginal_metrics(scores[h], scales[h]),
        }
    return result


def evaluate(calibration, evaluation, table, tracker_options, reference_kind, max_gap_ms=40, plane_y=None):
    if not evaluation:
        raise ValueError("at least one evaluation recording is required")
    runs = [*calibration, *evaluation]
    fingerprints = {run.name: run.fingerprint() for run in runs}
    if len(fingerprints) != len(runs) or len(set(fingerprints.values())) != len(runs):
        raise ValueError("duplicate tracker inputs across the benchmark recordings")
    seen = set()
    for run in runs:
        samples = {(f.ns, *f.position) for f in run.frames if f.position is not None}
        if seen.intersection(samples):
            raise ValueError(f"{run.name}: overlapping captured observations across benchmark recordings")
        seen.update(samples)

    def run_all(runs):
        return [replay(run, table, tracker_options, max_gap_ms * 1_000_000, plane_y) for run in runs]

    fit = run_all(calibration)
    evaluated = run_all(evaluation)
    scales = {h: fit_variance_scale([s for s, _, _ in fit], h) for h in HORIZONS_MS}
    reference = SYNTHETIC_REFERENCE if reference_kind == "synthetic" else CAMERA_REFERENCE
    report = {
        "schema_version": 1,
        "reference_kind": reference_kind,
        "reference": reference,
        "tracker_settings": tracker_options,
        "table": table,
        "reference_max_interpolation_gap_ms": max_gap_ms,
        "calibration_runs": [run.name for run in calibration],
        "input_fingerprints": fingerprints,
        "variance_scale_by_horizon_ms": {str(h): scale for h, scale in scales.items()},
        "evaluation_runs": [],
        "fixed_horizon": summarize(
            {h: [row for s, _, _ in evaluated for row in s[h]] for h in HORIZONS_MS},
            {h: {k: sum(c[h][k] for _, c, _ in evaluated) for k in evaluated[0][1][h]}
             for h in HORIZONS_MS},
            scales,
        ),
    }
    for run, (scores, counts, crossing) in zip(evaluation, evaluated):
        if crossing is not None:
            crossing["reference"] = f"{reference}, linearly interpolated at the virtual plane"
        report["evaluation_runs"].append({
            "input": run.name,
            "fixed_horizon": summarize(scores, counts, scales),
            "plane_crossing": crossing,
        })
    if plane_y is not None:
        crossings = [r["plane_crossing"] for r in report["evaluation_runs"]]
        observed = [r for r in crossings if r["status"] == "observed_crossing"]
        lead_metrics = {}
        for h in HORIZONS_MS:
            rows = [m for r in observed for m in r["lead_metrics"]
                    if round(m["requested_lead_s"] * 1000) == h and m["status"] == "evaluated"]
            lead_metrics[str(h)] = {
                "scored": len(rows),
                "mean_position_error_m": float(np.mean([m["position_error_m"] for m in rows])) if rows else None,
                "mean_absolute_time_error_s": float(np.mean([abs(m["time_error_s"]) for m in rows])) if rows else None,
                "mean_signed_time_error_s": float(np.mean([m["time_error_s"] for m in rows])) if rows else None,
            }
        report["virtual_plane"] = {
            "y_m": plane_y,
            "attempts": len(crossings),
            "observed_in_window": len(observed),
            "status_counts": {status: sum(r["status"] == status for r in crossings)
                              for status in sorted({r["status"] for r in crossings})},
            "by_requested_lead_ms": lead_metrics,
        }
    return report


def plot_report(report, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 4), layout="constrained")
    FigureCanvasAgg(figure)
    error, coverage = figure.subplots(1, 2)
    rows = [report["fixed_horizon"][str(h)] for h in HORIZONS_MS]
    for label in ("tracker", "persistence"):
        values = [row["paired_persistence"][label] for row in rows]
        error.plot(HORIZONS_MS, [100 * m["rmse_m"] if m else np.nan for m in values],
                   marker="o", label=label.capitalize())
    error.legend()
    error.set(xlabel="Forecast horizon (ms)", ylabel="Paired position RMSE (cm)", xticks=HORIZONS_MS)
    coverage.plot(LEVELS, LEVELS, "k--", label="Nominal")
    styles = (("o", "tab:blue"), ("s", "tab:orange"), ("^", "tab:green"))
    for h, row, (marker, color) in zip(HORIZONS_MS, rows, styles):
        for key, label, linestyle in (("uncertainty_raw", "raw", ":"), ("uncertainty_scaled", "scaled", "-")):
            metrics = row[key]
            if metrics is not None:
                empirical = [np.mean(list(metrics["central_coverage_xyz"][str(level)].values()))
                             for level in LEVELS]
                coverage.plot(LEVELS, empirical, color=color, marker=marker,
                              linestyle=linestyle, label=f"{h} ms {label}")
    coverage.legend(fontsize="small")
    coverage.set(xlabel="Nominal coverage", ylabel="Mean marginal coverage", ylim=(0, 1.02))
    figure.suptitle(report["reference"].capitalize())
    figure.savefig(path, dpi=160)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, action="append", default=[], help="Held-out measurements.jsonl (repeatable)")
    parser.add_argument("--calibrate", type=Path, action="append", default=[], help="Calibration measurements.jsonl (repeatable)")
    parser.add_argument("--synthetic", action="store_true", help="Run a seeded example with separate noiseless truth")
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="G1 Studio scene-calibration.json")
    parser.add_argument("--table-z", type=float, default=0.)
    parser.add_argument("--restitution", type=float, default=.8)
    parser.add_argument("--tangent-retention", type=float, default=.9)
    parser.add_argument("--contour-sigma", type=float, default=.012)
    parser.add_argument("--box-sigma", type=float, default=.025)
    parser.add_argument("--max-gap", type=float, default=.2, help="Tracker coasting/reset gap in seconds")
    parser.add_argument("--reference-max-gap-ms", type=int, default=40)
    parser.add_argument("--plane-y", type=float, help="Optional incoming virtual plane. Each file must contain one attempt.")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--plot", type=Path, help="Optional PNG plot (requires Matplotlib)")
    args = parser.parse_args(argv)
    if args.synthetic == bool(args.eval) or (args.synthetic and args.calibrate):
        parser.error("choose --synthetic or one or more --eval recordings")
    if len(args.calibrate) == 1:
        parser.error("variance scaling requires at least two separate calibration runs")
    numeric = [args.table_z, args.max_gap, args.contour_sigma, args.box_sigma]
    if not np.isfinite(numeric).all() or min(numeric[1:]) <= 0 or args.reference_max_gap_ms <= 0:
        parser.error("use finite settings and positive gaps and sigmas")
    if not (0 <= args.restitution <= 1 and 0 <= args.tangent_retention <= 1):
        parser.error("impact coefficients must be between 0 and 1")
    if args.plane_y is not None and not np.isfinite(args.plane_y):
        parser.error("plane-y must be finite")
    inputs = {p.resolve() for p in [args.scene, *args.eval, *args.calibrate]}
    outputs = [p.resolve() for p in (args.output, args.plot) if p is not None]
    if inputs.intersection(outputs) or len(set(outputs)) != len(outputs):
        parser.error("report and plot paths must be distinct from one another and from inputs")
    try:
        table = json.loads(args.scene.read_text())["world"]["table"]
        options = {k: getattr(args, k) for k in (
            "table_z", "restitution", "tangent_retention", "contour_sigma", "box_sigma", "max_gap",
        )}
        options["horizon"] = .5
        plane_y = args.plane_y
        if args.synthetic:
            runs = list(synthetic_runs(args.seed))
            calibration, evaluation = runs[:6], runs[6:]
            if plane_y is None:
                plane_y = .20 - table["tag_center_inset_m"]
        else:
            calibration, evaluation = [load_run(p) for p in args.calibrate], [load_run(p) for p in args.eval]
        report = evaluate(calibration, evaluation, table, options, "synthetic" if args.synthetic else "camera",
                          args.reference_max_gap_ms, plane_y)
        if args.synthetic:
            report["seed"] = args.seed
        if args.plot:
            plot_report(report, args.plot)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (ImportError, OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
