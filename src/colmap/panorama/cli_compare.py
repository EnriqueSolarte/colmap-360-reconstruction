"""Compare two panorama reconstruction reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from colmap.panorama.reports import format_elapsed_seconds


MetricPath = tuple[str, ...]


METRICS: tuple[tuple[str, MetricPath, str], ...] = (
    ("total_elapsed", ("elapsed_seconds",), "less_is_better"),
    ("matching_elapsed", ("elapsed_seconds_by_stage", "matching"), "less_is_better"),
    (
        "mapping_elapsed",
        ("elapsed_seconds_by_stage", "incremental_mapping"),
        "less_is_better",
    ),
    ("database_matches", ("database", "matches"), "less_is_better"),
    (
        "verified_pair_count",
        ("database", "two_view_geometries"),
        "context",
    ),
    (
        "planned_pair_count",
        ("options", "matching_pair_count"),
        "less_is_better",
    ),
    (
        "registered_images",
        ("best_model", "num_registered_images"),
        "more_is_better",
    ),
    ("points3D", ("best_model", "num_points3D"), "more_is_better"),
    (
        "mean_track_length",
        ("best_model", "mean_track_length"),
        "more_is_better",
    ),
    (
        "mean_reprojection_error_px",
        ("best_model", "mean_reprojection_error_px"),
        "less_is_better",
    ),
)


def read_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(data: dict[str, Any], path: MetricPath) -> int | float | None:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def metric_delta(
    baseline_value: int | float | None,
    candidate_value: int | float | None,
) -> dict[str, float | int | None]:
    if baseline_value is None or candidate_value is None:
        return {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": None,
            "ratio": None,
            "percent_change": None,
        }
    delta = candidate_value - baseline_value
    ratio = None if baseline_value == 0 else candidate_value / baseline_value
    percent_change = None if ratio is None else (ratio - 1.0) * 100.0
    return {
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": delta,
        "ratio": ratio,
        "percent_change": percent_change,
    }


def compare_reports(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
) -> dict[str, Any]:
    metrics = {}
    for name, path, interpretation in METRICS:
        metrics[name] = {
            **metric_delta(
                nested_get(baseline_report, path),
                nested_get(candidate_report, path),
            ),
            "interpretation": interpretation,
        }
    return {
        "baseline": run_summary(baseline_report),
        "candidate": run_summary(candidate_report),
        "metrics": metrics,
    }


def run_summary(report: dict[str, Any]) -> dict[str, Any]:
    options = report.get("options", {})
    best_model = report.get("best_model", {})
    return {
        "workspace": report.get("workspace"),
        "matcher": options.get("matcher"),
        "effective_pairing": options.get("effective_pairing"),
        "panorama_count": report.get("panorama_count"),
        "elapsed_seconds": report.get("elapsed_seconds"),
        "registered_images": best_model.get("num_registered_images"),
        "points3D": best_model.get("num_points3D"),
    }


def format_value(value: float | int | None, *, is_elapsed: bool = False) -> str:
    if value is None:
        return "-"
    if is_elapsed:
        return format_elapsed_seconds(float(value))
    if isinstance(value, int):
        return str(value)
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3g}"


def format_percent(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.1f}%"


def format_ratio(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3g}x"


def format_comparison_text(comparison: dict[str, Any]) -> str:
    baseline = comparison["baseline"]
    candidate = comparison["candidate"]
    lines = [
        "Panorama reconstruction comparison",
        (
            f"baseline: {baseline.get('matcher')} "
            f"({baseline.get('effective_pairing')}) "
            f"{baseline.get('workspace')}"
        ),
        (
            f"candidate: {candidate.get('matcher')} "
            f"({candidate.get('effective_pairing')}) "
            f"{candidate.get('workspace')}"
        ),
        "",
        (
            f"{'metric':31} {'baseline':>12} {'candidate':>12} "
            f"{'delta':>12} {'change':>10} {'ratio':>8}"
        ),
        "-" * 91,
    ]
    elapsed_metrics = {"total_elapsed", "matching_elapsed", "mapping_elapsed"}
    for name, metric in comparison["metrics"].items():
        is_elapsed = name in elapsed_metrics
        lines.append(
            f"{name:31} "
            f"{format_value(metric['baseline'], is_elapsed=is_elapsed):>12} "
            f"{format_value(metric['candidate'], is_elapsed=is_elapsed):>12} "
            f"{format_value(metric['delta'], is_elapsed=is_elapsed):>12} "
            f"{format_percent(metric['percent_change']):>10} "
            f"{format_ratio(metric['ratio']):>8}"
        )
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two panorama_equirectangular_report.json files. Use a "
            "sequential run as --baseline and a vocabtree or keyframe run as "
            "--candidate to estimate runtime and reconstruction-quality gain."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Baseline panorama_equirectangular_report.json path.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Candidate panorama_equirectangular_report.json path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable comparison JSON.",
    )
    args = parser.parse_args(argv)
    if not args.baseline.is_file():
        parser.error(f"--baseline does not exist: {args.baseline}")
    if not args.candidate.is_file():
        parser.error(f"--candidate does not exist: {args.candidate}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    comparison = compare_reports(read_report(args.baseline), read_report(args.candidate))
    if args.json:
        print(json.dumps(comparison, indent=2))
    else:
        print(format_comparison_text(comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
