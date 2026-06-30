import json

import pytest

from colmap.panorama.cli_compare import compare_reports, format_comparison_text, main


def sample_report(
    *,
    workspace: str,
    matcher: str,
    effective_pairing: str,
    elapsed_seconds: float,
    matching_seconds: float,
    mapping_seconds: float,
    matches: int,
    verified_pairs: int,
    planned_pairs: int | None,
    registered_images: int,
    points3D: int,
    reprojection_error: float,
) -> dict:
    options = {
        "matcher": matcher,
        "effective_pairing": effective_pairing,
    }
    if planned_pairs is not None:
        options["matching_pair_count"] = planned_pairs
    return {
        "workspace": workspace,
        "panorama_count": 20,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_seconds_by_stage": {
            "matching": matching_seconds,
            "incremental_mapping": mapping_seconds,
        },
        "options": options,
        "database": {
            "matches": matches,
            "two_view_geometries": verified_pairs,
        },
        "best_model": {
            "num_registered_images": registered_images,
            "num_points3D": points3D,
            "mean_track_length": 2.5,
            "mean_reprojection_error_px": reprojection_error,
        },
    }


def test_compare_reports_computes_runtime_and_quality_deltas() -> None:
    comparison = compare_reports(
        sample_report(
            workspace="seq",
            matcher="sequential",
            effective_pairing="sequential",
            elapsed_seconds=100.0,
            matching_seconds=40.0,
            mapping_seconds=50.0,
            matches=120,
            verified_pairs=100,
            planned_pairs=None,
            registered_images=18,
            points3D=1000,
            reprojection_error=0.7,
        ),
        sample_report(
            workspace="keyframes",
            matcher="keyframes",
            effective_pairing="sequential-vocabtree",
            elapsed_seconds=70.0,
            matching_seconds=10.0,
            mapping_seconds=55.0,
            matches=35,
            verified_pairs=30,
            planned_pairs=35,
            registered_images=19,
            points3D=900,
            reprojection_error=0.6,
        ),
    )

    assert comparison["metrics"]["total_elapsed"]["delta"] == -30.0
    assert comparison["metrics"]["total_elapsed"]["percent_change"] == pytest.approx(
        -30.0,
    )
    assert comparison["metrics"]["matching_elapsed"]["ratio"] == 0.25
    assert comparison["metrics"]["planned_pair_count"]["baseline"] is None
    assert comparison["metrics"]["registered_images"]["delta"] == 1
    assert comparison["metrics"]["mean_reprojection_error_px"]["delta"] == pytest.approx(
        -0.1,
    )


def test_format_comparison_text_includes_matcher_context() -> None:
    comparison = compare_reports(
        sample_report(
            workspace="seq",
            matcher="sequential",
            effective_pairing="sequential",
            elapsed_seconds=100.0,
            matching_seconds=40.0,
            mapping_seconds=50.0,
            matches=120,
            verified_pairs=100,
            planned_pairs=None,
            registered_images=18,
            points3D=1000,
            reprojection_error=0.7,
        ),
        sample_report(
            workspace="vocab",
            matcher="vocabtree",
            effective_pairing="vocabtree",
            elapsed_seconds=70.0,
            matching_seconds=10.0,
            mapping_seconds=55.0,
            matches=35,
            verified_pairs=30,
            planned_pairs=None,
            registered_images=18,
            points3D=950,
            reprojection_error=0.6,
        ),
    )

    text = format_comparison_text(comparison)

    assert "baseline: sequential (sequential) seq" in text
    assert "candidate: vocabtree (vocabtree) vocab" in text
    assert "matching_elapsed" in text
    assert "-75.0%" in text


def test_compare_cli_prints_json(tmp_path, capsys) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps(
            sample_report(
                workspace="seq",
                matcher="sequential",
                effective_pairing="sequential",
                elapsed_seconds=10.0,
                matching_seconds=4.0,
                mapping_seconds=5.0,
                matches=12,
                verified_pairs=10,
                planned_pairs=None,
                registered_images=8,
                points3D=100,
                reprojection_error=0.7,
            ),
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            sample_report(
                workspace="vocab",
                matcher="vocabtree",
                effective_pairing="vocabtree",
                elapsed_seconds=8.0,
                matching_seconds=2.0,
                mapping_seconds=5.0,
                matches=6,
                verified_pairs=5,
                planned_pairs=None,
                registered_images=8,
                points3D=90,
                reprojection_error=0.6,
            ),
        ),
        encoding="utf-8",
    )

    assert (
        main(["--baseline", str(baseline), "--candidate", str(candidate), "--json"])
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["metrics"]["total_elapsed"]["delta"] == -2.0
