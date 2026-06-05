from pathlib import Path

import numpy as np

import colmap.legacy_pano_visualize as legacy_visualize


class FakeReconstruction:
    stats: dict[str, tuple[int, int, int]] = {}

    def __init__(self, model_path: Path):
        self.model_path = model_path

    def num_reg_frames(self) -> int:
        return self.stats[self.model_path.name][0]

    def num_points3D(self) -> int:
        return self.stats[self.model_path.name][1]

    def num_reg_images(self) -> int:
        return self.stats[self.model_path.name][2]


class FakePycolmap:
    Reconstruction = FakeReconstruction


class FakePose:
    def __init__(self, center: list[float]):
        self.translation = np.asarray(center, dtype=np.float64)

    def inverse(self):
        return self


class FakeFrame:
    def __init__(self, center: list[float]):
        self.rig_from_world = FakePose(center)


class FakePointCloud:
    points = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    colors = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def write_model(path: Path) -> None:
    path.mkdir(parents=True)
    for filename in legacy_visualize.MODEL_FILES:
        (path / filename).write_bytes(b"model")


def test_discover_models_finds_numbered_legacy_models(tmp_path: Path) -> None:
    write_model(tmp_path / "sparse" / "0")
    write_model(tmp_path / "sparse" / "2")
    (tmp_path / "sparse" / "incomplete").mkdir()

    assert legacy_visualize.discover_models(tmp_path) == [
        tmp_path / "sparse" / "0",
        tmp_path / "sparse" / "2",
    ]


def test_select_model_prefers_registered_frames_then_points(
    tmp_path: Path, monkeypatch
) -> None:
    write_model(tmp_path / "sparse" / "0")
    write_model(tmp_path / "sparse" / "1")
    FakeReconstruction.stats = {
        "0": (10, 1000, 120),
        "1": (11, 500, 132),
    }
    monkeypatch.setattr(legacy_visualize, "require_pycolmap", lambda: FakePycolmap)

    selected_path, reconstruction = legacy_visualize.select_model(tmp_path)

    assert selected_path == tmp_path / "sparse" / "1"
    assert reconstruction.model_path == selected_path


def test_frame_centers_returns_one_center_per_panorama_frame() -> None:
    centers = legacy_visualize.frame_centers(
        [FakeFrame([1.0, 2.0, 3.0]), FakeFrame([4.0, 5.0, 6.0])]
    )

    np.testing.assert_allclose(centers, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_parse_args_defaults_to_pano2_workspace() -> None:
    args = legacy_visualize.parse_args([])

    assert args.workspace == legacy_visualize.DEFAULT_WORKSPACE
    assert args.model is None
    assert args.max_frames == 120


def test_save_html_map_writes_interactive_visualization(tmp_path: Path) -> None:
    output = tmp_path / "map.html"

    legacy_visualize.save_html_map(
        output,
        FakePointCloud(),
        np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        max_points=100,
        point_size=2.0,
    )

    assert output.stat().st_size > 0
    assert "panorama trajectory" in output.read_text(encoding="utf-8")
