import pytest

import colmap.visualize_map as visualize_map


class FakeReconstruction:
    def __init__(self, model_path):
        self.model_path = model_path

    def summary(self) -> str:
        return "fake reconstruction summary"

    def num_points3D(self) -> int:
        return 42

    def num_reg_images(self) -> int:
        return 7


class FakePycolmap:
    Reconstruction = FakeReconstruction


def test_visualize_summary_only_does_not_require_open3d(tmp_path, monkeypatch, capsys):
    model_path = tmp_path / "sparse_best"
    model_path.mkdir()
    monkeypatch.setattr(visualize_map, "pycolmap", FakePycolmap)
    monkeypatch.setattr(visualize_map, "o3d", None)

    visualize_map.visualize(
        model_path,
        scale_report=None,
        save_image=None,
        camera_scale=0.0,
        max_cameras=120,
        min_track_length=3,
        max_reprojection_error=0.0,
        point_size=2.0,
        remove_outliers=True,
        summary_only=True,
    )

    output = capsys.readouterr().out
    assert "fake reconstruction summary" in output
    assert "sparse points: 42" in output
    assert "registered cameras: 7" in output
    assert "warning: raw monocular COLMAP model is not metric." in output


def test_visualize_requires_open3d_for_geometry(tmp_path, monkeypatch):
    model_path = tmp_path / "sparse_best"
    model_path.mkdir()
    monkeypatch.setattr(visualize_map, "pycolmap", FakePycolmap)
    monkeypatch.setattr(visualize_map, "o3d", None)

    with pytest.raises(RuntimeError, match="Open3D is required"):
        visualize_map.visualize(
            model_path,
            scale_report=None,
            save_image=None,
            camera_scale=0.0,
            max_cameras=120,
            min_track_length=3,
            max_reprojection_error=0.0,
            point_size=2.0,
            remove_outliers=True,
            summary_only=False,
        )
