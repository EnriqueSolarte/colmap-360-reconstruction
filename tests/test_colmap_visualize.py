import pytest
import numpy as np

import colmap.visualize_map as visualize_map


class FakeTrack:
    def __init__(self, length):
        self._length = length

    def length(self):
        return self._length


class FakePoint3D:
    def __init__(self, track_length):
        self.track = FakeTrack(track_length)
        self.error = 0.0
        self.xyz = [1.0, 2.0, 3.0]
        self.color = [255, 128, 0]


class FakeImage:
    def __init__(self, name, center, rotation=None):
        self.name = name
        self._center = center
        self._rotation = np.eye(3) if rotation is None else np.asarray(rotation)

    def projection_center(self):
        return self._center

    def cam_from_world(self):
        return FakePose(self._rotation)


class FakePose:
    def __init__(self, rotation):
        self.rotation = FakeRotation(rotation)


class FakeRotation:
    def __init__(self, rotation):
        self._rotation = rotation

    def matrix(self):
        return self._rotation


class FakePointCloud:
    def __init__(self):
        self.points = []
        self.colors = []


class FakeOpen3D:
    class geometry:
        PointCloud = FakePointCloud

    class utility:
        @staticmethod
        def Vector3dVector(values):
            return list(values)


class FakeGeometry:
    def __init__(self, kind, **metadata):
        self.kind = kind
        self.metadata = metadata
        self.transforms = []

    def paint_uniform_color(self, color):
        self.metadata["color"] = color

    def compute_vertex_normals(self):
        self.metadata["normals_computed"] = True

    def transform(self, transform):
        self.transforms.append(np.asarray(transform, dtype=np.float64))


class FakeTriangleMesh:
    @staticmethod
    def create_sphere(radius, resolution):
        return FakeGeometry("sphere", radius=radius, resolution=resolution)

    @staticmethod
    def create_coordinate_frame(size):
        return FakeGeometry("coordinate_frame", size=size)


class FakeOpen3DGeometry:
    class geometry:
        TriangleMesh = FakeTriangleMesh


class FakeReconstruction:
    def __init__(self, model_path):
        self.model_path = model_path
        self.points3D = {}

    def summary(self) -> str:
        return "fake reconstruction summary"

    def num_points3D(self) -> int:
        return 42

    def num_reg_images(self) -> int:
        return 7


class FakePycolmap:
    Reconstruction = FakeReconstruction


class FakeCamera:
    def __init__(self, model_name, width=2048, height=1024, params=()):
        self.model_name = model_name
        self.width = width
        self.height = height
        self.params = params


def test_point_cloud_can_include_untracked_points(monkeypatch):
    reconstruction = FakeReconstruction("model")
    reconstruction.points3D = {1: FakePoint3D(track_length=0)}
    monkeypatch.setattr(visualize_map, "o3d", FakeOpen3D)

    filtered_cloud = visualize_map.point_cloud_from_reconstruction(
        reconstruction,
        min_track_length=3,
        max_reprojection_error=0.0,
        remove_outliers=False,
    )
    untracked_cloud = visualize_map.point_cloud_from_reconstruction(
        reconstruction,
        min_track_length=0,
        max_reprojection_error=0.0,
        remove_outliers=False,
    )

    assert len(filtered_cloud.points) == 0
    assert len(untracked_cloud.points) == 1


def test_show_untracked_points_sets_track_length_zero():
    args = visualize_map.parse_args(["--show-untracked-points"])

    assert args.min_track_length == 0


def test_camera_style_defaults_to_auto():
    args = visualize_map.parse_args([])

    assert args.camera_style == "auto"


def test_camera_style_accepts_sphere():
    args = visualize_map.parse_args(["--camera-style", "sphere"])

    assert args.camera_style == "sphere"


def test_auto_camera_style_uses_nearby_panorama_metadata(tmp_path):
    model_path = tmp_path / "sparse_best"
    model_path.mkdir()
    (tmp_path / "panorama_equirectangular_config.json").write_text(
        '{"views": [{"image_prefix": "pano_camera00/"}]}\n',
        encoding="utf-8",
    )

    style, metadata_path = visualize_map.resolve_camera_style(model_path, "auto")

    assert style == "sphere"
    assert metadata_path == tmp_path / "panorama_equirectangular_config.json"


def test_panorama_frame_groups_collapse_virtual_camera_images():
    images = [
        FakeImage("pano_camera00/a.png", [0.0, 0.0, 0.0]),
        FakeImage("pano_camera01/a.png", [0.0, 0.0, 0.0]),
        FakeImage("pano_camera00/b.png", [1.0, 2.0, 3.0]),
        FakeImage("pano_camera01/b.png", [1.0, 2.0, 3.0]),
    ]

    groups = visualize_map.panorama_frame_groups(
        images,
        ("pano_camera00/", "pano_camera01/"),
    )

    assert [group.name for group in groups] == ["a.png", "b.png"]
    assert [len(group.images) for group in groups] == [2, 2]
    assert groups[0].center.tolist() == [0.0, 0.0, 0.0]
    assert groups[1].center.tolist() == [1.0, 2.0, 3.0]


def test_panorama_sphere_radius_is_one_quarter_of_reference_frame():
    camera_scale = 8.0

    reference_frame_size = visualize_map.panorama_reference_frame_size(camera_scale)
    sphere_radius = visualize_map.panorama_sphere_radius(camera_scale)

    assert reference_frame_size == 6.0
    assert sphere_radius == 1.5
    assert sphere_radius == reference_frame_size / 4.0


def test_draw_camera_accepts_equirectangular_camera(monkeypatch):
    monkeypatch.setattr(visualize_map, "o3d", FakeOpen3DGeometry)
    image = FakeImage("panorama.jpg", [1.0, 2.0, 3.0])
    camera = FakeCamera("EQUIRECTANGULAR")

    geometries = visualize_map.draw_camera(image, camera, scale=8.0)

    assert [geometry.kind for geometry in geometries] == [
        "sphere",
        "coordinate_frame",
    ]
    assert geometries[0].metadata["radius"] == 1.5
    assert geometries[0].transforms[0][:3, 3].tolist() == [1.0, 2.0, 3.0]
    assert geometries[1].metadata["size"] == 6.0
    assert np.allclose(geometries[1].transforms[0][:3, :3], np.eye(3))
    assert geometries[1].transforms[0][:3, 3].tolist() == [1.0, 2.0, 3.0]


def test_panorama_coordinate_frame_transform_uses_pano_axes_and_center():
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform = visualize_map.panorama_coordinate_frame_transform(
        np.array([10.0, 20.0, 30.0]),
        rotation,
    )

    assert np.allclose(transform[:3, :3], rotation)
    assert transform[:3, 3].tolist() == [10.0, 20.0, 30.0]
    assert transform[3].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_panorama_rotation_world_from_group_undoes_virtual_camera_rotation():
    cam_from_pano = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    group = visualize_map.PanoramaFrameGroup(
        name="a.png",
        images=[FakeImage("pano_camera00/a.png", [0.0, 0.0, 0.0], cam_from_pano)],
        center=np.zeros(3),
    )
    views = (
        visualize_map.PanoramaViewMetadata(
            image_prefix="pano_camera00/",
            cam_from_pano_rotation=cam_from_pano,
        ),
    )

    rotation = visualize_map.panorama_rotation_world_from_group(group, views)

    assert np.allclose(rotation, np.eye(3))


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
        camera_style="auto",
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
            camera_style="auto",
            min_track_length=3,
            max_reprojection_error=0.0,
            point_size=2.0,
            remove_outliers=True,
            summary_only=False,
        )
