"""Browser-based selector for manual pinhole render directions."""

from __future__ import annotations

import io
import json
import mimetypes
import threading
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import degrees, pi, radians
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from colmap.camera import camera_from_fov
from render_pinhole.defaults import (
    DEFAULT_PINHOLE_FOV_X,
    DEFAULT_PINHOLE_FOV_Y,
    DEFAULT_PINHOLE_HEIGHT,
    DEFAULT_PINHOLE_INTERPOLATION,
    DEFAULT_PINHOLE_WIDTH,
)
from render_pinhole.renderer import EquirectangularToPinhole
from render_pinhole.selection_schema import (
    auto_grid_selection_store,
    build_selection_store,
    selection_record as flat_selection_record,
    selection_summary as flat_selection_summary,
)
from render_pinhole.geometry import validate_latitude, wrap_longitude
from render_pinhole.source_images import IMAGE_EXTENSIONS

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_PREVIEW_MAX_WIDTH = 512
DEFAULT_PREVIEW_MAX_HEIGHT = 384
DEFAULT_PREVIEW_JPEG_QUALITY = 80
DEFAULT_PREVIEW_SOURCE_MAX_WIDTH = 2048
DEFAULT_PREVIEW_SOURCE_MAX_HEIGHT = 1024
DEFAULT_PANORAMA_DISPLAY_MAX_WIDTH = 1024
DEFAULT_PANORAMA_DISPLAY_MAX_HEIGHT = 512
DEFAULT_PANORAMA_DISPLAY_JPEG_QUALITY = 75
PANORAMA_PREVIEW_CACHE_CONTROL = "no-store"
DEFAULT_DISPLAY_CACHE_SIZE = 8
DEFAULT_PREVIEW_RENDERER_CACHE_SIZE = 3
MAX_UNDO_DEPTH = 20
FLAT_SELECTIONS_FILENAME = "selections.json"
SELECTOR_MODE_IMAGE_DIR = "image_dir"

T = TypeVar("T")


def serve_image_dir_selector(
    image_dir: Path,
    output_dir: Path,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_records = image_dir_panorama_records(image_dir)
    if not image_records:
        raise RuntimeError(f"No panorama images found in {image_dir}")
    server = ManualPinholeSelectionServer(
        (host, port),
        ManualPinholeSelectionHandler,
        output_dir=output_dir,
        image_dir=image_dir,
        image_records=image_records,
    )
    url = f"http://{host}:{server.server_address[1]}"
    print(f"Manual pinhole selector: {url}")
    print(f"Image directory: {image_dir}")
    print(f"Selections: {output_dir / FLAT_SELECTIONS_FILENAME}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def pixel_to_longitude_latitude(
    x: float,
    y: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    clamped_x = min(max(float(x), 0.0), float(image_width))
    clamped_y = min(max(float(y), 0.0), float(image_height))
    longitude = (clamped_x / image_width) * 2.0 * pi - pi
    latitude = pi / 2.0 - (clamped_y / image_height) * pi
    return longitude, latitude


def remember_undo_state(
    undo_stack: list[dict[str, str | None] | str],
    snapshot: dict[str, str | None] | str,
    *,
    max_depth: int = MAX_UNDO_DEPTH,
) -> None:
    undo_stack.append(snapshot)
    del undo_stack[:-max_depth]


def read_flat_selection_snapshot(output_dir: Path) -> str | None:
    selections_path = output_dir / FLAT_SELECTIONS_FILENAME
    if not selections_path.is_file():
        return None
    return selections_path.read_text(encoding="utf-8")


def restore_flat_selection_snapshot(
    output_dir: Path,
    image_dir: Path,
    image_records: list[dict[str, Any]],
    undo_stack: list[Any],
) -> dict[str, Any] | None:
    if not undo_stack:
        return None
    snapshot = undo_stack.pop()
    selections_path = output_dir / FLAT_SELECTIONS_FILENAME
    if snapshot is None:
        selections_path.unlink(missing_ok=True)
    else:
        selections_path.write_text(str(snapshot), encoding="utf-8")
    return load_flat_selection_store(output_dir, image_dir, image_records)


def append_flat_selection(
    store: dict[str, Any],
    image_record: dict[str, Any],
    longitude: float,
    latitude: float,
) -> dict[str, Any]:
    pano_image_name = str(image_record["pano_image_name"])
    longitude = wrap_longitude(float(longitude))
    latitude = float(latitude)
    validate_latitude(latitude)
    selection = flat_selection_record(
        pano_image_name=pano_image_name,
        selection_index=next_flat_selection_index(
            store.get("selections", []),
            pano_image_name,
        ),
        longitude=longitude,
        latitude=latitude,
        selected_by="manual-pinhole-select",
    )
    store.setdefault("selections", []).append(selection)
    store["selection_mode"] = "manual"
    store["selection_summary"] = flat_selection_summary(store["selections"])
    return selection


def delete_flat_selection(
    store: dict[str, Any],
    image_record: dict[str, Any],
    selection_index: int,
) -> bool:
    pano_image_name = str(image_record["pano_image_name"])
    selections = store.setdefault("selections", [])
    original_count = len(selections)
    store["selections"] = [
        selection
        for selection in selections
        if not (
            selection.get("pano_image_name") == pano_image_name
            and int(selection.get("selection_index", -1)) == selection_index
        )
    ]
    deleted = len(store["selections"]) != original_count
    if deleted:
        store["selection_summary"] = flat_selection_summary(store["selections"])
    return deleted


def reset_flat_panorama_selections(
    store: dict[str, Any],
    image_record: dict[str, Any],
) -> int:
    pano_image_name = str(image_record["pano_image_name"])
    selections = store.setdefault("selections", [])
    remaining = [
        selection
        for selection in selections
        if selection.get("pano_image_name") != pano_image_name
    ]
    removed_count = len(selections) - len(remaining)
    store["selections"] = remaining
    store["selection_summary"] = flat_selection_summary(remaining)
    return removed_count


def reset_flat_collection_selections(store: dict[str, Any]) -> int:
    removed_count = len(store.get("selections", []))
    store["selections"] = []
    store["selection_summary"] = flat_selection_summary([])
    return removed_count


def replace_flat_auto_grid_selections(
    *,
    image_dir: Path,
    image_records: list[dict[str, Any]],
    delta_degrees: float,
) -> dict[str, Any]:
    return auto_grid_selection_store(
        image_dir=image_dir,
        pano_image_names=[
            str(image_record["pano_image_name"])
            for image_record in image_records
        ],
        delta_degrees=delta_degrees,
    )


def append_flat_roi_selections(
    store: dict[str, Any],
    image_record: dict[str, Any],
    *,
    longitude_min: float,
    longitude_max: float,
    latitude_min: float,
    latitude_max: float,
    delta_degrees: float,
) -> list[dict[str, Any]]:
    selections = []
    for longitude, latitude in roi_points_by_delta(
        longitude_min=longitude_min,
        longitude_max=longitude_max,
        latitude_min=latitude_min,
        latitude_max=latitude_max,
        delta_degrees=delta_degrees,
    ):
        selections.append(
            append_flat_selection(store, image_record, longitude, latitude)
        )
    return selections


def roi_points_by_delta(
    *,
    longitude_min: float,
    longitude_max: float,
    latitude_min: float,
    latitude_max: float,
    delta_degrees: float,
) -> list[tuple[float, float]]:
    if delta_degrees <= 0:
        raise ValueError("ROI delta angle must be positive")
    longitude_min = wrap_longitude(float(longitude_min))
    longitude_max = wrap_longitude(float(longitude_max))
    latitude_min = float(latitude_min)
    latitude_max = float(latitude_max)
    validate_latitude(latitude_min)
    validate_latitude(latitude_max)
    if longitude_max <= longitude_min:
        raise ValueError("ROI longitude_max must be greater than longitude_min")
    if latitude_max <= latitude_min:
        raise ValueError("ROI latitude_max must be greater than latitude_min")

    delta = radians(float(delta_degrees))
    longitude_values = axis_values_by_delta(longitude_min, longitude_max, delta)
    latitude_values = list(
        reversed(axis_values_by_delta(latitude_min, latitude_max, delta))
    )
    return [
        (longitude, latitude)
        for latitude in latitude_values
        for longitude in longitude_values
    ]


def axis_values_by_delta(minimum: float, maximum: float, delta: float) -> list[float]:
    span = maximum - minimum
    if span <= delta:
        return [(minimum + maximum) / 2.0]
    value = minimum + delta / 2.0
    points = []
    while value < maximum:
        points.append(value)
        value += delta
    return points


def render_flat_preview_jpeg(
    image_record: dict[str, Any],
    longitude: float,
    latitude: float,
    overrides: dict[str, Any] | None = None,
    *,
    renderer: EquirectangularToPinhole | None = None,
) -> bytes:
    pinhole = render_flat_preview_array(
        image_record,
        longitude,
        latitude,
        overrides,
        renderer=renderer,
    )
    buffer = io.BytesIO()
    Image.fromarray(pinhole).convert("RGB").save(
        buffer,
        format="JPEG",
        quality=DEFAULT_PREVIEW_JPEG_QUALITY,
        optimize=False,
    )
    return buffer.getvalue()


def render_flat_preview_png(
    image_record: dict[str, Any],
    longitude: float,
    latitude: float,
    overrides: dict[str, Any] | None = None,
    *,
    renderer: EquirectangularToPinhole | None = None,
) -> bytes:
    pinhole = render_flat_preview_array(
        image_record,
        longitude,
        latitude,
        overrides,
        renderer=renderer,
    )
    buffer = io.BytesIO()
    Image.fromarray(pinhole).save(buffer, format="PNG")
    return buffer.getvalue()


def render_flat_preview_array(
    image_record: dict[str, Any],
    longitude: float,
    latitude: float,
    overrides: dict[str, Any] | None = None,
    *,
    renderer: EquirectangularToPinhole | None = None,
) -> np.ndarray:
    source_path = Path(str(image_record["source_path"]))
    params = preview_render_params(render_params(flat_render_metadata(), overrides))
    longitude = wrap_longitude(float(longitude))
    latitude = float(latitude)
    validate_latitude(latitude)

    if renderer is None:
        renderer = EquirectangularToPinhole.from_file(source_path)
    return renderer.render(
        height=params["height"],
        width=params["width"],
        longitude=longitude,
        latitude=latitude,
        fov_x=params["fov_x"],
        fov_y=params["fov_y"],
        interpolation=params["interpolation"],
    )


def load_preview_renderer(
    source_path: Path,
    *,
    max_width: int = DEFAULT_PREVIEW_SOURCE_MAX_WIDTH,
    max_height: int = DEFAULT_PREVIEW_SOURCE_MAX_HEIGHT,
) -> EquirectangularToPinhole:
    if max_width <= 0 or max_height <= 0:
        raise ValueError("preview source max dimensions must be positive")
    with Image.open(source_path) as image:
        preview = image.convert("RGB")
        preview.thumbnail((max_width, max_height), Image.Resampling.BILINEAR)
        return EquirectangularToPinhole(np.asarray(preview))


def image_cache_key(source_path: Path) -> tuple[str, int, int]:
    stat = source_path.stat()
    return (str(source_path), int(stat.st_mtime_ns), int(stat.st_size))


def lru_cache_get(
    cache: OrderedDict[tuple[Any, ...], T],
    key: tuple[Any, ...],
) -> T | None:
    value = cache.get(key)
    if value is None:
        return None
    cache.move_to_end(key)
    return value


def lru_cache_put(
    cache: OrderedDict[tuple[Any, ...], T],
    key: tuple[Any, ...],
    value: T,
    *,
    max_size: int,
) -> T:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)
    return value


def render_flat_panorama_display_jpeg(
    image_record: dict[str, Any],
    *,
    max_width: int = DEFAULT_PANORAMA_DISPLAY_MAX_WIDTH,
    max_height: int = DEFAULT_PANORAMA_DISPLAY_MAX_HEIGHT,
) -> bytes:
    if max_width <= 0 or max_height <= 0:
        raise ValueError("display panorama max dimensions must be positive")
    source_path = Path(str(image_record["source_path"]))
    with Image.open(source_path) as image:
        display = image.convert("RGB")
        display.thumbnail((max_width, max_height), Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        display.save(
            buffer,
            format="JPEG",
            quality=DEFAULT_PANORAMA_DISPLAY_JPEG_QUALITY,
            optimize=False,
        )
    return buffer.getvalue()


def render_params(
    metadata: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = metadata.get("render_defaults", {})
    overrides = overrides or {}
    params = {
        "width": int(overrides.get("width", defaults.get("width"))),
        "height": int(overrides.get("height", defaults.get("height"))),
        "fov_x": float(overrides.get("fov_x", defaults.get("fov_x"))),
        "fov_y": float(overrides.get("fov_y", defaults.get("fov_y"))),
        "interpolation": str(
            overrides.get("interpolation", defaults.get("interpolation", "linear"))
        ),
        "output_extension": str(
            overrides.get("output_extension", defaults.get("output_extension", ".png"))
        ),
    }
    camera_from_fov(
        width=params["width"],
        height=params["height"],
        fov_x=params["fov_x"],
        fov_y=params["fov_y"],
    )
    if params["interpolation"] not in {"linear", "bilinear", "nearest"}:
        raise ValueError("interpolation must be one of: linear, bilinear, nearest")
    if not params["output_extension"].startswith("."):
        params["output_extension"] = f".{params['output_extension']}"
    return params


def flat_render_metadata() -> dict[str, Any]:
    return {
        "render_defaults": {
            "width": DEFAULT_PINHOLE_WIDTH,
            "height": DEFAULT_PINHOLE_HEIGHT,
            "fov_x": DEFAULT_PINHOLE_FOV_X,
            "fov_y": DEFAULT_PINHOLE_FOV_Y,
            "fov_x_degrees": degrees(DEFAULT_PINHOLE_FOV_X),
            "fov_y_degrees": degrees(DEFAULT_PINHOLE_FOV_Y),
            "interpolation": DEFAULT_PINHOLE_INTERPOLATION,
            "output_extension": ".png",
        }
    }


def preview_render_params(
    params: dict[str, Any],
    *,
    max_width: int = DEFAULT_PREVIEW_MAX_WIDTH,
    max_height: int = DEFAULT_PREVIEW_MAX_HEIGHT,
) -> dict[str, Any]:
    if max_width <= 0 or max_height <= 0:
        raise ValueError("preview max dimensions must be positive")
    preview_params = dict(params)
    scale = min(
        1.0,
        float(max_width) / float(params["width"]),
        float(max_height) / float(params["height"]),
    )
    preview_params["width"] = max(1, int(round(float(params["width"]) * scale)))
    preview_params["height"] = max(1, int(round(float(params["height"]) * scale)))
    return preview_params


def next_flat_selection_index(
    selections: list[dict[str, Any]],
    pano_image_name: str,
) -> int:
    pano_selections = [
        selection
        for selection in selections
        if selection.get("pano_image_name") == pano_image_name
    ]
    if not pano_selections:
        return 0
    return max(
        int(selection.get("selection_index", -1))
        for selection in pano_selections
    ) + 1


def image_dir_panorama_records(image_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(
        (
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.relative_to(image_dir).as_posix(),
    )
    return [
        {
            "pano_index": index,
            "pano_image_name": path.relative_to(image_dir).as_posix(),
            "source_path": str(path),
        }
        for index, path in enumerate(paths)
    ]


def image_record_at_index(
    image_records: list[dict[str, Any]],
    pano_index: int,
) -> dict[str, Any]:
    for image_record in image_records:
        if int(image_record["pano_index"]) == int(pano_index):
            return image_record
    raise KeyError(f"Panorama index not found: {pano_index}")


def load_flat_selection_store(
    output_dir: Path,
    image_dir: Path,
    image_records: list[dict[str, Any]],
) -> dict[str, Any]:
    selections_path = output_dir / FLAT_SELECTIONS_FILENAME
    if selections_path.is_file():
        store = json.loads(selections_path.read_text(encoding="utf-8"))
        store.setdefault("schema_version", 1)
        store.setdefault("source", {"image_dir": str(image_dir)})
        store.setdefault("selection_mode", "manual")
        store.setdefault("selections", [])
        store["selection_summary"] = flat_selection_summary(store["selections"])
        return store
    return build_selection_store(
        image_dir=image_dir,
        selections=[],
        selection_mode="manual",
    )


def write_flat_selection_store(output_dir: Path, store: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    store["selection_summary"] = flat_selection_summary(store.get("selections", []))
    (output_dir / FLAT_SELECTIONS_FILENAME).write_text(
        json.dumps(store, indent=2) + "\n",
        encoding="utf-8",
    )


def flat_api_state(
    *,
    image_dir: Path,
    output_dir: Path,
    image_records: list[dict[str, Any]],
    store: dict[str, Any],
) -> dict[str, Any]:
    selections_by_name: dict[str, list[dict[str, Any]]] = {}
    for selection in store.get("selections", []):
        selections_by_name.setdefault(str(selection.get("pano_image_name")), []).append(
            selection
        )
    panoramas = []
    for image_record in image_records:
        pano_image_name = str(image_record["pano_image_name"])
        selections = sorted(
            selections_by_name.get(pano_image_name, []),
            key=lambda selection: int(selection.get("selection_index", -1)),
        )
        panoramas.append(
            {
                "pano_index": image_record["pano_index"],
                "pano_filename": pano_image_name,
                "pano_image_name": pano_image_name,
                "scene_pano_filename": None,
                "source_pano_filename": pano_image_name,
                "source_status": "matched",
                "pose_status": "missing",
                "selections": selections,
                "quality": {},
            }
        )
    return {
        "schema_version": store.get("schema_version"),
        "selector_mode": SELECTOR_MODE_IMAGE_DIR,
        "pano_dir": str(image_dir),
        "image_dir": str(image_dir),
        "scene_workspace": None,
        "output_dir": str(output_dir),
        "selections_path": str(output_dir / FLAT_SELECTIONS_FILENAME),
        "render_defaults": flat_render_metadata()["render_defaults"],
        "selection_summary": flat_selection_summary(store.get("selections", [])),
        "panorama_count": len(panoramas),
        "panoramas": panoramas,
    }


class ManualPinholeSelectionServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        output_dir: Path,
        image_dir: Path,
        image_records: list[dict[str, Any]],
    ):
        super().__init__(server_address, handler_class)
        self.output_dir = output_dir
        self.image_dir = image_dir
        self.image_records = image_records
        self.state_lock = threading.Lock()
        self.undo_stack: list[Any] = []
        self.display_jpeg_cache: OrderedDict[tuple[Any, ...], bytes] = OrderedDict()
        self.preview_renderer_cache: OrderedDict[
            tuple[Any, ...],
            EquirectangularToPinhole,
        ] = OrderedDict()

    def image_dir_path(self) -> Path:
        return self.image_dir

    def cached_panorama_display_jpeg(self, image_record: dict[str, Any]) -> bytes:
        source_path = Path(str(image_record["source_path"]))
        key = (
            *image_cache_key(source_path),
            DEFAULT_PANORAMA_DISPLAY_MAX_WIDTH,
            DEFAULT_PANORAMA_DISPLAY_MAX_HEIGHT,
        )
        cached = lru_cache_get(self.display_jpeg_cache, key)
        if cached is not None:
            return cached
        display = render_flat_panorama_display_jpeg(image_record)
        return lru_cache_put(
            self.display_jpeg_cache,
            key,
            display,
            max_size=DEFAULT_DISPLAY_CACHE_SIZE,
        )

    def cached_preview_renderer(
        self,
        image_record: dict[str, Any],
    ) -> EquirectangularToPinhole:
        source_path = Path(str(image_record["source_path"]))
        key = (
            *image_cache_key(source_path),
            DEFAULT_PREVIEW_SOURCE_MAX_WIDTH,
            DEFAULT_PREVIEW_SOURCE_MAX_HEIGHT,
        )
        cached = lru_cache_get(self.preview_renderer_cache, key)
        if cached is not None:
            return cached
        renderer = load_preview_renderer(source_path)
        return lru_cache_put(
            self.preview_renderer_cache,
            key,
            renderer,
            max_size=DEFAULT_PREVIEW_RENDERER_CACHE_SIZE,
        )


class ManualPinholeSelectionHandler(BaseHTTPRequestHandler):
    server: ManualPinholeSelectionServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/":
                self.send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                with self.server.state_lock:
                    store = self.load_flat_store()
                    self.send_json(self.flat_state(store))
                return
            if path.startswith("/api/panorama-preview/"):
                pano_index = int(path.rsplit("/", 1)[1])
                with self.server.state_lock:
                    display = self.server.cached_panorama_display_jpeg(
                        self.image_record_at_index(pano_index)
                    )
                self.send_bytes(
                    display,
                    "image/jpeg",
                    headers={"Cache-Control": PANORAMA_PREVIEW_CACHE_CONTROL},
                )
                return
            if path.startswith("/api/panorama/"):
                pano_index = int(path.rsplit("/", 1)[1])
                with self.server.state_lock:
                    source_path = Path(
                        str(self.image_record_at_index(pano_index)["source_path"])
                    )
                content_type = mimetypes.guess_type(source_path.name)[0] or "image/png"
                self.send_bytes(source_path.read_bytes(), content_type)
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except (OSError, ValueError, KeyError) as error:
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self.read_json_body()
            if path == "/api/preview":
                with self.server.state_lock:
                    image_record = self.image_record_at_index(
                        int(payload["pano_index"]),
                    )
                    preview = render_flat_preview_jpeg(
                        image_record,
                        float(payload["longitude"]),
                        float(payload["latitude"]),
                        payload.get("render"),
                        renderer=self.server.cached_preview_renderer(image_record),
                    )
                self.send_bytes(preview, "image/jpeg")
                return
            if path == "/api/selections":
                with self.server.state_lock:
                    previous_snapshot = read_flat_selection_snapshot(
                        self.server.output_dir
                    )
                    store = self.load_flat_store()
                    selection = append_flat_selection(
                        store,
                        self.image_record_at_index(int(payload["pano_index"])),
                        float(payload["longitude"]),
                        float(payload["latitude"]),
                    )
                    remember_undo_state(self.server.undo_stack, previous_snapshot)
                    write_flat_selection_store(self.server.output_dir, store)
                    state = self.flat_state(store)
                self.send_json({"selection": selection, "state": state})
                return
            if path == "/api/sample-roi-selections":
                with self.server.state_lock:
                    previous_snapshot = read_flat_selection_snapshot(
                        self.server.output_dir
                    )
                    store = self.load_flat_store()
                    selections = append_flat_roi_selections(
                        store,
                        self.image_record_at_index(int(payload["pano_index"])),
                        longitude_min=float(payload["longitude_min"]),
                        longitude_max=float(payload["longitude_max"]),
                        latitude_min=float(payload["latitude_min"]),
                        latitude_max=float(payload["latitude_max"]),
                        delta_degrees=float(payload["delta_degrees"]),
                    )
                    remember_undo_state(self.server.undo_stack, previous_snapshot)
                    write_flat_selection_store(self.server.output_dir, store)
                    state = self.flat_state(store)
                self.send_json(
                    {
                        "created_count": len(selections),
                        "selections": selections,
                        "state": state,
                    }
                )
                return
            if path == "/api/auto-grid-selections":
                with self.server.state_lock:
                    previous_snapshot = read_flat_selection_snapshot(
                        self.server.output_dir
                    )
                    store = replace_flat_auto_grid_selections(
                        image_dir=self.server.image_dir_path(),
                        image_records=self.server.image_records,
                        delta_degrees=float(payload["delta_degrees"]),
                    )
                    remember_undo_state(self.server.undo_stack, previous_snapshot)
                    write_flat_selection_store(self.server.output_dir, store)
                    state = self.flat_state(store)
                self.send_json(
                    {
                        "created_count": len(store.get("selections", [])),
                        "state": state,
                    }
                )
                return
            if path == "/api/delete-selection":
                with self.server.state_lock:
                    previous_snapshot = read_flat_selection_snapshot(
                        self.server.output_dir
                    )
                    store = self.load_flat_store()
                    deleted = delete_flat_selection(
                        store,
                        self.image_record_at_index(int(payload["pano_index"])),
                        int(payload["selection_index"]),
                    )
                    if deleted:
                        remember_undo_state(
                            self.server.undo_stack,
                            previous_snapshot,
                        )
                        write_flat_selection_store(self.server.output_dir, store)
                    state = self.flat_state(store)
                status = HTTPStatus.OK if deleted else HTTPStatus.NOT_FOUND
                self.send_json({"deleted": deleted, "state": state}, status)
                return
            if path == "/api/reset-panorama-selections":
                with self.server.state_lock:
                    previous_snapshot = read_flat_selection_snapshot(
                        self.server.output_dir
                    )
                    store = self.load_flat_store()
                    removed_count = reset_flat_panorama_selections(
                        store,
                        self.image_record_at_index(int(payload["pano_index"])),
                    )
                    if removed_count:
                        remember_undo_state(
                            self.server.undo_stack,
                            previous_snapshot,
                        )
                        write_flat_selection_store(self.server.output_dir, store)
                    state = self.flat_state(store)
                self.send_json(
                    {"removed_count": removed_count, "state": state}
                )
                return
            if path == "/api/reset-collection-selections":
                with self.server.state_lock:
                    previous_snapshot = read_flat_selection_snapshot(
                        self.server.output_dir
                    )
                    store = self.load_flat_store()
                    removed_count = reset_flat_collection_selections(store)
                    if removed_count:
                        remember_undo_state(
                            self.server.undo_stack,
                            previous_snapshot,
                        )
                        write_flat_selection_store(self.server.output_dir, store)
                    state = self.flat_state(store)
                self.send_json(
                    {"removed_count": removed_count, "state": state}
                )
                return
            if path == "/api/undo":
                with self.server.state_lock:
                    store = restore_flat_selection_snapshot(
                        self.server.output_dir,
                        self.server.image_dir_path(),
                        self.server.image_records,
                        self.server.undo_stack,
                    )
                    undone = store is not None
                    if store is None:
                        store = self.load_flat_store()
                    state = self.flat_state(store)
                self.send_json({"undone": undone, "state": state})
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))

    def load_flat_store(self) -> dict[str, Any]:
        return load_flat_selection_store(
            self.server.output_dir,
            self.server.image_dir_path(),
            self.server.image_records,
        )

    def flat_state(self, store: dict[str, Any]) -> dict[str, Any]:
        return flat_api_state(
            image_dir=self.server.image_dir_path(),
            output_dir=self.server.output_dir,
            image_records=self.server.image_records,
            store=store,
        )

    def image_record_at_index(self, pano_index: int) -> dict[str, Any]:
        return image_record_at_index(self.server.image_records, pano_index)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8"))

    def send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_bytes(
        self,
        data: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        return


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Manual Pinhole Select</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #181c20;
      --panel-2: #20262b;
      --text: #eef2f3;
      --muted: #a6b0b8;
      --line: #303841;
      --accent: #4fb7a5;
      --warn: #d9a441;
      --danger: #df6c63;
    }
    * { box-sizing: border-box; }
    html {
      height: 100%;
      overflow: hidden;
    }
    body {
      margin: 0;
      height: 100%;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input, select {
      font: inherit;
    }
    button {
      min-height: 34px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 6px;
      padding: 6px 10px;
      cursor: pointer;
    }
    button.primary {
      border-color: var(--accent);
      background: #1b4a45;
    }
    button.danger {
      border-color: var(--danger);
      color: #ffd7d4;
    }
    button:disabled {
      color: #69737c;
      cursor: default;
    }
    input, select {
      width: 100%;
      min-height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #0f1215;
      color: var(--text);
      padding: 5px 8px;
    }
    .app {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      height: 100vh;
      overflow: hidden;
    }
    .stage {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      border-right: 1px solid var(--line);
    }
    .toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-height: 54px;
      padding: 10px 14px;
      border-bottom: 1px solid var(--line);
      background: #14181c;
    }
    .nav {
      display: flex;
      gap: 8px;
      align-items: center;
      min-width: 0;
    }
    .title {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--muted);
      font-size: 13px;
    }
    .viewport {
      min-height: 0;
      display: grid;
      place-items: center;
      padding: 14px;
      background: #0b0d0f;
    }
    .pano-frame {
      position: relative;
      display: inline-grid;
      max-width: 100%;
      max-height: calc(100vh - 82px);
    }
    #panorama {
      max-width: 100%;
      max-height: calc(100vh - 82px);
      object-fit: contain;
      border: 1px solid var(--line);
      cursor: crosshair;
      display: block;
      grid-area: 1 / 1;
    }
    .cursor-line {
      position: absolute;
      display: none;
      pointer-events: none;
      background: rgba(79, 183, 165, 0.86);
      z-index: 2;
    }
    .cursor-line.visible { display: block; }
    #cursorVertical {
      top: 0;
      bottom: 0;
      width: 1px;
    }
    #cursorHorizontal {
      left: 0;
      right: 0;
      height: 1px;
    }
    .marker-layer {
      position: absolute;
      inset: 0;
      pointer-events: none;
      z-index: 3;
    }
    .selection-marker {
      position: absolute;
      width: 9px;
      height: 9px;
      border: 1px solid #ffffff;
      border-radius: 50%;
      background: #ff3b30;
      box-shadow: 0 0 0 2px rgba(255, 59, 48, 0.32);
      transform: translate(-50%, -50%);
    }
    .selection-marker.highlighted {
      background: #32d74b;
      box-shadow: 0 0 0 2px rgba(50, 215, 75, 0.36);
      transform: translate(-50%, -50%) scale(1.1);
    }
    .roi-box {
      position: absolute;
      display: none;
      border: 2px solid var(--warn);
      background: rgba(217, 164, 65, 0.16);
      pointer-events: none;
      z-index: 4;
    }
    .roi-box.visible { display: block; }
    aside {
      display: grid;
      grid-template-rows: auto auto auto minmax(0, 1fr);
      height: 100vh;
      min-height: 0;
      overflow: hidden;
      background: var(--panel);
    }
    .panel {
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }
    .fields {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    label {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
    }
    .meta {
      display: grid;
      gap: 7px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .meta strong { color: var(--text); font-weight: 600; }
    .meta span { overflow-wrap: anywhere; }
    #preview {
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      border: 1px solid var(--line);
      background: #0c0e11;
      margin-top: 10px;
    }
    .actions {
      display: flex;
      gap: 8px;
      margin-top: 10px;
    }
    .reset-actions {
      display: flex;
      gap: 8px;
      padding: 10px 14px;
      border-bottom: 1px solid var(--line);
    }
    .checkbox-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .checkbox-row input {
      width: 16px;
      min-height: 16px;
    }
    .roi-controls {
      display: grid;
      grid-template-columns: 1fr 72px;
      align-items: end;
      gap: 8px;
      margin-top: 10px;
    }
    .roi-controls .checkbox-row {
      margin-top: 0;
    }
    .list {
      min-height: 0;
      overflow-y: auto;
      overscroll-behavior: contain;
      padding: 10px 14px 14px;
    }
    .selection {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 8px 0;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }
    .selection span { color: var(--text); }
    .selection-actions {
      display: flex;
      gap: 6px;
    }
    .status {
      color: var(--muted);
      font-size: 12px;
      min-height: 16px;
      margin-top: 8px;
    }
    @media (max-width: 920px) {
      .app {
        grid-template-columns: 1fr;
      }
      .stage {
        min-height: 64vh;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      aside {
        height: auto;
        min-height: 0;
      }
    }
  </style>
</head>
<body>
  <main class="app">
    <section class="stage">
      <div class="toolbar">
        <div class="nav">
          <button id="prev" title="Previous panorama">&larr;</button>
          <button id="next" title="Next panorama">&rarr;</button>
          <span id="counter" class="title"></span>
        </div>
        <div id="filename" class="title"></div>
      </div>
      <div class="viewport">
        <div id="panoFrame" class="pano-frame">
          <img id="panorama" alt="">
          <div id="markerLayer" class="marker-layer"></div>
          <div id="roiBox" class="roi-box"></div>
          <div id="cursorVertical" class="cursor-line"></div>
          <div id="cursorHorizontal" class="cursor-line"></div>
        </div>
      </div>
    </section>
    <aside>
      <section class="panel meta">
        <div><strong id="sourceLabel">Source</strong> <span id="source"></span></div>
        <div><strong id="sceneLabel">Scene</strong> <span id="scene"></span></div>
        <div><strong id="qualityLabel">Quality</strong> <span id="quality"></span></div>
        <div><strong>Selected</strong> <span id="selectedCount"></span></div>
      </section>
      <section class="panel">
        <div class="fields">
          <label>Longitude deg<input id="lonDeg" type="number" step="0.1"></label>
          <label>Latitude deg<input id="latDeg" type="number" step="0.1"></label>
          <label>Width<input id="width" type="number" step="1"></label>
          <label>Height<input id="height" type="number" step="1"></label>
          <label>FOV X deg<input id="fovXDeg" type="number" step="0.1"></label>
          <label>FOV Y deg<input id="fovYDeg" type="number" step="0.1"></label>
          <label>Interpolation
            <select id="interpolation">
              <option value="linear">linear</option>
              <option value="bilinear">bilinear</option>
              <option value="nearest">nearest</option>
            </select>
          </label>
        </div>
        <img id="preview" alt="">
        <div class="actions">
          <button id="previewBtn">Preview</button>
          <button id="saveBtn" class="primary">Save</button>
        </div>
        <div class="roi-controls">
          <label class="checkbox-row">
            <input id="roiMode" type="checkbox">
            ROI sampling
          </label>
          <label>Delta deg<input id="roiDeltaDeg" type="number" min="0.1" step="0.1" value="10"></label>
        </div>
        <div id="status" class="status"></div>
      </section>
      <section class="reset-actions">
        <button id="undoBtn">Undo</button>
        <button id="autoGridBtn">Auto Grid</button>
        <button id="resetPanoBtn" class="danger">Reset Pano</button>
        <button id="resetCollectionBtn" class="danger">Reset Collection</button>
      </section>
      <section class="list" id="selectionList"></section>
    </aside>
  </main>
  <script>
    const els = {
      panoFrame: document.getElementById("panoFrame"),
      panorama: document.getElementById("panorama"),
      markerLayer: document.getElementById("markerLayer"),
      roiBox: document.getElementById("roiBox"),
      cursorVertical: document.getElementById("cursorVertical"),
      cursorHorizontal: document.getElementById("cursorHorizontal"),
      prev: document.getElementById("prev"),
      next: document.getElementById("next"),
      counter: document.getElementById("counter"),
      filename: document.getElementById("filename"),
      sourceLabel: document.getElementById("sourceLabel"),
      source: document.getElementById("source"),
      sceneLabel: document.getElementById("sceneLabel"),
      scene: document.getElementById("scene"),
      qualityLabel: document.getElementById("qualityLabel"),
      quality: document.getElementById("quality"),
      selectedCount: document.getElementById("selectedCount"),
      lonDeg: document.getElementById("lonDeg"),
      latDeg: document.getElementById("latDeg"),
      width: document.getElementById("width"),
      height: document.getElementById("height"),
      fovXDeg: document.getElementById("fovXDeg"),
      fovYDeg: document.getElementById("fovYDeg"),
      interpolation: document.getElementById("interpolation"),
      preview: document.getElementById("preview"),
      previewBtn: document.getElementById("previewBtn"),
      saveBtn: document.getElementById("saveBtn"),
      roiMode: document.getElementById("roiMode"),
      roiDeltaDeg: document.getElementById("roiDeltaDeg"),
      undoBtn: document.getElementById("undoBtn"),
      autoGridBtn: document.getElementById("autoGridBtn"),
      resetPanoBtn: document.getElementById("resetPanoBtn"),
      resetCollectionBtn: document.getElementById("resetCollectionBtn"),
      status: document.getElementById("status"),
      selectionList: document.getElementById("selectionList")
    };
    let state = null;
    let index = 0;
    let roiStart = null;
    let isDrawingRoi = false;
    let suppressNextClick = false;
    let highlightedSelectionIndex = null;
    let loadedPanoramaKey = null;
    const PANORAMA_PREFETCH_RADIUS = 10;
    const PANORAMA_PREVIEW_CACHE_LIMIT = 21;
    const PANORAMA_PREFETCH_CONCURRENCY = 3;
    const HIGHLIGHT_DISTANCE_PX = 14;
    const panoramaPreviewCache = new Map();
    let panoramaPrefetchQueue = [];
    let activePanoramaPrefetches = 0;

    function radians(value) { return Number(value) * Math.PI / 180; }
    function degrees(value) { return Number(value) * 180 / Math.PI; }
    function pano() { return state.panoramas[index]; }
    function renderOverrides() {
      return {
        width: Number(els.width.value),
        height: Number(els.height.value),
        fov_x: radians(els.fovXDeg.value),
        fov_y: radians(els.fovYDeg.value),
        interpolation: els.interpolation.value
      };
    }
    function setStatus(message) { els.status.textContent = message || ""; }

    function navigatePanorama(delta) {
      if (!state || !state.panorama_count) return;
      const nextIndex = clamp(index + delta, 0, state.panorama_count - 1);
      if (nextIndex === index) return;
      index = nextIndex;
      draw();
    }

    async function refresh() {
      const response = await fetch("/api/state");
      state = await response.json();
      const defaults = state.render_defaults || {};
      els.width.value = defaults.width || 1280;
      els.height.value = defaults.height || 1280;
      els.fovXDeg.value = formatNumber(defaults.fov_x_degrees ?? degrees(defaults.fov_x ?? 1.221730476));
      els.fovYDeg.value = formatNumber(defaults.fov_y_degrees ?? degrees(defaults.fov_y ?? 1.221730476));
      els.interpolation.value = defaults.interpolation || "linear";
      index = Math.min(index, Math.max(state.panorama_count - 1, 0));
      draw();
    }

    function panoramaPreviewName(current) {
      return current.pano_image_name || current.pano_filename || current.pano_index;
    }

    function panoramaPreviewKey(current) {
      return `${current.pano_index}:${panoramaPreviewName(current)}`;
    }

    function panoramaPreviewUrl(current) {
      return `/api/panorama-preview/${current.pano_index}?v=${encodeURIComponent(panoramaPreviewKey(current))}`;
    }

    function touchPanoramaPreview(key, entry) {
      entry.lastUsed = Date.now();
      panoramaPreviewCache.delete(key);
      panoramaPreviewCache.set(key, entry);
      return entry;
    }

    function cachedPanoramaPreview(current) {
      const key = panoramaPreviewKey(current);
      const entry = panoramaPreviewCache.get(key);
      if (!entry || !entry.objectUrl) return null;
      return touchPanoramaPreview(key, entry);
    }

    async function loadPanoramaPreview(current) {
      const key = panoramaPreviewKey(current);
      const cached = panoramaPreviewCache.get(key);
      if (cached) {
        touchPanoramaPreview(key, cached);
        if (cached.objectUrl) return cached;
        if (cached.promise) return cached.promise;
      }

      const entry = {
        objectUrl: null,
        promise: null,
        panoIndex: Number(current.pano_index),
        lastUsed: Date.now()
      };
      entry.promise = fetch(panoramaPreviewUrl(current), {cache: "no-store"})
        .then(response => {
          if (!response.ok) throw new Error(`Could not load panorama preview ${current.pano_index}`);
          return response.blob();
        })
        .then(blob => {
          entry.objectUrl = URL.createObjectURL(blob);
          entry.promise = null;
          touchPanoramaPreview(key, entry);
          evictPanoramaPreviewCache(index);
          return entry;
        })
        .catch(error => {
          panoramaPreviewCache.delete(key);
          throw error;
        });
      panoramaPreviewCache.set(key, entry);
      return entry.promise;
    }

    function setPanoramaImage(current, entry) {
      const key = panoramaPreviewKey(current);
      if (!entry || !entry.objectUrl || loadedPanoramaKey !== key) return;
      els.panorama.onload = () => drawMarkers(pano());
      els.panorama.src = entry.objectUrl;
      if (els.panorama.complete) {
        requestAnimationFrame(() => drawMarkers(current));
      }
    }

    function evictPanoramaPreviewCache(centerIndex) {
      while (panoramaPreviewCache.size > PANORAMA_PREVIEW_CACHE_LIMIT) {
        let candidateKey = null;
        let candidateEntry = null;
        let candidateDistance = -1;
        for (const [key, entry] of panoramaPreviewCache) {
          if (entry.promise || key === loadedPanoramaKey) continue;
          const distance = Math.abs(entry.panoIndex - centerIndex);
          if (distance > candidateDistance) {
            candidateKey = key;
            candidateEntry = entry;
            candidateDistance = distance;
          }
        }
        if (!candidateKey || !candidateEntry) return;
        URL.revokeObjectURL(candidateEntry.objectUrl);
        panoramaPreviewCache.delete(candidateKey);
      }
    }

    function prefetchNearbyPanoramas(centerIndex) {
      if (!state || !state.panorama_count) return;
      const records = [];
      for (let offset = 1; offset <= PANORAMA_PREFETCH_RADIUS; offset += 1) {
        for (const candidateIndex of [centerIndex + offset, centerIndex - offset]) {
          if (candidateIndex < 0 || candidateIndex >= state.panorama_count) continue;
          records.push(state.panoramas[candidateIndex]);
        }
      }
      panoramaPrefetchQueue = records.filter(current => {
        const key = panoramaPreviewKey(current);
        return !panoramaPreviewCache.has(key);
      });
      pumpPanoramaPrefetchQueue();
      evictPanoramaPreviewCache(centerIndex);
    }

    function pumpPanoramaPrefetchQueue() {
      while (
        activePanoramaPrefetches < PANORAMA_PREFETCH_CONCURRENCY
        && panoramaPrefetchQueue.length
      ) {
        const current = panoramaPrefetchQueue.shift();
        if (!current || panoramaPreviewCache.has(panoramaPreviewKey(current))) continue;
        activePanoramaPrefetches += 1;
        loadPanoramaPreview(current)
          .catch(() => {})
          .finally(() => {
            activePanoramaPrefetches -= 1;
            pumpPanoramaPrefetchQueue();
          });
      }
    }

    function draw() {
      if (!state || !state.panorama_count) return;
      highlightedSelectionIndex = null;
      const current = pano();
      els.counter.textContent = `${index + 1} / ${state.panorama_count}`;
      els.filename.textContent = current.pano_filename || "";
      els.sourceLabel.textContent = "Image";
      els.sceneLabel.textContent = "Image Dir";
      els.qualityLabel.textContent = "Selections Path";
      els.source.textContent = current.pano_image_name || current.pano_filename || "";
      els.scene.textContent = state.image_dir || state.pano_dir || "";
      els.quality.textContent = state.selections_path || "";
      els.selectedCount.textContent = `${current.selections.length}`;
      els.prev.disabled = index === 0;
      els.next.disabled = index >= state.panorama_count - 1;
      els.autoGridBtn.disabled = false;
      clearMarkers();
      const nextPanoramaKey = panoramaPreviewKey(current);
      const panoramaChanged = loadedPanoramaKey !== nextPanoramaKey;
      if (panoramaChanged) {
        loadedPanoramaKey = nextPanoramaKey;
        const cached = cachedPanoramaPreview(current);
        if (cached) {
          setPanoramaImage(current, cached);
        } else {
          loadPanoramaPreview(current)
            .then(entry => setPanoramaImage(current, entry))
            .catch(error => setStatus(error.message));
        }
      }
      drawSelections(current);
      if (!panoramaChanged && els.panorama.complete) {
        requestAnimationFrame(() => drawMarkers(current));
      }
      prefetchNearbyPanoramas(index);
      setStatus("");
    }

    function clearMarkers() {
      els.markerLayer.innerHTML = "";
    }

    function selectionDisplayPoint(selection, rect) {
      const longitude = Number(selection.longitude);
      const latitude = Number(selection.latitude);
      if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) return null;
      return {
        x: clamp(((longitude + Math.PI) / (2 * Math.PI)) * rect.width, 0, rect.width),
        y: clamp(((Math.PI / 2 - latitude) / Math.PI) * rect.height, 0, rect.height)
      };
    }

    function drawMarkers(current) {
      clearMarkers();
      if (!current || !current.selections.length) return;
      const rect = els.panorama.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      for (const selection of current.selections) {
        const point = selectionDisplayPoint(selection, rect);
        if (!point) continue;
        const selectionIndex = Number(selection.selection_index);
        const marker = document.createElement("div");
        marker.className = "selection-marker";
        if (selectionIndex === highlightedSelectionIndex) {
          marker.classList.add("highlighted");
        }
        marker.style.left = `${point.x}px`;
        marker.style.top = `${point.y}px`;
        marker.title = `sel-${String(selection.selection_index).padStart(3, "0")}`;
        els.markerLayer.appendChild(marker);
      }
    }

    function nearestSelectionIndex(point) {
      const current = pano();
      if (!current || !current.selections.length) return null;
      let nearest = null;
      let nearestDistance = HIGHLIGHT_DISTANCE_PX;
      for (const selection of current.selections) {
        const selectionPoint = selectionDisplayPoint(selection, point.rect);
        if (!selectionPoint) continue;
        const distance = Math.hypot(point.x - selectionPoint.x, point.y - selectionPoint.y);
        if (distance <= nearestDistance) {
          nearest = Number(selection.selection_index);
          nearestDistance = distance;
        }
      }
      return nearest;
    }

    function updateHighlightedSelection(point) {
      const nextHighlightedSelectionIndex = nearestSelectionIndex(point);
      if (nextHighlightedSelectionIndex === highlightedSelectionIndex) return;
      highlightedSelectionIndex = nextHighlightedSelectionIndex;
      drawMarkers(pano());
    }

    function clearHighlightedSelection() {
      if (highlightedSelectionIndex === null) return;
      highlightedSelectionIndex = null;
      drawMarkers(pano());
    }

    function drawSelections(current) {
      els.selectionList.innerHTML = "";
      if (!current.selections.length) {
        els.selectionList.textContent = "No saved selections.";
        return;
      }
      for (const selection of current.selections) {
        const row = document.createElement("div");
        row.className = "selection";
        const label = document.createElement("div");
        label.innerHTML = `<span>sel-${String(selection.selection_index).padStart(3, "0")}</span> lon ${formatNumber(selection.longitude_degrees)} lat ${formatNumber(selection.latitude_degrees)}`;
        const actions = document.createElement("div");
        actions.className = "selection-actions";
        const check = document.createElement("button");
        check.textContent = "Check";
        check.onclick = () => checkSelection(selection);
        const remove = document.createElement("button");
        remove.className = "danger";
        remove.textContent = "Delete";
        remove.onclick = () => deleteSelection(selection.selection_index);
        actions.append(check, remove);
        row.append(label, actions);
        els.selectionList.appendChild(row);
      }
    }

    function checkSelection(selection) {
      els.lonDeg.value = formatNumber(selection.longitude_degrees ?? degrees(selection.longitude));
      els.latDeg.value = formatNumber(selection.latitude_degrees ?? degrees(selection.latitude));
      if (selection.width) els.width.value = selection.width;
      if (selection.height) els.height.value = selection.height;
      if (selection.fov_x) els.fovXDeg.value = formatNumber(degrees(selection.fov_x));
      if (selection.fov_y) els.fovYDeg.value = formatNumber(degrees(selection.fov_y));
      if (selection.interpolation) els.interpolation.value = selection.interpolation;
      preview().catch(error => setStatus(error.message));
    }

    async function preview() {
      const current = pano();
      setStatus("Rendering preview...");
      const response = await fetch("/api/preview", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          pano_index: current.pano_index,
          longitude: radians(els.lonDeg.value),
          latitude: radians(els.latDeg.value),
          render: renderOverrides()
        })
      });
      if (!response.ok) throw new Error(await response.text());
      const blob = await response.blob();
      els.preview.src = URL.createObjectURL(blob);
      setStatus("Preview ready.");
    }

    async function saveSelection({redraw = true} = {}) {
      const current = pano();
      setStatus("Saving...");
      const response = await fetch("/api/selections", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          pano_index: current.pano_index,
          longitude: radians(els.lonDeg.value),
          latitude: radians(els.latDeg.value),
          render: renderOverrides()
        })
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      state = payload.state;
      if (redraw) {
        draw();
        setStatus("Saved.");
      }
      return payload;
    }

    async function saveSelectionAndAdvance(advanceAfterSave) {
      await saveSelection({redraw: false});
      if (advanceAfterSave && index < state.panorama_count - 1) {
        index += 1;
        draw();
        setStatus("Saved. Moved to next panorama.");
      } else {
        draw();
        setStatus("Saved.");
      }
    }

    async function deleteSelection(selectionIndex) {
      const current = pano();
      const response = await fetch("/api/delete-selection", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          pano_index: current.pano_index,
          selection_index: selectionIndex
        })
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      state = payload.state;
      draw();
      setStatus("Deleted.");
    }

    async function resetPanoSelections() {
      const current = pano();
      if (!confirm("Reset all saved pinhole views for this panorama?")) return;
      const response = await fetch("/api/reset-panorama-selections", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({pano_index: current.pano_index})
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      state = payload.state;
      draw();
      setStatus(`Reset ${payload.removed_count} pano view(s).`);
    }

    async function resetCollectionSelections() {
      if (!confirm("Reset the entire saved pinhole-view collection?")) return;
      const response = await fetch("/api/reset-collection-selections", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({})
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      state = payload.state;
      draw();
      setStatus(`Reset ${payload.removed_count} collection view(s).`);
    }

    async function undoLastAction() {
      const response = await fetch("/api/undo", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({})
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      state = payload.state;
      draw();
      setStatus(payload.undone ? "Undone." : "Nothing to undo.");
    }

    async function sampleRoiSelections(bounds) {
      const current = pano();
      const deltaDegrees = Math.max(0.1, Number(els.roiDeltaDeg.value) || 10);
      setStatus(`Sampling ROI at ${formatNumber(deltaDegrees)} deg...`);
      const response = await fetch("/api/sample-roi-selections", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          pano_index: current.pano_index,
          longitude_min: bounds.longitudeMin,
          longitude_max: bounds.longitudeMax,
          latitude_min: bounds.latitudeMin,
          latitude_max: bounds.latitudeMax,
          delta_degrees: deltaDegrees,
          render: renderOverrides()
        })
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      state = payload.state;
      draw();
      setStatus(`Sampled ${payload.created_count} ROI view(s).`);
    }

    async function autoGridSelections() {
      const deltaDegrees = Math.max(0.1, Number(els.roiDeltaDeg.value) || 45);
      if (!confirm(`Replace all saved selections with a ${formatNumber(deltaDegrees)} degree auto grid for every panorama?`)) return;
      setStatus(`Creating ${formatNumber(deltaDegrees)} deg auto grid...`);
      const response = await fetch("/api/auto-grid-selections", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({delta_degrees: deltaDegrees})
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      state = payload.state;
      draw();
      setStatus(`Auto Grid created ${payload.created_count} view(s).`);
    }

    function cursorPoint(event) {
      const rect = els.panorama.getBoundingClientRect();
      const x = Math.min(Math.max(event.clientX - rect.left, 0), rect.width);
      const y = Math.min(Math.max(event.clientY - rect.top, 0), rect.height);
      return {rect, x, y};
    }

    function pointAngles(point) {
      return {
        longitude: (point.x / point.rect.width) * 2 * Math.PI - Math.PI,
        latitude: Math.PI / 2 - (point.y / point.rect.height) * Math.PI
      };
    }

    function updateCursorLinesFromPoint(point) {
      els.cursorVertical.style.left = `${point.x}px`;
      els.cursorHorizontal.style.top = `${point.y}px`;
      els.cursorVertical.classList.add("visible");
      els.cursorHorizontal.classList.add("visible");
    }

    function updateCursorLines(event) {
      const point = cursorPoint(event);
      updateCursorLinesFromPoint(point);
    }

    function hideCursorLines() {
      els.cursorVertical.classList.remove("visible");
      els.cursorHorizontal.classList.remove("visible");
      clearHighlightedSelection();
    }

    function showRoiBox(start, end) {
      const left = Math.min(start.x, end.x);
      const top = Math.min(start.y, end.y);
      const width = Math.abs(end.x - start.x);
      const height = Math.abs(end.y - start.y);
      els.roiBox.style.left = `${left}px`;
      els.roiBox.style.top = `${top}px`;
      els.roiBox.style.width = `${width}px`;
      els.roiBox.style.height = `${height}px`;
      els.roiBox.classList.add("visible");
    }

    function hideRoiBox() {
      els.roiBox.classList.remove("visible");
    }

    function startRoi(event) {
      if (!els.roiMode.checked || event.button !== 0) return;
      event.preventDefault();
      roiStart = cursorPoint(event);
      isDrawingRoi = true;
      suppressNextClick = true;
      showRoiBox(roiStart, roiStart);
    }

    function updatePanoramaPointer(event) {
      const point = cursorPoint(event);
      updateCursorLinesFromPoint(point);
      updateHighlightedSelection(point);
      if (isDrawingRoi && roiStart) {
        showRoiBox(roiStart, point);
      }
    }

    function finishRoi(event) {
      if (!isDrawingRoi || !roiStart) return;
      event.preventDefault();
      const end = cursorPoint(event);
      const start = roiStart;
      isDrawingRoi = false;
      roiStart = null;
      hideRoiBox();
      if (end.x <= start.x || end.y <= start.y) {
        setStatus("Draw ROI from upper-left to lower-right.");
        return;
      }
      if (end.x - start.x < 4 || end.y - start.y < 4) {
        setStatus("ROI is too small.");
        return;
      }
      const topLeft = pointAngles(start);
      const bottomRight = pointAngles(end);
      sampleRoiSelections({
        longitudeMin: topLeft.longitude,
        longitudeMax: bottomRight.longitude,
        latitudeMin: bottomRight.latitude,
        latitudeMax: topLeft.latitude
      }).catch(error => setStatus(error.message));
    }

    function pickFromClick(event) {
      event.preventDefault();
      if (suppressNextClick) {
        suppressNextClick = false;
        return;
      }
      if (els.roiMode.checked) return;
      const point = cursorPoint(event);
      const angles = pointAngles(point);
      els.lonDeg.value = degrees(angles.longitude).toFixed(2);
      els.latDeg.value = degrees(angles.latitude).toFixed(2);
      const saveModifier = event.metaKey || event.ctrlKey;
      if (saveModifier) {
        preview()
          .then(() => saveSelectionAndAdvance(event.shiftKey))
          .catch(error => setStatus(error.message));
      } else {
        preview().catch(error => setStatus(error.message));
      }
    }

    function formatNumber(value) {
      return value === null || value === undefined ? "n/a" : Number(value).toFixed(2);
    }

    function clamp(value, min, max) {
      return Math.min(Math.max(value, min), max);
    }

    els.prev.onclick = () => navigatePanorama(-1);
    els.next.onclick = () => navigatePanorama(1);
    els.panorama.onclick = pickFromClick;
    els.panorama.onmousedown = startRoi;
    els.panorama.onmouseup = finishRoi;
    els.panorama.onmousemove = updatePanoramaPointer;
    els.panorama.onmouseleave = hideCursorLines;
    els.previewBtn.onclick = () => preview().catch(error => setStatus(error.message));
    els.saveBtn.onclick = () => saveSelection().catch(error => setStatus(error.message));
    els.undoBtn.onclick = () => undoLastAction().catch(error => setStatus(error.message));
    els.autoGridBtn.onclick = () => autoGridSelections().catch(error => setStatus(error.message));
    els.resetPanoBtn.onclick = () => resetPanoSelections().catch(error => setStatus(error.message));
    els.resetCollectionBtn.onclick = () => resetCollectionSelections().catch(error => setStatus(error.message));
    window.addEventListener("keydown", event => {
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        navigatePanorama(-1);
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        navigatePanorama(1);
      }
      if (event.key.toLowerCase() === "d" && highlightedSelectionIndex !== null) {
        event.preventDefault();
        deleteSelection(highlightedSelectionIndex).catch(error => setStatus(error.message));
      }
      if (event.key.toLowerCase() === "s") els.saveBtn.click();
    });
    window.addEventListener("resize", () => {
      if (state && state.panorama_count) drawMarkers(pano());
    });
    refresh().catch(error => setStatus(error.message));
  </script>
</body>
</html>
"""
