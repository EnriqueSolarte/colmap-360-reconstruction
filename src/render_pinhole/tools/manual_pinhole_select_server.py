"""Browser-based selector for manual pinhole render directions."""

from __future__ import annotations

import io
import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import ceil, degrees, pi, sqrt
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image

from colmap.camera import camera_from_fov
from render_pinhole.renderer import EquirectangularToPinhole
from render_pinhole.tools.manual_pinhole_render import (
    METADATA_FILENAME,
    SOURCE_STATUS_MISSING,
    pinhole_camera_pose_report,
    validate_latitude,
    wrap_longitude,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_PREVIEW_MAX_WIDTH = 512
DEFAULT_PREVIEW_MAX_HEIGHT = 384
DEFAULT_PANORAMA_DISPLAY_MAX_WIDTH = 1600
DEFAULT_PANORAMA_DISPLAY_MAX_HEIGHT = 800


def serve_selector(
    output_dir: Path,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> None:
    metadata_path = output_dir / METADATA_FILENAME
    server = ManualPinholeSelectionServer(
        (host, port),
        ManualPinholeSelectionHandler,
        output_dir=output_dir,
    )
    url = f"http://{host}:{server.server_address[1]}"
    print(f"Manual pinhole selector: {url}")
    print(f"Metadata: {metadata_path}")
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


def load_metadata(output_dir: Path) -> dict[str, Any]:
    metadata_path = output_dir / METADATA_FILENAME
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def write_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    metadata_path = output_dir / METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def append_selection(
    metadata: dict[str, Any],
    pano_index: int,
    longitude: float,
    latitude: float,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    panorama = panorama_at_index(metadata, pano_index)
    selection = selection_record(metadata, panorama, longitude, latitude, overrides)
    panorama.setdefault("selections", []).append(selection)
    metadata["selection_summary"] = selection_summary(metadata)
    return selection


def delete_selection(
    metadata: dict[str, Any],
    pano_index: int,
    selection_index: int,
) -> bool:
    panorama = panorama_at_index(metadata, pano_index)
    selections = panorama.setdefault("selections", [])
    original_count = len(selections)
    panorama["selections"] = [
        selection
        for selection in selections
        if int(selection.get("selection_index", -1)) != selection_index
    ]
    deleted = len(panorama["selections"]) != original_count
    if deleted:
        metadata["selection_summary"] = selection_summary(metadata)
    return deleted


def reset_panorama_selections(metadata: dict[str, Any], pano_index: int) -> int:
    panorama = panorama_at_index(metadata, pano_index)
    removed_count = len(panorama.get("selections", []))
    panorama["selections"] = []
    metadata["selection_summary"] = selection_summary(metadata)
    return removed_count


def reset_collection_selections(metadata: dict[str, Any]) -> int:
    removed_count = 0
    for panorama in metadata.get("panoramas", []):
        removed_count += len(panorama.get("selections", []))
        panorama["selections"] = []
    metadata["selection_summary"] = selection_summary(metadata)
    return removed_count


def append_roi_selections(
    metadata: dict[str, Any],
    pano_index: int,
    *,
    longitude_min: float,
    longitude_max: float,
    latitude_min: float,
    latitude_max: float,
    count: int,
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    panorama = panorama_at_index(metadata, pano_index)
    selections = []
    for longitude, latitude in uniform_roi_points(
        longitude_min=longitude_min,
        longitude_max=longitude_max,
        latitude_min=latitude_min,
        latitude_max=latitude_max,
        count=count,
    ):
        selection = selection_record(
            metadata,
            panorama,
            longitude,
            latitude,
            overrides,
        )
        panorama.setdefault("selections", []).append(selection)
        selections.append(selection)
    metadata["selection_summary"] = selection_summary(metadata)
    return selections


def uniform_roi_points(
    *,
    longitude_min: float,
    longitude_max: float,
    latitude_min: float,
    latitude_max: float,
    count: int,
) -> list[tuple[float, float]]:
    if count <= 0:
        raise ValueError("ROI sample count must be positive")
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

    cols = int(ceil(sqrt(count)))
    rows = int(ceil(count / cols))
    points = []
    for row in range(rows):
        latitude = latitude_max - ((row + 0.5) / rows) * (latitude_max - latitude_min)
        for col in range(cols):
            longitude = longitude_min + (
                (col + 0.5) / cols
            ) * (longitude_max - longitude_min)
            points.append((longitude, latitude))
            if len(points) == count:
                return points
    return points


def selection_record(
    metadata: dict[str, Any],
    panorama: dict[str, Any],
    longitude: float,
    latitude: float,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    longitude = wrap_longitude(float(longitude))
    latitude = float(latitude)
    validate_latitude(latitude)
    params = render_params(metadata, overrides)
    selection = {
        "selection_index": next_selection_index(panorama.get("selections", [])),
        "longitude": longitude,
        "latitude": latitude,
        "longitude_degrees": degrees(longitude),
        "latitude_degrees": degrees(latitude),
        "width": params["width"],
        "height": params["height"],
        "fov_x": params["fov_x"],
        "fov_y": params["fov_y"],
        "interpolation": params["interpolation"],
        "output_extension": params["output_extension"],
        "render_status": "pending",
        "selected_by": "manual-pinhole-select",
        "pinhole_camera_pose": pinhole_camera_pose_report(
            panorama.get("pano_camera_pose"),
            longitude,
            latitude,
        ),
    }
    return selection


def render_preview_png(
    metadata: dict[str, Any],
    pano_index: int,
    longitude: float,
    latitude: float,
    overrides: dict[str, Any] | None = None,
) -> bytes:
    panorama = panorama_at_index(metadata, pano_index)
    source_path = source_panorama_path(panorama)
    params = preview_render_params(render_params(metadata, overrides))
    longitude = wrap_longitude(float(longitude))
    latitude = float(latitude)
    validate_latitude(latitude)

    renderer = EquirectangularToPinhole.from_file(source_path)
    pinhole = renderer.render(
        height=params["height"],
        width=params["width"],
        longitude=longitude,
        latitude=latitude,
        fov_x=params["fov_x"],
        fov_y=params["fov_y"],
        interpolation=params["interpolation"],
    )
    buffer = io.BytesIO()
    Image.fromarray(pinhole).save(buffer, format="PNG")
    return buffer.getvalue()


def render_panorama_display_jpeg(
    metadata: dict[str, Any],
    pano_index: int,
    *,
    max_width: int = DEFAULT_PANORAMA_DISPLAY_MAX_WIDTH,
    max_height: int = DEFAULT_PANORAMA_DISPLAY_MAX_HEIGHT,
) -> bytes:
    if max_width <= 0 or max_height <= 0:
        raise ValueError("display panorama max dimensions must be positive")
    panorama = panorama_at_index(metadata, pano_index)
    source_path = source_panorama_path(panorama)
    with Image.open(source_path) as image:
        display = image.convert("RGB")
        display.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        display.save(buffer, format="JPEG", quality=85, optimize=True)
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


def panorama_at_index(metadata: dict[str, Any], pano_index: int) -> dict[str, Any]:
    for panorama in metadata.get("panoramas", []):
        if int(panorama.get("pano_index", -1)) == int(pano_index):
            return panorama
    raise KeyError(f"Panorama index not found: {pano_index}")


def source_panorama_path(panorama: dict[str, Any]) -> Path:
    if panorama.get("source_status") == SOURCE_STATUS_MISSING:
        raise FileNotFoundError(
            "Source panorama is missing for "
            f"{panorama.get('scene_pano_filename') or panorama.get('pano_filename')}"
        )
    path = panorama.get("source_pano_path") or panorama.get("pano_path")
    if path is None:
        raise FileNotFoundError(
            "Source panorama path is missing for "
            f"{panorama.get('scene_pano_filename') or panorama.get('pano_filename')}"
        )
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source panorama file not found: {source_path}")
    return source_path


def next_selection_index(selections: list[dict[str, Any]]) -> int:
    if not selections:
        return 0
    return max(int(selection.get("selection_index", -1)) for selection in selections) + 1


def selection_summary(metadata: dict[str, Any]) -> dict[str, int]:
    panoramas = metadata.get("panoramas", [])
    selected_panorama_count = sum(
        1 for panorama in panoramas if panorama.get("selections")
    )
    selection_count = sum(
        len(panorama.get("selections", [])) for panorama in panoramas
    )
    return {
        "selected_panorama_count": selected_panorama_count,
        "selection_count": selection_count,
    }


def api_state(metadata: dict[str, Any]) -> dict[str, Any]:
    panoramas = []
    for panorama in metadata.get("panoramas", []):
        quality = panorama.get("colmap_quality") or {}
        panoramas.append(
            {
                "pano_index": panorama.get("pano_index"),
                "pano_filename": panorama.get("pano_filename"),
                "scene_pano_filename": panorama.get("scene_pano_filename"),
                "source_pano_filename": panorama.get("source_pano_filename"),
                "source_status": panorama.get("source_status"),
                "pose_status": panorama.get("pose_status"),
                "selections": panorama.get("selections", []),
                "quality": {
                    "registered_virtual_image_count": quality.get(
                        "registered_virtual_image_count"
                    ),
                    "expected_virtual_image_count": quality.get(
                        "expected_virtual_image_count"
                    ),
                    "active_virtual_camera_count": quality.get(
                        "active_virtual_camera_count"
                    ),
                    "triangulated_observation_count": quality.get(
                        "triangulated_observation_count"
                    ),
                    "unique_point3D_count": quality.get("unique_point3D_count"),
                    "mean_reprojection_error_px": quality.get(
                        "mean_reprojection_error_px"
                    ),
                    "median_reprojection_error_px": quality.get(
                        "median_reprojection_error_px"
                    ),
                    "p90_reprojection_error_px": quality.get(
                        "p90_reprojection_error_px"
                    ),
                    "connected_panorama_count": quality.get(
                        "connected_panorama_count"
                    ),
                },
            }
        )
    return {
        "schema_version": metadata.get("schema_version"),
        "pano_dir": metadata.get("pano_dir"),
        "scene_workspace": metadata.get("scene_workspace"),
        "output_dir": metadata.get("output_dir"),
        "render_defaults": metadata.get("render_defaults", {}),
        "selection_summary": selection_summary(metadata),
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
    ):
        super().__init__(server_address, handler_class)
        self.output_dir = output_dir
        self.metadata_lock = threading.Lock()


class ManualPinholeSelectionHandler(BaseHTTPRequestHandler):
    server: ManualPinholeSelectionServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/":
                self.send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                with self.server.metadata_lock:
                    metadata = load_metadata(self.server.output_dir)
                self.send_json(api_state(metadata))
                return
            if path.startswith("/api/panorama-preview/"):
                pano_index = int(path.rsplit("/", 1)[1])
                with self.server.metadata_lock:
                    metadata = load_metadata(self.server.output_dir)
                display = render_panorama_display_jpeg(metadata, pano_index)
                self.send_bytes(
                    display,
                    "image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"},
                )
                return
            if path.startswith("/api/panorama/"):
                pano_index = int(path.rsplit("/", 1)[1])
                with self.server.metadata_lock:
                    metadata = load_metadata(self.server.output_dir)
                    panorama = panorama_at_index(metadata, pano_index)
                    source_path = source_panorama_path(panorama)
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
                with self.server.metadata_lock:
                    metadata = load_metadata(self.server.output_dir)
                preview = render_preview_png(
                    metadata,
                    int(payload["pano_index"]),
                    float(payload["longitude"]),
                    float(payload["latitude"]),
                    payload.get("render"),
                )
                self.send_bytes(preview, "image/png")
                return
            if path == "/api/selections":
                with self.server.metadata_lock:
                    metadata = load_metadata(self.server.output_dir)
                    selection = append_selection(
                        metadata,
                        int(payload["pano_index"]),
                        float(payload["longitude"]),
                        float(payload["latitude"]),
                        payload.get("render"),
                    )
                    write_metadata(self.server.output_dir, metadata)
                self.send_json({"selection": selection, "state": api_state(metadata)})
                return
            if path == "/api/sample-roi-selections":
                with self.server.metadata_lock:
                    metadata = load_metadata(self.server.output_dir)
                    selections = append_roi_selections(
                        metadata,
                        int(payload["pano_index"]),
                        longitude_min=float(payload["longitude_min"]),
                        longitude_max=float(payload["longitude_max"]),
                        latitude_min=float(payload["latitude_min"]),
                        latitude_max=float(payload["latitude_max"]),
                        count=int(payload["count"]),
                        overrides=payload.get("render"),
                    )
                    write_metadata(self.server.output_dir, metadata)
                self.send_json(
                    {
                        "created_count": len(selections),
                        "selections": selections,
                        "state": api_state(metadata),
                    }
                )
                return
            if path == "/api/delete-selection":
                with self.server.metadata_lock:
                    metadata = load_metadata(self.server.output_dir)
                    deleted = delete_selection(
                        metadata,
                        int(payload["pano_index"]),
                        int(payload["selection_index"]),
                    )
                    write_metadata(self.server.output_dir, metadata)
                status = HTTPStatus.OK if deleted else HTTPStatus.NOT_FOUND
                self.send_json({"deleted": deleted, "state": api_state(metadata)}, status)
                return
            if path == "/api/reset-panorama-selections":
                with self.server.metadata_lock:
                    metadata = load_metadata(self.server.output_dir)
                    removed_count = reset_panorama_selections(
                        metadata,
                        int(payload["pano_index"]),
                    )
                    write_metadata(self.server.output_dir, metadata)
                self.send_json(
                    {"removed_count": removed_count, "state": api_state(metadata)}
                )
                return
            if path == "/api/reset-collection-selections":
                with self.server.metadata_lock:
                    metadata = load_metadata(self.server.output_dir)
                    removed_count = reset_collection_selections(metadata)
                    write_metadata(self.server.output_dir, metadata)
                self.send_json(
                    {"removed_count": removed_count, "state": api_state(metadata)}
                )
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))

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
        <div><strong>Source</strong> <span id="source"></span></div>
        <div><strong>Scene</strong> <span id="scene"></span></div>
        <div><strong>Quality</strong> <span id="quality"></span></div>
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
        <label class="checkbox-row">
          <input id="cmdSaveOnly" type="checkbox">
          Cmd-click save only
        </label>
        <div class="roi-controls">
          <label class="checkbox-row">
            <input id="roiMode" type="checkbox">
            ROI sampling
          </label>
          <label>N<input id="roiCount" type="number" min="1" step="1" value="9"></label>
        </div>
        <div id="status" class="status"></div>
      </section>
      <section class="reset-actions">
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
      source: document.getElementById("source"),
      scene: document.getElementById("scene"),
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
      cmdSaveOnly: document.getElementById("cmdSaveOnly"),
      roiMode: document.getElementById("roiMode"),
      roiCount: document.getElementById("roiCount"),
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

    function draw() {
      if (!state || !state.panorama_count) return;
      const current = pano();
      const quality = current.quality || {};
      els.counter.textContent = `${index + 1} / ${state.panorama_count}`;
      els.filename.textContent = current.pano_filename || "";
      els.source.textContent = current.source_pano_filename || current.source_status || "";
      els.scene.textContent = current.scene_pano_filename || "none";
      els.quality.textContent = [
        `obs ${quality.triangulated_observation_count ?? "n/a"}`,
        `pts ${quality.unique_point3D_count ?? "n/a"}`,
        `err ${formatNumber(quality.mean_reprojection_error_px)}`
      ].join("  ");
      els.selectedCount.textContent = `${current.selections.length}`;
      els.prev.disabled = index === 0;
      els.next.disabled = index >= state.panorama_count - 1;
      clearMarkers();
      els.panorama.onload = () => drawMarkers(pano());
      els.panorama.src = `/api/panorama-preview/${current.pano_index}`;
      drawSelections(current);
      if (els.panorama.complete) {
        requestAnimationFrame(() => drawMarkers(current));
      }
      setStatus("");
    }

    function clearMarkers() {
      els.markerLayer.innerHTML = "";
    }

    function drawMarkers(current) {
      clearMarkers();
      if (!current || !current.selections.length) return;
      const rect = els.panorama.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      for (const selection of current.selections) {
        const longitude = Number(selection.longitude);
        const latitude = Number(selection.latitude);
        if (!Number.isFinite(longitude) || !Number.isFinite(latitude)) continue;
        const x = clamp(((longitude + Math.PI) / (2 * Math.PI)) * rect.width, 0, rect.width);
        const y = clamp(((Math.PI / 2 - latitude) / Math.PI) * rect.height, 0, rect.height);
        const marker = document.createElement("div");
        marker.className = "selection-marker";
        marker.style.left = `${x}px`;
        marker.style.top = `${y}px`;
        marker.title = `sel-${String(selection.selection_index).padStart(3, "0")}`;
        els.markerLayer.appendChild(marker);
      }
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

    async function saveSelection() {
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
      draw();
      setStatus("Saved.");
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

    async function sampleRoiSelections(bounds) {
      const current = pano();
      const count = Math.max(1, Math.floor(Number(els.roiCount.value) || 1));
      setStatus(`Sampling ${count} ROI view(s)...`);
      const response = await fetch("/api/sample-roi-selections", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          pano_index: current.pano_index,
          longitude_min: bounds.longitudeMin,
          longitude_max: bounds.longitudeMax,
          latitude_min: bounds.latitudeMin,
          latitude_max: bounds.latitudeMax,
          count,
          render: renderOverrides()
        })
      });
      if (!response.ok) throw new Error(await response.text());
      const payload = await response.json();
      state = payload.state;
      draw();
      setStatus(`Sampled ${payload.created_count} ROI view(s).`);
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
      if (event.metaKey) {
        if (els.cmdSaveOnly.checked) {
          saveSelection().catch(error => setStatus(error.message));
        } else {
          preview()
            .then(() => saveSelection())
            .catch(error => setStatus(error.message));
        }
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

    els.prev.onclick = () => { index = Math.max(index - 1, 0); draw(); };
    els.next.onclick = () => { index = Math.min(index + 1, state.panorama_count - 1); draw(); };
    els.panorama.onclick = pickFromClick;
    els.panorama.onmousedown = startRoi;
    els.panorama.onmouseup = finishRoi;
    els.panorama.onmousemove = updatePanoramaPointer;
    els.panorama.onmouseleave = hideCursorLines;
    els.previewBtn.onclick = () => preview().catch(error => setStatus(error.message));
    els.saveBtn.onclick = () => saveSelection().catch(error => setStatus(error.message));
    els.resetPanoBtn.onclick = () => resetPanoSelections().catch(error => setStatus(error.message));
    els.resetCollectionBtn.onclick = () => resetCollectionSelections().catch(error => setStatus(error.message));
    window.addEventListener("keydown", event => {
      if (event.key === "ArrowLeft") els.prev.click();
      if (event.key === "ArrowRight") els.next.click();
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
