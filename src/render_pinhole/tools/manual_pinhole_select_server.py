"""Browser-based selector for manual pinhole render directions."""

from __future__ import annotations

import io
import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import degrees, pi
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
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
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
    body {
      margin: 0;
      min-height: 100vh;
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
      min-height: 100vh;
    }
    .stage {
      min-width: 0;
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
    #panorama {
      max-width: 100%;
      max-height: calc(100vh - 82px);
      object-fit: contain;
      border: 1px solid var(--line);
      cursor: crosshair;
    }
    aside {
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      min-height: 100vh;
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
    .list {
      min-height: 0;
      overflow: auto;
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
        min-height: auto;
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
        <img id="panorama" alt="">
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
          <label>FOV X rad<input id="fovX" type="number" step="0.001"></label>
          <label>FOV Y rad<input id="fovY" type="number" step="0.001"></label>
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
        <div id="status" class="status"></div>
      </section>
      <section class="list" id="selectionList"></section>
    </aside>
  </main>
  <script>
    const els = {
      panorama: document.getElementById("panorama"),
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
      fovX: document.getElementById("fovX"),
      fovY: document.getElementById("fovY"),
      interpolation: document.getElementById("interpolation"),
      preview: document.getElementById("preview"),
      previewBtn: document.getElementById("previewBtn"),
      saveBtn: document.getElementById("saveBtn"),
      status: document.getElementById("status"),
      selectionList: document.getElementById("selectionList")
    };
    let state = null;
    let index = 0;

    function radians(value) { return Number(value) * Math.PI / 180; }
    function degrees(value) { return Number(value) * 180 / Math.PI; }
    function pano() { return state.panoramas[index]; }
    function renderOverrides() {
      return {
        width: Number(els.width.value),
        height: Number(els.height.value),
        fov_x: Number(els.fovX.value),
        fov_y: Number(els.fovY.value),
        interpolation: els.interpolation.value
      };
    }
    function setStatus(message) { els.status.textContent = message || ""; }

    async function refresh() {
      const response = await fetch("/api/state");
      state = await response.json();
      const defaults = state.render_defaults || {};
      els.width.value = defaults.width || 1280;
      els.height.value = defaults.height || 960;
      els.fovX.value = defaults.fov_x || 1.221730476;
      els.fovY.value = defaults.fov_y || 0.967145867;
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
      els.panorama.src = `/api/panorama/${current.pano_index}?t=${Date.now()}`;
      drawSelections(current);
      setStatus("");
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
      if (selection.fov_x) els.fovX.value = selection.fov_x;
      if (selection.fov_y) els.fovY.value = selection.fov_y;
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

    function pickFromClick(event) {
      const rect = els.panorama.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const longitude = (x / rect.width) * 2 * Math.PI - Math.PI;
      const latitude = Math.PI / 2 - (y / rect.height) * Math.PI;
      els.lonDeg.value = degrees(longitude).toFixed(2);
      els.latDeg.value = degrees(latitude).toFixed(2);
      preview().catch(error => setStatus(error.message));
    }

    function formatNumber(value) {
      return value === null || value === undefined ? "n/a" : Number(value).toFixed(2);
    }

    els.prev.onclick = () => { index = Math.max(index - 1, 0); draw(); };
    els.next.onclick = () => { index = Math.min(index + 1, state.panorama_count - 1); draw(); };
    els.panorama.onclick = pickFromClick;
    els.previewBtn.onclick = () => preview().catch(error => setStatus(error.message));
    els.saveBtn.onclick = () => saveSelection().catch(error => setStatus(error.message));
    window.addEventListener("keydown", event => {
      if (event.key === "ArrowLeft") els.prev.click();
      if (event.key === "ArrowRight") els.next.click();
      if (event.key.toLowerCase() === "s") els.saveBtn.click();
    });
    refresh().catch(error => setStatus(error.message));
  </script>
</body>
</html>
"""
