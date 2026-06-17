"""Selection storage helpers for manual pinhole rendering tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METADATA_FILENAME = "metadata.json"
SELECTIONS_FILENAME = "selections.json"
SELECTION_STORAGE_VERSION = 1


def load_metadata_with_selections(output_dir: Path) -> dict[str, Any]:
    metadata_path = output_dir / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    store = load_selection_store(output_dir)
    return merge_selection_store(metadata, store)


def write_metadata_with_sidecar(output_dir: Path, metadata: dict[str, Any]) -> None:
    metadata_path = output_dir / METADATA_FILENAME
    selections_path = output_dir / SELECTIONS_FILENAME
    metadata_without_selections, store = split_metadata_and_selection_store(metadata)
    metadata_path.write_text(
        json.dumps(metadata_without_selections, indent=2) + "\n",
        encoding="utf-8",
    )
    selections_path.write_text(
        json.dumps(store, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def migrate_metadata_to_sidecar(output_dir: Path) -> dict[str, Any]:
    metadata_path = output_dir / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    merged = merge_selection_store(metadata, load_selection_store(output_dir))
    write_metadata_with_sidecar(output_dir, merged)
    return load_metadata_with_selections(output_dir)


def load_selection_store(output_dir: Path) -> dict[str, Any]:
    selections_path = output_dir / SELECTIONS_FILENAME
    if not selections_path.is_file():
        return empty_selection_store()
    store = json.loads(selections_path.read_text(encoding="utf-8"))
    store.setdefault("schema_version", SELECTION_STORAGE_VERSION)
    store.setdefault("tool", "manual-pinhole-select")
    store.setdefault("selections_by_pano_index", {})
    store["selection_summary"] = selection_summary_from_store(store)
    return store


def merge_selection_store(
    metadata: dict[str, Any],
    store: dict[str, Any],
) -> dict[str, Any]:
    selections_by_pano_index = store.get("selections_by_pano_index", {})
    for panorama in metadata.get("panoramas", []):
        pano_index = str(int(panorama.get("pano_index", 0)))
        sidecar_selections = selections_by_pano_index.get(pano_index)
        if sidecar_selections is not None:
            panorama["selections"] = list(sidecar_selections)
        else:
            panorama["selections"] = list(panorama.get("selections", []))
    metadata["selection_summary"] = selection_summary_from_metadata(metadata)
    return metadata


def split_metadata_and_selection_store(
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = dict(metadata)
    panoramas = []
    selections_by_pano_index: dict[str, list[dict[str, Any]]] = {}
    for panorama in metadata.get("panoramas", []):
        clean_panorama = dict(panorama)
        selections = [
            compact_selection(selection)
            for selection in clean_panorama.pop("selections", [])
        ]
        pano_index = str(int(clean_panorama.get("pano_index", 0)))
        clean_panorama["selections"] = []
        clean_panorama["selection_count"] = len(selections)
        panoramas.append(clean_panorama)
        if selections:
            selections_by_pano_index[pano_index] = selections

    base["panoramas"] = panoramas
    base["selection_storage"] = {
        "type": "sidecar",
        "path": SELECTIONS_FILENAME,
        "schema_version": SELECTION_STORAGE_VERSION,
    }
    store = {
        "schema_version": SELECTION_STORAGE_VERSION,
        "tool": "manual-pinhole-select",
        "selections_by_pano_index": selections_by_pano_index,
    }
    summary = selection_summary_from_store(store)
    base["selection_summary"] = summary
    store["selection_summary"] = summary
    return base, store


def compact_selection(selection: dict[str, Any]) -> dict[str, Any]:
    compact = dict(selection)
    if compact.get("render_status") != "rendered":
        compact.pop("pinhole_camera_pose", None)
    return compact


def empty_selection_store() -> dict[str, Any]:
    return {
        "schema_version": SELECTION_STORAGE_VERSION,
        "tool": "manual-pinhole-select",
        "selections_by_pano_index": {},
        "selection_summary": {
            "selected_panorama_count": 0,
            "selection_count": 0,
        },
    }


def selection_summary_from_metadata(metadata: dict[str, Any]) -> dict[str, int]:
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


def selection_summary_from_store(store: dict[str, Any]) -> dict[str, int]:
    selection_lists = store.get("selections_by_pano_index", {}).values()
    counts = [len(selections) for selections in selection_lists if selections]
    return {
        "selected_panorama_count": len(counts),
        "selection_count": sum(counts),
    }


def read_collection_snapshot(output_dir: Path) -> dict[str, str | None]:
    selections_path = output_dir / SELECTIONS_FILENAME
    return {
        "metadata": (output_dir / METADATA_FILENAME).read_text(encoding="utf-8"),
        "selections": (
            selections_path.read_text(encoding="utf-8")
            if selections_path.is_file()
            else None
        ),
    }


def restore_collection_snapshot(
    output_dir: Path,
    snapshot: dict[str, str | None] | str,
) -> dict[str, Any]:
    if isinstance(snapshot, str):
        (output_dir / METADATA_FILENAME).write_text(snapshot, encoding="utf-8")
        return json.loads(snapshot)
    else:
        (output_dir / METADATA_FILENAME).write_text(
            snapshot["metadata"] or "{}",
            encoding="utf-8",
        )
        selections_path = output_dir / SELECTIONS_FILENAME
        if snapshot.get("selections") is None:
            selections_path.unlink(missing_ok=True)
        else:
            selections_path.write_text(snapshot["selections"] or "", encoding="utf-8")
    return load_metadata_with_selections(output_dir)
