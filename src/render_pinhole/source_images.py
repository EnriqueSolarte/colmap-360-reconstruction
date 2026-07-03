"""Source panorama image discovery constants."""

from __future__ import annotations

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)
SOURCE_STATUS_MATCHED = "matched"
SOURCE_STATUS_MISSING = "missing"
