"""Constants for panorama reconstruction."""

from __future__ import annotations

import re

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
COLMAP_IMAGE_PAIR_ID_PRIME = 2_147_483_647
FILENAME_TIMESTAMP_PATTERN = re.compile(r"(?<!\d)(\d{10,})(?!\d)")
NATIVE_CAMERA_SETUP = "native_equirectangular"
