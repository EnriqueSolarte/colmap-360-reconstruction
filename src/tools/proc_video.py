#!/usr/bin/env python3
"""Console entry point for the frame extraction shell script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    script_path = Path(__file__).with_name("proc_video.sh")
    return subprocess.call([str(script_path), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
