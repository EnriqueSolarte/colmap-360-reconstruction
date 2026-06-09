#!/usr/bin/env bash
set -e
# v2026-06-02  14:56:23 by EnriqueSolarte
# Extracts frames from an input video using ffmpeg, saving every Nth frame
# into a directory named after the video (without extension).
# The optional second argument sets N (default 10).
# Example: ./proc_video.sh /path/to/video.mp4 5
# input video

usage() {
  echo "Usage: ${BASH_SOURCE[0]} -v VIDEO_INPUT"
  echo
  echo "Options:"
  echo "  -v, --video-input VIDEO_INPUT   Video input. Required."
  echo "  -h, --help                  Show this help message."
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -v|--video-input)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for $1" >&2
        return 1 2>/dev/null || exit 1
      fi
      VIDEO=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      return 0 2>/dev/null || exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      return 1 2>/dev/null || exit 1
      ;;
  esac
done
if [ -z "${VIDEO}" ]; then
    echo "Missing value for $1" >&2
    usage >&2
    return 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg is required but was not found on PATH." >&2
    echo "Install it with your system package manager, for example: apt-get install ffmpeg" >&2
    return 1 2>/dev/null || exit 1
fi

# VIDEO=$1
# compute base name without extension
BASENAME="${VIDEO%.*}"
PARENT_NAME="$(basename "$BASENAME")"

# output directory
OUTDIR="$BASENAME"

SKIP="${2:-1}"   # if user passes a second argument use that, otherwise default=10

# make sure output directory exists
mkdir -p "$OUTDIR"

# extract all frames
# you can change "png" to "jpg" or other format
ffmpeg -i "$VIDEO" -vf "select='not(mod(n\,${SKIP}))'" -vsync vfr "$OUTDIR/${PARENT_NAME}_%08d.png"

echo "Extracted frames into $OUTDIR/"
