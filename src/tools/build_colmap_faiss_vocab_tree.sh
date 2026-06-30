#!/usr/bin/env bash
set -Eeuo pipefail

DEFAULT_DATABASE="/data/itri-3d-reconstruction/scene-reconstruction/vocab-tree-2026-06-30-001/colmap/database.db"
DEFAULT_OUTPUT="/data/itri-3d-reconstruction/colmap-vocab-tree-faiss/itri_vocab_tree_words32K.faiss.bin"

DATABASE_PATH="$DEFAULT_DATABASE"
OUTPUT_PATH="$DEFAULT_OUTPUT"
VISUAL_WORDS=32768
MAX_DESCRIPTORS=1000000
ITERATIONS=15
ROUNDS=1
CHECKS=256
THREADS=-1
TMP_DIR="${TMPDIR:-/tmp}"
COPY_DATABASE=1
KEEP_TEMP=0
OVERWRITE=0
VALIDATE=1
VALIDATION_NUM_IMAGES=1
VALIDATION_NEIGHBORS=1
VALIDATION_CHECKS=64
LOG_PATH=""
VALIDATION_OUTPUT=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Build a FAISS-compatible COLMAP vocabulary tree from an existing database.

Defaults are set for the vocab-tree-2026-06-30-001 workspace. The script copies
the SQLite database to /tmp by default before building so long-running COLMAP
jobs do not create lock/journal files beside the source database.

This script trains a new FAISS tree with `colmap vocab_tree_builder`; it does
not convert an existing legacy `vocab_tree_*.bin` file. To convert a legacy
FLANN tree, use a COLMAP build that includes `vocab_tree_upgrader`.

Usage:
  src/tools/build_colmap_faiss_vocab_tree.sh [options]

Options:
  --database PATH              COLMAP SQLite database.db to train from.
  --output PATH                Output FAISS vocabulary tree path.
  --visual-words N             Number of visual words. Default: 32768.
  --max-descriptors N          Training descriptor sample size. Default: 1000000.
  --iterations N               K-means iterations. Default: 15.
  --rounds N                   K-means rounds. Default: 1.
  --checks N                   Builder checks. Default: 256.
  --threads N                  COLMAP thread count. Default: -1.
  --tmp-dir PATH               Directory for copied database. Default: ${TMPDIR:-/tmp}.
  --log PATH                   Log file. Default: <output>.log.
  --validation-output PATH     Retriever validation output. Default: <output>.retrieval.txt.
  --validation-num-images N    Retrieved images per query during validation. Default: 1.
  --validation-neighbors N     Visual-word neighbors during validation. Default: 1.
  --validation-checks N        Retriever checks during validation. Default: 64.
  --no-copy-database           Read the database in place instead of copying it.
  --keep-temp                  Keep the copied temp database after exit.
  --skip-validation            Only build; do not run vocab_tree_retriever.
  --overwrite                  Replace existing output/log/validation files.
  --dry-run                    Print the resolved commands without running them.
  -h, --help                   Show this help.

Example:
  nohup src/tools/build_colmap_faiss_vocab_tree.sh \
    --visual-words 32768 \
    --max-descriptors 1000000 \
    --output /data/itri-3d-reconstruction/colmap-vocab-tree-faiss/itri_vocab_tree_words32K.faiss.bin \
    > /data/itri-3d-reconstruction/colmap-vocab-tree-faiss/build_words32K.nohup.log 2>&1 &
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

positive_integer() {
  [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -gt 0 ]]
}

is_sqlite_database() {
  local path="$1"
  head -c 15 "$path" 2>/dev/null | cmp -s - <(printf 'SQLite format 3')
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --database)
      DATABASE_PATH="${2:?missing value for --database}"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="${2:?missing value for --output}"
      shift 2
      ;;
    --visual-words)
      VISUAL_WORDS="${2:?missing value for --visual-words}"
      shift 2
      ;;
    --max-descriptors)
      MAX_DESCRIPTORS="${2:?missing value for --max-descriptors}"
      shift 2
      ;;
    --iterations)
      ITERATIONS="${2:?missing value for --iterations}"
      shift 2
      ;;
    --rounds)
      ROUNDS="${2:?missing value for --rounds}"
      shift 2
      ;;
    --checks)
      CHECKS="${2:?missing value for --checks}"
      shift 2
      ;;
    --threads)
      THREADS="${2:?missing value for --threads}"
      shift 2
      ;;
    --tmp-dir)
      TMP_DIR="${2:?missing value for --tmp-dir}"
      shift 2
      ;;
    --log)
      LOG_PATH="${2:?missing value for --log}"
      shift 2
      ;;
    --validation-output)
      VALIDATION_OUTPUT="${2:?missing value for --validation-output}"
      shift 2
      ;;
    --validation-num-images)
      VALIDATION_NUM_IMAGES="${2:?missing value for --validation-num-images}"
      shift 2
      ;;
    --validation-neighbors)
      VALIDATION_NEIGHBORS="${2:?missing value for --validation-neighbors}"
      shift 2
      ;;
    --validation-checks)
      VALIDATION_CHECKS="${2:?missing value for --validation-checks}"
      shift 2
      ;;
    --no-copy-database)
      COPY_DATABASE=0
      shift
      ;;
    --keep-temp)
      KEEP_TEMP=1
      shift
      ;;
    --skip-validation)
      VALIDATE=0
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

require_command colmap
[[ -f "$DATABASE_PATH" ]] || die "database does not exist: $DATABASE_PATH"
if ! is_sqlite_database "$DATABASE_PATH"; then
  die "--database must point to a COLMAP SQLite database.db, not a vocabulary tree .bin file.
The file is not SQLite: $DATABASE_PATH

For your current workspace, use:
  --database /data/itri-3d-reconstruction/scene-reconstruction/vocab-tree-2026-06-30-001/colmap/database.db

This script trains a new FAISS tree. It cannot convert:
  /data/itri-3d-reconstruction/colmap-vocab-tree/vocab_tree_flickr100K_words32K.bin

To convert that legacy FLANN tree directly, run COLMAP's vocab_tree_upgrader from the specific older source build mentioned in the report."
fi
positive_integer "$VISUAL_WORDS" || die "--visual-words must be a positive integer"
positive_integer "$MAX_DESCRIPTORS" || die "--max-descriptors must be a positive integer"
positive_integer "$ITERATIONS" || die "--iterations must be a positive integer"
positive_integer "$ROUNDS" || die "--rounds must be a positive integer"
positive_integer "$CHECKS" || die "--checks must be a positive integer"
positive_integer "$VALIDATION_NUM_IMAGES" || die "--validation-num-images must be a positive integer"
positive_integer "$VALIDATION_NEIGHBORS" || die "--validation-neighbors must be a positive integer"
positive_integer "$VALIDATION_CHECKS" || die "--validation-checks must be a positive integer"

OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
mkdir -p "$OUTPUT_DIR"

if [[ -z "$LOG_PATH" ]]; then
  LOG_PATH="${OUTPUT_PATH}.build.log"
fi
if [[ -z "$VALIDATION_OUTPUT" ]]; then
  VALIDATION_OUTPUT="${OUTPUT_PATH}.retrieval.txt"
fi
mkdir -p "$(dirname "$LOG_PATH")"
mkdir -p "$(dirname "$VALIDATION_OUTPUT")"

if [[ "$OVERWRITE" -eq 0 ]]; then
  [[ ! -e "$OUTPUT_PATH" ]] || die "output already exists: $OUTPUT_PATH (use --overwrite)"
  [[ ! -e "$LOG_PATH" ]] || die "log already exists: $LOG_PATH (use --overwrite)"
  if [[ "$VALIDATE" -eq 1 ]]; then
    [[ ! -e "$VALIDATION_OUTPUT" ]] || die "validation output already exists: $VALIDATION_OUTPUT (use --overwrite)"
  fi
else
  rm -f "$OUTPUT_PATH" "$LOG_PATH"
  if [[ "$VALIDATE" -eq 1 ]]; then
    rm -f "$VALIDATION_OUTPUT"
  fi
fi

TEMP_DATABASE=""
WORK_DATABASE="$DATABASE_PATH"
cleanup() {
  if [[ -n "$TEMP_DATABASE" && "$KEEP_TEMP" -eq 0 ]]; then
    rm -f "$TEMP_DATABASE" "$TEMP_DATABASE-shm" "$TEMP_DATABASE-wal" "$TEMP_DATABASE-journal"
  fi
}
trap cleanup EXIT

log() {
  printf '%s\n' "$*" | tee -a "$LOG_PATH"
}

run_logged() {
  log "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@" 2>&1 | tee -a "$LOG_PATH"
  fi
}

if [[ "$COPY_DATABASE" -eq 1 ]]; then
  mkdir -p "$TMP_DIR"
  TEMP_DATABASE="${TMP_DIR%/}/colmap_vocab_tree_build.$$.db"
  log "copying database:"
  log "  from: $DATABASE_PATH"
  log "  to:   $TEMP_DATABASE"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    cp "$DATABASE_PATH" "$TEMP_DATABASE"
  fi
  WORK_DATABASE="$TEMP_DATABASE"
fi

log "COLMAP version:"
run_logged colmap version
log "building FAISS vocabulary tree:"
log "  database:        $WORK_DATABASE"
log "  output:          $OUTPUT_PATH"
log "  visual_words:    $VISUAL_WORDS"
log "  max_descriptors: $MAX_DESCRIPTORS"
log "  iterations:      $ITERATIONS"
log "  rounds:          $ROUNDS"
log "  checks:          $CHECKS"
log "  threads:         $THREADS"

run_logged \
  colmap vocab_tree_builder \
    --database_path "$WORK_DATABASE" \
    --vocab_tree_path "$OUTPUT_PATH" \
    --num_visual_words "$VISUAL_WORDS" \
    --num_iterations "$ITERATIONS" \
    --num_rounds "$ROUNDS" \
    --num_checks "$CHECKS" \
    --max_num_descriptors "$MAX_DESCRIPTORS" \
    --num_threads "$THREADS" \
    --log_target stderr

if [[ "$DRY_RUN" -eq 0 ]]; then
  [[ -s "$OUTPUT_PATH" ]] || die "tree was not written or is empty: $OUTPUT_PATH"
fi

if [[ "$VALIDATE" -eq 1 ]]; then
  log "validating tree with COLMAP vocab_tree_retriever:"
  run_logged \
    colmap vocab_tree_retriever \
      --database_path "$WORK_DATABASE" \
      --vocab_tree_path "$OUTPUT_PATH" \
      --output_index_path "$VALIDATION_OUTPUT" \
      --num_images "$VALIDATION_NUM_IMAGES" \
      --num_neighbors "$VALIDATION_NEIGHBORS" \
      --num_checks "$VALIDATION_CHECKS" \
      --num_threads "$THREADS" \
      --log_target stderr
fi

log "done"
log "tree: $OUTPUT_PATH"
log "log:  $LOG_PATH"
if [[ "$VALIDATE" -eq 1 ]]; then
  log "validation output: $VALIDATION_OUTPUT"
fi
