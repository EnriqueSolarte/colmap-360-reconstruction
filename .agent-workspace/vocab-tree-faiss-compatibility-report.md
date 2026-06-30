Date: 2026-06-30
Project version: v0.1.0
Description: Documents the FLANN-vs-FAISS vocabulary tree incompatibility blocking sequential-vocabtree clustering.

# Vocab Tree FAISS Compatibility Issue Report

## Executive Summary

The `colmap-panorama-clusters` command failed because the vocabulary tree file
is in the legacy COLMAP FLANN format, while the installed `pycolmap` in this
Docker environment is `4.1.0` and expects the newer FAISS vocabulary-tree
format.

This is not a bug in the keyframe clustering logic and not a typo in the command.
It is a binary format mismatch between:

- the downloaded official prebuilt tree:
  `vocab_tree_flickr100K_words256K.bin`;
- the current COLMAP/pycolmap retrieval backend, which now reads FAISS indexes.

Best next step: use the other Docker where COLMAP was built from source to run
COLMAP's `vocab_tree_upgrader` command at the specific commit named in the error:

```text
c7a58462b813e406c304a9dafb475b87036924cf
```

The upgrader should convert:

```text
legacy FLANN tree -> FAISS-compatible tree
```

Then rerun `colmap-panorama-clusters` with the converted output file.

## User Command That Failed

The failed command was:

```bash
colmap-panorama-clusters \
  -w /data/itri-3d-reconstruction/scene-reconstruction/vocab-tree-2026-06-30-001/colmap/ \
  --vocab-tree-path /data/itri-3d-reconstruction/colmap-vocab-tree/vocab_tree_flickr100K_words256K.bin \
  --keyframe-backbone-policy none
```

The workspace path points to a panorama COLMAP workspace that already contains
feature extraction output:

```text
/data/itri-3d-reconstruction/scene-reconstruction/vocab-tree-2026-06-30-001/colmap/
```

The selected vocabulary tree was:

```text
/data/itri-3d-reconstruction/colmap-vocab-tree/vocab_tree_flickr100K_words256K.bin
```

That file came from the official prebuilt COLMAP Flickr100K vocabulary tree
download set.

## Observed Error

COLMAP started vocabulary-tree pair generation:

```text
I20260630 02:26:54.050169 pairing.cc:246] Generating image pairs with vocabulary tree...
I20260630 02:26:54.095626 pairing.cc:367] Indexing image [1/1224]
```

Then it failed:

```text
E20260630 02:26:54.197521 visual_index.cc:690] Check failed: file_version == 1 || file_version == 2 Failed to read faiss index.
This may be caused by reading a legacy flann-based index, because COLMAP switched from flann to faiss in May 2025.
If you want to upgrade your existing flann-based index to faiss, you can check out COLMAP commit c7a58462b813e406c304a9dafb475b87036924cf and apply the vocab_tree_upgrader command.
```

The Python traceback ended at our pycolmap adapter:

```text
File "/workspace/src/colmap/panorama/pycolmap_adapter.py", line 239, in build_vocabtree_candidate_pairs
  pair_generator = pycolmap_module.VocabTreePairGenerator(...)

ValueError: [visual_index.cc:690] Check failed: file_version == 1 || file_version == 2 Failed to read faiss index.
```

## Current Local Environment Evidence

In this Docker environment:

```bash
python - <<'PY'
import pycolmap
print(pycolmap.__version__)
PY
```

Observed:

```text
4.1.0
```

Also:

```bash
colmap version
```

Observed:

```text
colmap: command not found
```

Meaning:

- this Docker has `pycolmap`;
- it does not have the `colmap` CLI binary available in `PATH`;
- upgrading the vocab tree must be done in the other Docker where COLMAP was
  built from source, or by adding a COLMAP binary here.

## Root Cause

The prebuilt vocabulary tree is legacy FLANN format.

The installed `pycolmap 4.1.0` expects FAISS vocabulary-tree indexes. When
`pycolmap.VocabTreePairGenerator` tries to read the old tree, COLMAP checks the
FAISS file version and fails.

This is why the failure happens immediately during vocab-tree indexing, before
our clustering code receives any candidate image pairs.

The relevant chain is:

```text
colmap-panorama-clusters
  -> cli_clusters.py
  -> cluster_pipeline.build_keyframe_pair_plan
  -> pycolmap_adapter.build_vocabtree_candidate_pairs
  -> pycolmap.VocabTreePairGenerator
  -> COLMAP retrieval::VisualIndex::Read
  -> fails because input tree is legacy FLANN, not FAISS
```

## Why The Downloaded Tree Was Misleading

The official COLMAP download page still lists the prebuilt Flickr100K vocabulary
trees at release `3.11.1`. There are two filename families:

Legacy FLANN-format assets:

```text
https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_flickr100K_words32K.bin
https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_flickr100K_words256K.bin
https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_flickr100K_words1M.bin
```

FAISS-format assets:

```text
https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words32K.bin
https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin
https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words1M.bin
```

The legacy files are useful for older
COLMAP builds, but they can be incompatible with newer COLMAP/pycolmap builds
after the May 2025 FAISS migration.

The best prebuilt-tree fix is to download the `vocab_tree_faiss_...` asset from
release `3.11.1`, not to change the release number to `4.1.0`.

Confirmed result:

```text
https://github.com/colmap/colmap/releases/download/4.1.0/vocab_tree_flickr100K_words256K.bin
```

returned HTTP `404`; the working replacement is the FAISS-prefixed filename in
the `3.11.1` release.

## What We Changed In This Repo

The repo was patched so future failures are less cryptic.

### Adapter Error Translation

File:

```text
src/colmap/panorama/pycolmap_adapter.py
```

Now catches the specific `"Failed to read faiss index"` `ValueError` from
pycolmap and raises a clearer message explaining:

- the tree is likely legacy FLANN format;
- the installed COLMAP expects FAISS;
- the user should use a FAISS-compatible tree, run `vocab_tree_upgrader`, or
  temporarily run interval keyframes.

### Downloader Format Selection

File:

```text
src/tools/download_colmap_vocab_tree.py
```

Now defaults to the FAISS-format asset family and accepts `--format legacy` for
older COLMAP builds. The FAISS download uses URLs like:

```text
https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words256K.bin
```

Tests were updated and the full suite passed:

```text
195 passed
```

## Recommended Fix Path

Download the prebuilt FAISS Flickr100K vocabulary tree first.

This is the best first option because:

- it avoids training a new tree;
- it avoids needing the temporary `vocab_tree_upgrader` command;
- it preserves the known Flickr100K prebuilt vocabulary;
- it should be enough to unblock our sequential-vocabtree keyframe experiment;
- it should be compatible with the FAISS-based reader used by pycolmap.

Use the source-built COLMAP Docker and `vocab_tree_upgrader` only if the
FAISS-prefixed asset cannot be downloaded or fails validation. Building a custom
tree is useful as a fallback, but it is more expensive and adds more variables
to the experiment.

Recommended download:

```bash
tools-download-colmap-vocab-tree \
  --format faiss \
  --variant 256k \
  --destination-dir /data/itri-3d-reconstruction/colmap-vocab-tree
```

Then rerun clustering with:

```bash
colmap-panorama-clusters \
  -w /data/itri-3d-reconstruction/scene-reconstruction/vocab-tree-2026-06-30-001/colmap/ \
  --vocab-tree-path /data/itri-3d-reconstruction/colmap-vocab-tree/vocab_tree_faiss_flickr100K_words256K.bin \
  --keyframe-selection sequential-vocabtree \
  --keyframe-backbone-policy none
```

## Source-Built Docker Upgrader Plan

### Step 1: Check Whether The Upgrader Already Exists

Inside the source-built COLMAP Docker:

```bash
colmap help | grep vocab_tree
```

Look for:

```text
vocab_tree_upgrader
```

If it exists, run:

```bash
colmap vocab_tree_upgrader -h
```

Expected options from the COLMAP source at the named commit:

```text
--input_path
--output_path
```

### Step 2: If The Upgrader Exists, Convert The Tree

Recommended output directory:

```bash
mkdir -p /data/itri-3d-reconstruction/colmap-vocab-tree-faiss
```

Run:

```bash
colmap vocab_tree_upgrader \
  --input_path /data/itri-3d-reconstruction/colmap-vocab-tree/vocab_tree_flickr100K_words256K.bin \
  --output_path /data/itri-3d-reconstruction/colmap-vocab-tree-faiss/vocab_tree_flickr100K_words256K.faiss.bin
```

Do not overwrite the original file. Keep both:

```text
colmap-vocab-tree/
  vocab_tree_flickr100K_words256K.bin

colmap-vocab-tree-faiss/
  vocab_tree_flickr100K_words256K.faiss.bin
```

This makes it obvious which file is legacy and which file is converted.

### Step 3: If The Upgrader Does Not Exist

The current COLMAP `main` source checked during debugging did not list
`vocab_tree_upgrader` in `colmap help`. The error specifically says to check out:

```text
c7a58462b813e406c304a9dafb475b87036924cf
```

In the source-built Docker, use a separate clone or a temporary worktree so the
normal source build is not disturbed:

```bash
cd /tmp
git clone https://github.com/colmap/colmap.git colmap-vocab-upgrader
cd /tmp/colmap-vocab-upgrader
git checkout c7a58462b813e406c304a9dafb475b87036924cf
```

Build COLMAP at that commit using the same dependency setup as your existing
source Docker. Example shape:

```bash
mkdir -p build
cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release
ninja colmap
```

Depending on the Docker/build layout, the built binary may be one of:

```text
/tmp/colmap-vocab-upgrader/build/src/colmap/exe/colmap
/tmp/colmap-vocab-upgrader/build/colmap
```

Use whichever binary exists.

Then run:

```bash
/tmp/colmap-vocab-upgrader/build/src/colmap/exe/colmap vocab_tree_upgrader \
  --input_path /data/itri-3d-reconstruction/colmap-vocab-tree/vocab_tree_flickr100K_words256K.bin \
  --output_path /data/itri-3d-reconstruction/colmap-vocab-tree-faiss/vocab_tree_flickr100K_words256K.faiss.bin
```

If the binary path differs:

```bash
find /tmp/colmap-vocab-upgrader/build -type f -name colmap -executable
```

Then use the found path.

## Confirmed Upgrader Implementation Details

The error message named commit:

```text
c7a58462b813e406c304a9dafb475b87036924cf
```

At that commit:

```text
src/colmap/exe/colmap.cc
```

registers:

```cpp
commands.emplace_back("vocab_tree_upgrader", &colmap::RunVocabTreeUpgrader);
```

And:

```text
src/colmap/exe/vocab_tree.cc
```

implements:

```cpp
int RunVocabTreeUpgrader(int argc, char** argv) {
  std::string input_path;
  std::string output_path;

  OptionManager options;
  options.AddRequiredOption("input_path", &input_path);
  options.AddRequiredOption("output_path", &output_path);
  options.Parse(argc, argv);

  std::unique_ptr<retrieval::VisualIndex> index =
      retrieval::VisualIndex::Read(input_path, /*legacy_flann=*/true);
  index->Write(output_path);

  return EXIT_SUCCESS;
}
```

Interpretation:

- it reads the old tree using `legacy_flann=true`;
- it writes the output using the newer default format;
- the output should be readable by newer FAISS-based COLMAP/pycolmap.

## Validation After Upgrade

After the converted file exists, rerun our clustering command with the converted
tree:

```bash
colmap-panorama-clusters \
  -w /data/itri-3d-reconstruction/scene-reconstruction/vocab-tree-2026-06-30-001/colmap/ \
  --vocab-tree-path /data/itri-3d-reconstruction/colmap-vocab-tree-faiss/vocab_tree_flickr100K_words256K.faiss.bin \
  --keyframe-selection sequential-vocabtree \
  --keyframe-backbone-policy none
```

Expected behavior:

- COLMAP should index all `1224` images without the FAISS read error.
- `colmap-panorama-clusters` should write:
  - `match_pairs.txt`;
  - `panorama_keyframe_cluster_report.json`.
- The report should show:
  - `options.effective_pairing == "sequential-vocabtree"`;
  - nonzero `vocabtree_pair_count`;
  - selected keyframes;
  - clusters.

Quick inspection:

```bash
python - <<'PY'
import json
from pathlib import Path

report_path = Path("/data/itri-3d-reconstruction/scene-reconstruction/vocab-tree-2026-06-30-001/colmap/panorama_keyframe_cluster_report.json")
report = json.loads(report_path.read_text())
print("effective:", report["options"]["effective_pairing"])
print("images:", report["generated_image_count"])
print("keyframes:", report["keyframe_count"])
print("clusters:", report["cluster_count"])
print("pairs:", report["options"]["matching_pair_count"])
print("vocabtree pairs:", report["options"]["vocabtree_pair_count"])
print("backbone pairs:", report["options"]["sequential_backbone_pair_count"])
PY
```

If the command falls back to interval mode, the report will show:

```text
effective_pairing: interval_fallback
```

That would mean the tree loaded, but vocabtree generated no usable candidate
pairs for our image IDs. That is a different issue from the current FAISS/FLANN
file-format failure.

## If Upgrade Fails

If `vocab_tree_upgrader` fails, record:

- exact COLMAP version;
- exact command;
- full stderr/stdout;
- input file size:

```bash
ls -lh /data/itri-3d-reconstruction/colmap-vocab-tree/vocab_tree_flickr100K_words256K.bin
```

- whether the input file is complete or partially downloaded;
- whether the source-built Docker has FAISS enabled.

Then use interval mode as a temporary unblocker:

```bash
colmap-panorama-clusters \
  -w /data/itri-3d-reconstruction/scene-reconstruction/vocab-tree-2026-06-30-001/colmap/ \
  --keyframe-selection interval \
  --keyframe-backbone-policy none
```

This does not test the vocabtree idea, but it validates the downstream pair-list
path.

## Custom Vocab Tree Alternative

If upgrading the old tree is not possible, build a new FAISS-compatible tree
with the source-built COLMAP that matches the current pycolmap generation.

Example:

```bash
colmap vocab_tree_builder \
  --database_path /data/itri-3d-reconstruction/scene-reconstruction/vocab-tree-2026-06-30-001/colmap/database.db \
  --vocab_tree_path /data/itri-3d-reconstruction/colmap-vocab-tree-faiss/itri_vocab_tree_words256K.faiss.bin \
  --num_visual_words 256000 \
  --num_iterations 11 \
  --num_checks 256 \
  --num_threads -1 \
  --num_rounds 1 \
  --max_num_images -1
```

Notes:

- exact defaults may vary by COLMAP version;
- `--num_visual_words 256000` roughly corresponds to the 256K official tree;
- using this dataset's own features trains a dataset-specific tree;
- dataset-specific retrieval may work well for this scene, but it is less
  general than the Flickr100K tree;
- for a reusable ITRI tree, train on a larger representative set of ITRI
  panorama/image data, not only this one sequence.

## Recommendation

Use this order:

1. Try `vocab_tree_upgrader` in the source-built COLMAP Docker.
2. Validate the converted file with `colmap-panorama-clusters`.
3. If upgrade is blocked, build a custom tree with the current source-built
   COLMAP.
4. Use interval keyframes only as a temporary pipeline/debug fallback.

The upgrader is the best immediate move because it changes only the tree format,
not the retrieval vocabulary itself. That keeps the experiment focused on the
keyframe/vocabtree clustering behavior rather than introducing a newly trained
tree at the same time.

## Related Repo Files

Key repo files involved:

- `src/colmap/panorama/cli_clusters.py`
- `src/colmap/panorama/cluster_pipeline.py`
- `src/colmap/panorama/keyframe_matching.py`
- `src/colmap/panorama/pycolmap_adapter.py`
- `src/tools/download_colmap_vocab_tree.py`
- `.agent-workspace/plan-for-key-frame-based-solution.md`

## References Checked

- COLMAP official download page:
  `https://demuc.de/colmap/`
- COLMAP GitHub releases:
  `https://github.com/colmap/colmap/releases`
- Commit named by the error:
  `c7a58462b813e406c304a9dafb475b87036924cf`
- Source file at that commit:
  `src/colmap/exe/vocab_tree.cc`
- Source file at that commit:
  `src/colmap/exe/colmap.cc`
