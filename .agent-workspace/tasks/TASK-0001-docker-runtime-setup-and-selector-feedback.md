# TASK-0001 — Docker runtime setup and selector feedback

```text
Protocol: amber@v3
Record schema: 2
Status: ready-to-merge
Owner: Codex
Parallel-safe: no
Parallel approval: not-required
Plan source: .agent-workspace/plans/TASK-0001-docker-runtime-setup-and-selector-feedback.md
Current milestone: M5
Last verified commit: 17a63211630dad734e3f2349805685a560cb0671; M4 selector-workflow commit
Definition approval: approved 2026-08-25
Parent task: none
Parent milestone: none
```

## Objective

Make the Docker image ready to run this project, consistently mount a
user-configured host data directory at `/data`, document the container workflow,
and make interactive selector startup observable.

## Acceptance criteria

- [x] The Docker image installs `.[test,colmap,visualize]` during image build.
- [x] The build has access to the repository source needed for that install.
- [x] The user-configured `DATA` host directory mounts at `/data` without
  changing `docker_run.sh` to derive it from the user's home directory.
- [x] README container examples use a valid container-name variable and label
  `/data/1782890858308` as an example only.
- [x] README documents the verified local FAISS vocabulary-tree asset at its
  `/workspace/data/...` container path.
- [x] The selector start command visibly reports its URL in the documented
  Docker invocation.
- [x] README starts with a practical Docker-first usage path and defers
  detailed reference material and optional extras to the end.

## Scope

In scope: Docker build context and image dependency installation; existing
`docker_run.sh` data mount semantics; `env` configuration guidance; README
container examples; selector startup feedback when required to make the
documented invocation observable.

Out of scope: changing the user's configured `DATA` path automatically,
changing reconstruction algorithms, or modifying any dataset contents.

## Affected areas and conflict assessment

- `.gh-rule.md`, `docker/dockerfile.dev-container`, `docker/docker_build.sh`,
  `docker/docker_run.sh`, `env`, `README.md`, and possibly
  `src/render_pinhole/tools/manual_pinhole_select_server.py` and focused tests.
- Existing uncommitted edits touch `README.md`, `docker/docker_run.sh`, and
  `env`; M1 must preserve and reconcile them before implementation.
- One canonical checkout is shared, so this task is not parallel-safe.

## Repository scope

| Repository | Target branch | Task branch | Worktree | Baseline / validation |
| --- | --- | --- | --- | --- |
| `.` | `dev` | `task/TASK-0001-docker-runtime-setup-and-selector-feedback` | `/home/kike/ws/scene-reconstruction-ws/colmap-reconstruction-ws/colmap-reconstruction-ws` | `6dee2e5`; Amber validator passed 2026-08-25 |

## Plan

[`.agent-workspace/plans/TASK-0001-docker-runtime-setup-and-selector-feedback.md`](../plans/TASK-0001-docker-runtime-setup-and-selector-feedback.md), current milestone `M5`.

## Documentation and ADR traceability

- Changed path: `README.md`
- Reason and supporting evidence: M3 documents the user-configured host
  `DATA` mount, valid `COLMAP_REC` Docker invocation, example scene path, and
  the repository FAISS asset. M5 reorganizes the same README into a
  Docker-first guide followed by detailed reference and extras. The original
  user report remains open evidence at
  `.agent-workspace/evidence/TASK-0001-M1-current-issues-2026-08-24.md`.

## Current verified state

- The image build context is the repository root; the Dockerfile copies it into
  `/workspace` and installs the requested editable project extras.
- `DATA=~/data/360-data` is user configuration; its sourced host value drives
  the `/data` mount. It must not be replaced by automatic home resolution in
  `docker_run.sh`.
- `data/colmap-vocab-tree-faiss/itri_vocab_tree_words8K.faiss.bin` exists in
  the repository and is available in the container at `/workspace/data/...`.
- The selector prints startup information; M4 confirmed that the documented
  `docker exec -it` invocation visibly reports its URL.
- The canonical checkout is
  `/home/kike/ws/scene-reconstruction-ws/colmap-reconstruction-ws/colmap-reconstruction-ws`.
- The task branch was created from `dev` at `6dee2e5` under the repository
  Git policy in `.gh-rule.md`.
- M1 established the static Docker and documentation contract. It did not build
  an image, run Docker, or verify selector output through `docker exec`.
- M2 built `scene-reconstruction-colmap:dev` as image
  `sha256:8cb5455fc1842825dabbbd00a72cba70b8638e9e529e2c7e78de17301f8ffc32`.
  Its requested extras and project console commands were verified inside the
  resulting image.
- M3 documents `DATA=~/data/360-data` as a user-configured host path mounted
  at `/data`, while the checked-in FAISS asset remains at `/workspace/data/...`.
  The selector example uses the valid `COLMAP_REC` shell variable and labels
  `/data/1782890858308` as an example.
- M4 ran the selector against `/data/1782890858308`; its TTY invocation printed
  `http://127.0.0.1:8765`, and a temporary probe received a non-empty response
  from that URL before the selector was stopped.
- M5 places the build, container start, native reconstruction, and selector
  paths before local-install alternatives and detailed reference material.

## Validation

- Command or check: Amber workspace validator passed during M1.
- Command or check: `bash -n docker/docker_run.sh docker/docker_build.sh` passed during M1.
- Command or check: sourced `./env` and the same absolute-path form used by
  `docker_run.sh`; both resolved `DATA=/home/kike/data/360-data`.
- Command or check: inspected Docker build context, Dockerfile installation
  directives, README Docker invocation, selector startup prints, and the local
  FAISS asset during M1.
- Command or check: M2 no-cache Docker build completed with the repository root
  as its context.
- Command or check: in the M2 image, imports of `pycolmap`, `open3d`, `cv2`,
  `colmap.panorama.cli`, and the selector server passed.
- Command or check: in the M2 image, `colmap-panorama-reconstruct`,
  `manual-pinhole-select`, and `colmap-visualize` were found on `PATH`; pip
  reported `colmap-360-reconstruction 0.1.0`.
- Command or check: focused project tests not run; M2 changed only image build
  configuration.
- Command or check: M3 `bash -n docker/docker_run.sh docker/docker_build.sh`
  passed; sourcing `./env` resolved `DATA=/home/kike/data/360-data`.
- Command or check: inspected README for `COLMAP_REC`, `docker exec -it`, the
  example scene path, the checked-in FAISS path, and removal of the unresolved
  intake note.
- Command or check: inside `scene-reconstruction-colmap`,
  `pytest -q tests/test_renderer_api.py` passed (`45 passed`).
- Command or check: `docker exec -it scene-reconstruction-colmap timeout 12s
  manual-pinhole-select --image-dir /data/1782890858308 --host 127.0.0.1
  --port 8765` printed the selector URL and startup paths.
- Command or check: a temporary detached selector returned a non-empty
  `http://127.0.0.1:8765/` response and was stopped after the probe.
- Command or check: M5 inspected heading hierarchy and the Docker command,
  selector path, FAISS path, and detailed-command prefix guidance.

## Next action

Request explicit approval to merge
`task/TASK-0001-docker-runtime-setup-and-selector-feedback` into `dev` after
reviewing its commit range and residual risks.

## Closure

| Repository | Source branch | Target branch | Integration commit | Post-merge validation |
| --- | --- | --- | --- | --- |
| `.` | `not merged` | `dev` | `not merged` | `not run` |

- Deviations from plan: none.
