# TASK-0001 plan — Docker runtime setup and selector feedback

```text
Protocol: amber@v3
Record schema: 2
Task: TASK-0001
Status: ready
Owner: Codex
Dependencies: none
Plan revision: 5
Current milestone: M3
```

The task is the durable goal and integration contract. This plan is the
permanent evolving execution strategy.

## Milestone states

| State | Meaning |
| --- | --- |
| `proposed` | Recorded discovery with no execution authority. |
| `pending` | Approved future work whose prerequisites are incomplete. |
| `ready-for-approval` | Defined next slice awaiting explicit implementation approval. |
| `in-progress` | Explicitly approved slice currently executing. |
| `blocked` | Approved work stopped by a recorded condition. |
| `completed` | Completion condition met with validation evidence. |
| `deferred` | Valid work intentionally postponed. |
| `superseded` | Work replaced by another milestone or decision. |

## Milestone index

| ID | Milestone | State | Revision | Depends on | Last change |
| --- | --- | --- | --- | --- | --- |
| M1 | Confirm Docker runtime contract | `completed` | 3 | none | Completed 2026-08-25 |
| M2 | Install project dependencies in image | `completed` | 4 | M1 | Completed 2026-08-25 |
| M3 | Align data mounting and Docker documentation | `ready-for-approval` | 2 | M2 | Ready for approval 2026-08-25 |
| M4 | Verify documented selector workflow | `proposed` | 1 | M3 | Created 2026-08-25 |

## M1 — Confirm Docker runtime contract

State: `completed`
Revision: 3
Depends on: none
Milestone type: diagnostic

### Deliverable

Verified build-context, bind-mount, data-path, and selector-output behavior
with the current user edits reconciled as baseline.

### Scope

Inspect only; do not overwrite uncommitted user edits or build an image until
this slice is approved.

### Validation

- Inspect Docker build command, Dockerfile paths, run script, and README.
- Run safe syntax and targeted non-mutating checks where available.

### Completion condition

The task records the exact build-context change, runtime data contract,
documentation paths, and selector feedback remedy supported by evidence.

### Decisions and risks

`DATA` remains a user-edited configuration value. `/data/1782890858308` is a
documentation example, not a required dataset location. The existing dirty
files must be preserved.

### Diagnostics and findings

**What and why.** M1 established the current Docker runtime contract before
implementation, so M2 can correct the image-install path without overwriting
the preserved baseline edits.

**How and where.** In this repository, `docker/docker_build.sh` invokes
`docker build` with `docker/` as its context. `docker/dockerfile.dev-container`
does not copy `pyproject.toml` or `src/` and has no project `pip install`
directive. `docker/docker_run.sh` sources the repository `env` file by absolute
path; its `DATA` value resolves to `/home/kike/data/360-data` and is bound at
`/data/`. The repository is mounted independently at `/workspace`, where
`data/colmap-vocab-tree-faiss/itri_vocab_tree_words8K.faiss.bin` exists.

**Documentation and selector facts.** The README has no `docker exec` or
`COLMAP-REC` invocation and retains an unresolved note about the FAISS asset.
`src/render_pinhole/tools/manual_pinhole_select_server.py` prints its URL but
does not request flushing, so the cause of silent non-TTY `docker exec` output
is not established by static inspection.

**Evidence.** `bash -n docker/docker_build.sh docker/docker_run.sh` passed.
Sourcing `./env` and the absolute path used by `docker_run.sh` both yielded
`DATA=/home/kike/data/360-data`. The Amber validator passed. The related user
report remains open evidence at
`.agent-workspace/evidence/TASK-0001-M1-current-issues-2026-08-24.md`.

**Known limits and downstream effect.** No Docker image build, package import,
focused selector test, or live `docker exec` invocation ran. M2 must change and
verify the build context and image dependency installation; M3 must repair the
README/mount documentation; M4 must test selector observability. This closeout
is committed as `chore(task): complete TASK-0001 M1 Docker contract diagnosis`.

## M2 — Install project dependencies in image

State: `completed`
Revision: 4
Depends on: M1
Milestone type: implementation

### Deliverable

An image build that can access the repository and installs the editable project
with `test`, `colmap`, and `visualize` extras.

### Scope

Change Docker build context and Dockerfile installation steps only as supported
by M1; do not alter reconstruction behavior.

### Validation

- Build the Docker image.
- In the resulting image, verify package imports and console-command discovery.

### Completion condition

The image build completes and the requested project extras are installed.

### Decisions and risks

The runtime `/workspace` bind mount must remain compatible with the editable
installation. Image build time and dependency availability are external risks.

### Implementation closeout

**What and why.** The image now builds from the repository root and installs
the editable project with the `test`, `colmap`, and `visualize` extras. This
makes the image self-contained for project command discovery before its
`/workspace` runtime bind mount is attached.

**How and where.** `docker/docker_build.sh` passes `${DIR}/..` as the Docker
build context. `docker/dockerfile.dev-container` sets `WORKDIR /workspace`,
copies that context there, then runs `python -m pip install --no-cache-dir -e
'[test,colmap,visualize]'`.

**Evidence.** A no-cache build completed and produced
`scene-reconstruction-colmap:dev` image
`sha256:8cb5455fc1842825dabbbd00a72cba70b8638e9e529e2c7e78de17301f8ffc32`.
Inside it, imports of `pycolmap`, `open3d`, `cv2`, `colmap.panorama.cli`, and
`render_pinhole.tools.manual_pinhole_select_server` passed. `command -v`
found `colmap-panorama-reconstruct`, `manual-pinhole-select`, and
`colmap-visualize`; pip reported project version `0.1.0`.

**Known limits and downstream effect.** This does not verify runtime bind
mount behavior, README accuracy, live selector output, or focused tests. M3
owns mount/documentation alignment; M4 owns the live selector workflow. This
closeout is committed as `feat(docker): complete TASK-0001 M2 image install`.

## M3 — Align data mounting and Docker documentation

State: `ready-for-approval`
Revision: 2
Depends on: M2
Milestone type: implementation

### Deliverable

A consistent host `DATA` to container `/data` contract and README examples
using valid `docker exec` syntax, including labelled example scene paths.

### Scope

Update only the run configuration and documentation needed for the verified
contract. Preserve the user-owned `DATA` configuration model.

### Validation

- Run shell syntax checks.
- Inspect the rendered README command paths and container-name variable.

### Completion condition

The documented data and vocabulary-tree paths distinguish `/data` from the
repository mount at `/workspace/data`.

### Decisions and risks

Do not infer a user's home directory in `docker_run.sh`. Avoid documenting a
dataset-specific path as a required default.

## M4 — Verify documented selector workflow

State: `proposed`
Revision: 1
Depends on: M3
Milestone type: implementation

### Deliverable

A verified interactive selector invocation that immediately exposes the URL to
the user.

### Scope

Use `docker exec -it` in documentation and add application-level flushing only
if M1 confirms it is needed for supported non-TTY usage.

### Validation

- Run focused selector tests.
- Run the documented command against a suitable mounted image directory when
  Docker and data are available.

### Completion condition

The command exposes a reachable URL or reports a concrete startup error.

### Decisions and risks

Host networking makes `127.0.0.1` valid for the documented local workflow;
remote access is out of scope.

## Plan change log

| Date | Milestone | Change | Reason | Approval |
| --- | --- | --- | --- | --- |
| 2026-08-25 | Plan | Revision 1 created | Approved task definition from current issue note | approved |
| 2026-08-25 | M1 | Revision 2 started | User approved task-branch setup and M1 diagnostic execution | approved |
| 2026-08-25 | M1 | Revision 3 completed; M2 revision 2 ready for approval | Static Docker/runtime contract verified; implementation remains approval-gated | approved |
| 2026-08-25 | M2 | Revision 3 started | User approved M2 execution | approved |
| 2026-08-25 | M2 | Revision 4 completed; M3 revision 2 ready for approval | Image build and requested package/command verification passed | approved |
