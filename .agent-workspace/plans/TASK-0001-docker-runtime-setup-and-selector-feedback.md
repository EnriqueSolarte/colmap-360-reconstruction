# TASK-0001 plan — Docker runtime setup and selector feedback

```text
Protocol: amber@v3
Record schema: 2
Task: TASK-0001
Status: completed
Owner: Codex
Dependencies: none
Plan revision: 13
Current milestone: M6
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
| M3 | Align data mounting and Docker documentation | `completed` | 4 | M2 | Completed 2026-08-25 |
| M4 | Verify documented selector workflow | `completed` | 4 | M3 | Completed 2026-08-25 |
| M5 | Reorganize README as a usage-first guide | `completed` | 2 | M4 | Completed 2026-08-25 |
| M6 | Polish practical README workflow | `completed` | 2 | M5 | Completed 2026-08-25 |

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

State: `completed`
Revision: 4
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

### Implementation closeout

**What and why.** M3 made the existing user-controlled data mount and Docker
command boundary explicit, so readers do not confuse host scene data with
repository-provided assets.

**How and where.** `env` now labels `DATA` as the host directory that
`docker/docker_run.sh` mounts at `/data`. `README.md` adds a Docker workflow
that sets `COLMAP_REC=scene-reconstruction-colmap`, uses
`docker exec -it "$COLMAP_REC"`, identifies `/data/1782890858308` as an
example scene path, and gives the FAISS asset's repository mount path
`/workspace/data/colmap-vocab-tree-faiss/itri_vocab_tree_words8K.faiss.bin`.
The mount implementation itself is unchanged.

**Evidence.** `bash -n docker/docker_run.sh docker/docker_build.sh` passed.
Sourcing `./env` resolved `DATA=/home/kike/data/360-data`. README inspection
confirmed the container variable, `docker exec -it` selector invocation,
example-path label, FAISS path, and removal of the unresolved intake note.

**Known limits and downstream effect.** M3 does not start a container or prove
selector output visibility. M4 must run the documented selector command and
verify that it reports a reachable URL or a concrete startup error. This
closeout is committed as `docs(docker): complete TASK-0001 M3 workflow`.

## M4 — Verify documented selector workflow

State: `completed`
Revision: 4
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

### Implementation closeout

**What and why.** M4 verified that the documented interactive workflow makes
the selector startup URL visible, so no application-level output-flushing
change is required for the supported TTY invocation.

**How and where.** In the running `scene-reconstruction-colmap` container,
`docker exec -it scene-reconstruction-colmap timeout 12s
manual-pinhole-select --image-dir /data/1782890858308 --host 127.0.0.1
--port 8765` reported `http://127.0.0.1:8765`, the image directory, and the
selections path. A short detached probe then returned a non-empty response from
`http://127.0.0.1:8765/` through host networking and was stopped.

**Evidence.** `pytest -q tests/test_renderer_api.py` passed in the container
with `45 passed`. The documented TTY command printed its startup output; the
host-network probe established reachability.

**Known limits.** The running container was created from the pre-M2 image, but
it uses the repository bind mount and was sufficient to verify the selector
workflow; M2 separately verified the new image's project installation and
commands. Non-TTY output remains potentially buffered and is not the supported
documented workflow. This closeout is committed as
`test(docker): complete TASK-0001 M4 selector workflow`.

## M5 — Reorganize README as a usage-first guide

State: `completed`
Revision: 2
Depends on: M4
Milestone type: implementation

### Deliverable

A README that begins with a concise Docker-first path for the two primary
workflows and places detailed reference material and optional extras at the
end.

### Scope

Reorder and clarify README content only. Preserve command behavior, current
Docker contract, and the documented data and FAISS paths.

### Validation

- Inspect heading hierarchy and the top-level usage flow.
- Verify the documented Docker commands, paths, and task traceability remain
  present.
- Run the Amber validator and `git diff --check`.

### Completion condition

A new reader can identify how to start the container, run a native panorama
reconstruction, and launch the pinhole selector before encountering detailed
reference and optional workflows.

### Decisions and risks

Keep advanced material rather than removing it. Reorganization can introduce
broken heading hierarchy or accidentally alter commands, so validation compares
the primary command and path strings explicitly.

### Implementation closeout

**What and why.** The README now leads with the practical Docker sequence and
the two primary user choices before presenting local-install alternatives and
the full command reference. This makes the intended first run discoverable
without discarding advanced material.

**How and where.** `README.md` has a `Use it now` section with container setup,
native reconstruction, selector launch, output locations, and a reminder that
detailed commands run inside the container. `Details and extras` contains the
previous local installation, API, reconstruction options, pinhole workflow,
visualization, utility, and limitation material with a nested heading
hierarchy.

**Evidence.** Heading inspection confirmed the usage path precedes the
details boundary. Command/path inspection confirmed `COLMAP_REC`,
`docker exec -it`, `/data/1782890858308`, and the repository FAISS path remain
present. The Amber validator and `git diff --check` passed.

**Known limits.** This reorganizes documentation only; it does not re-run
Docker, selector, or reconstruction tests. This closeout is committed as
`docs(readme): complete TASK-0001 M5 usage guide`.

## M6 — Polish practical README workflow

State: `completed`
Revision: 2
Depends on: M5
Milestone type: implementation

### Deliverable

A concise, Docker-ready pinhole workflow that explains browser selection
controls and continues from saved selections through rendering, optional pose
attachment, and visualization.

### Scope

Polish the user-edited README only and update the Amber task records. Preserve
the concrete example scene path and keep advanced behavior/reference material.

### Validation

- Check Markdown hierarchy, command prefixes, and documented paths.
- Run `git diff --check` and the Amber workspace validator.

### Completion condition

A reader can follow the selector workflow from browser interaction through the
next command-line stages without needing to infer container prefixes or paths.

### Decisions and risks

Use `/data/1782890858308` consistently as the clearly labelled example path.
Documentation-only changes must not claim that reconstruction or GUI behavior
was re-tested.

### Implementation closeout

**What and why.** M6 turns the user-added selector notes into a concise,
actionable browser-control guide and completes the first-run pinhole path.

**How and where.** `README.md` now retains `/data/1782890858308` as the
explicit example path, explains preview/save/delete/navigation controls, and
uses `docker exec -it "$COLMAP_REC"` for rendering, optional pose attachment,
and visualization. Detailed selection guidance and advanced reconstruction
material remain in `Details and extras`.

**Evidence.** Markdown heading and command/path inspection passed.
`git diff --check` passed, and the Amber workspace validator passed after the
milestone records were synchronized.

**Known limits.** This is a documentation-only polish. It does not rerun
Docker, browser interaction, rendering, pose attachment, or visualization.
This closeout is committed as `docs(readme): complete TASK-0001 M6 workflow polish`.

## Plan change log

| Date | Milestone | Change | Reason | Approval |
| --- | --- | --- | --- | --- |
| 2026-08-25 | Plan | Revision 1 created | Approved task definition from current issue note | approved |
| 2026-08-25 | M1 | Revision 2 started | User approved task-branch setup and M1 diagnostic execution | approved |
| 2026-08-25 | M1 | Revision 3 completed; M2 revision 2 ready for approval | Static Docker/runtime contract verified; implementation remains approval-gated | approved |
| 2026-08-25 | M2 | Revision 3 started | User approved M2 execution | approved |
| 2026-08-25 | M2 | Revision 4 completed; M3 revision 2 ready for approval | Image build and requested package/command verification passed | approved |
| 2026-08-25 | M3 | Revision 3 started | User approved M3 execution | approved |
| 2026-08-25 | M3 | Revision 4 completed; M4 revision 2 ready for approval | Mount guidance and Docker documentation checks passed | approved |
| 2026-08-25 | M4 | Revision 3 started | User approved M4 execution | approved |
| 2026-08-25 | M4 | Revision 4 completed | TTY selector output, URL reachability, and focused tests passed | approved |
| 2026-08-25 | M5 | Revision 1 started | User approved README usage-first reorganization | approved |
| 2026-08-25 | M5 | Revision 2 completed | Docker-first usage guide and end-of-document reference structure verified | approved |
| 2026-08-25 | M6 | Revision 1 started | User requested README review and polish after adding selector-operation notes | approved |
| 2026-08-25 | M6 | Revision 2 completed | Browser controls and Docker-ready pinhole command sequence validated | approved |
