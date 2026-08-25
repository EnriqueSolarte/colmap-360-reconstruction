# TASK-0001 plan — Docker runtime setup and selector feedback

```text
Protocol: amber@v3
Record schema: 2
Task: TASK-0001
Status: proposed
Owner: unassigned
Dependencies: none
Plan revision: 1
Current milestone: M1
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
| M1 | Confirm Docker runtime contract | `proposed` | 1 | none | Created 2026-08-25 |
| M2 | Install project dependencies in image | `proposed` | 1 | M1 | Created 2026-08-25 |
| M3 | Align data mounting and Docker documentation | `proposed` | 1 | M2 | Created 2026-08-25 |
| M4 | Verify documented selector workflow | `proposed` | 1 | M3 | Created 2026-08-25 |

## M1 — Confirm Docker runtime contract

State: `proposed`
Revision: 1
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

## M2 — Install project dependencies in image

State: `proposed`
Revision: 1
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

## M3 — Align data mounting and Docker documentation

State: `proposed`
Revision: 1
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
