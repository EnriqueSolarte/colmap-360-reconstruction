# Agent Workspace Protocol — amber@v3

Protocol: amber@v3
Protocol source: agent-skills@611ac359e7520ca0a023012d94ff54aee89f98ff

This workspace uses Amber v3. This file is its local protocol authority; do not
add a local `PROTOCOL.md`.

## Workspace layout

```text
AGENTS.md
docs/                               # only when durable knowledge exists
  01-project-overview.md
  adr/0001-decision-title.md        # only when a durable decision exists
.agent-workspace/
  active.md                         # sole derived index of open tasks
  tasks/                            # required; permanent task records
  plans/                            # required; permanent plans when warranted
  archive/                          # legacy material and resolved evidence only
    evidence/
```

`AGENTS.md`, `.agent-workspace/active.md`, `tasks/`, `plans/`, and `archive/`
are required. Track otherwise-empty required directories with an appropriate
placeholder. `PLANS.md` is not permitted.

Every Amber v3 `active.md`, task record, and task plan declares `Protocol:
amber@v3`; legacy records retain their historical marker (or lack of one).
Amber task and plan records use `Record schema: 2` and metadata headers use
fenced `text` blocks with fields at column 1.

## Authority

| Artifact | Authority |
| --- | --- |
| `AGENTS.md` | Protocol pin, stable repository rules, and required checks. |
| Task record | Durable outcome, acceptance criteria, scope, repository ownership, evidence summary, next action, and integration state. |
| Task-specific plan | Evolving milestone strategy, dependencies, states, revisions, validation, and change history. |
| `active.md` | Derived navigation index of open tasks only. |
| Numbered docs and ADRs | Durable verified knowledge and decisions. |
| Git and validation output | Evidence of current repository and implementation state. |

`active.md` has one row per non-archived task with task ID, status, owner,
repositories, parallel safety, plan path, current milestone, and dependencies.
The task and plan remain authoritative. For planned tasks, use a link relative
to `active.md`, such as `[plans/TASK-0001-title.md](plans/TASK-0001-title.md)`.

## Task and plan lifecycle

Create a permanent task for non-trivial, multi-file, risky, multi-session,
approval-gated, or cross-repository work. Task IDs are permanent and
project-local; scan records, plans, branches, and traceability before assigning
an ID. Task states are `proposed`, `ready`, `in-progress`, `blocked`,
`ready-to-merge`, and `archived`.

Create a permanent task plan only for multiple implementation slices, approval
boundaries, dependencies, or meaningful uncertainty. A task without a plan uses
`Plan source: none` and `Current milestone: none`. Plans remain in `plans/`
after closure; task records also remain in `tasks/`, marked archived and removed
from `active.md`.

Milestones have permanent IDs, positive revisions, dependencies, deliverable,
scope, validation, completion condition, and decisions/risks. Only one may be
in progress. Every milestone declares `Milestone type: implementation` or
`diagnostic`; a completed diagnostic milestone ends with `### Diagnostics and
findings` documenting evidence, findings, limitations, and downstream effect.

## Repository, evidence, and closure

Use one canonical checkout per repository and switch task branches sequentially.
Multiple worktrees and parallel execution in a shared repository are not
supported. Each task records every affected repository, target branch, task
branch, canonical checkout, baseline, validation evidence, and integration
result.

When an active milestone reveals independently verifiable diagnostic or repair
work, create a child task with parent task/milestone traceability. Block the
parent milestone, preserve branch state before switching, and keep open evidence
in `.agent-workspace/evidence/`. After verified integration, move resolved
evidence to `.agent-workspace/archive/evidence/` and repair links.

`ready-to-merge` is not complete. Before a mainline merge, show source and
target branches, commit range, validation, unrelated changes, conflicts,
ordering, and residual risk; explicit merge approval is required. Archive only
after every integration and post-merge validation is verified.

## Migration and compatibility

Before migrating, inventory all records and identities, present a dry-run
mapping, and obtain approval before moving, renaming, or rewriting meaningful
files. Preserve legacy and prior-protocol records under their original
contract. New Amber v3 records belong in `tasks/` and `plans/`.

Traceability: workspace initialization approved 2026-08-24.
