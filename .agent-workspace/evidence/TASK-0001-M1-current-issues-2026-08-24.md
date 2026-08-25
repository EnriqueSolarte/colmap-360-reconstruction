# TASK-0001 / M1 — User intake report, 2026-08-24

```text
Task: TASK-0001
Milestone: M1
Evidence status: open
Captured: 2026-08-24
Source: user-provided intake note
Verification: unverified; this report records requested changes and an observed selector symptom.
```

## Original report

1. It need to install python -m pip install -e '.[test,colmap,visualize]' inside the docker container. This must be already build when creating the docker image
2. Better define a env varibale in env file as DATA=<data stored in the host, which is mount in /data in the docker images> Use this as default
3. The commands run inside the docker, so the command must be define inthe README as docker exec $COLMAP-REC <command>
4. Lets assumes some data at the HOME -> ~/data/360-data which is mounted at /data/
5. when I run  docker exec scene-reconstruction-colmap manual-pinhole-select \
  --image-dir /data/1782890858308 \
  --host 127.0.0.1 \
  --port 8765
  there is not any message in the terminal it goes silently

## Traceability

This is open diagnostic evidence for [TASK-0001](../tasks/TASK-0001-docker-runtime-setup-and-selector-feedback.md), milestone M1. It must remain under `.agent-workspace/evidence/` until the relevant work is integrated and post-merge validation has verified the outcome. The report itself does not establish that any requested behavior is implemented or working.
