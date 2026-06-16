Date: 2026-06-15
Project version: v0.1.0
Description: Define an executable plan to keep continuation runs efficient as sessions grow.

# Improve Sequential Estimation

## Goal

Reduce the runtime growth of repeated `colmap-panorama-reconstruct
--initial-session` runs while preserving the current continuation semantics:
the previous session is the fixed seed scaffold, new panoramas are added as new
rig frames, and the output is a new merged COLMAP model.

The problem is not primarily disk usage anymore. Previous images and masks are
hardlinked by default, so storage duplication is reduced. The remaining concern
is compute: as each continuation includes more old frames, COLMAP has more
images, features, matches, points, and frames to load, match, triangulate, and
optimize around.

## Working Assumptions

- The input sequence is usually spatially and temporally ordered.
- Most new panoramas overlap strongly with recent frames from the previous
  session.
- Full-history matching on every continuation is safe but increasingly wasteful.
- Existing seed frames are fixed in current continuation mode, so optimization
  should focus on registering and stabilizing new frames.
- Loop closure against old history is useful, but it should be explicit and
  bounded rather than paid for on every pair.

These assumptions are plausible for the current workflow, but they must be
validated with timing reports and reconstruction quality checks.

## Proposed Direction

Introduce a scoped continuation matching strategy. Instead of matching the full
accumulated database every time, seeded runs should identify the newly generated
image IDs and match only the useful subset:

- new images against other new images;
- new images against a configurable recent `last-k` set of old frames;
- later, new images against scored historical keyframes;
- later still, new images against a hybrid of recent frames and scored
  keyframes.

This should preserve the local registration path while reducing repeated work
against old history.

## Keyframe Scoring Discussion

A scored keyframe strategy is promising but should not be the first matching
policy. The strongest version would score prior rig frames by how useful they
are as future anchors. Candidate score terms include:

- how many other rig frames have verified matches to the frame;
- how many verified inlier matches support those connections;
- whether the supporting points or matches are distributed around the full
  panorama rig instead of concentrated in one virtual camera or yaw sector;
- whether the frame remains geometrically reliable after registration.

The circularity problem is manageable because scores can be initialized from
the first completed session and then carried forward. A frame does not need to
be scored before its first reconstruction; it can become a candidate anchor for
the next session after its match graph and reconstruction quality are known.

The critical risk is that high-scoring frames are not automatically good
matching anchors. Match count can overvalue repetitive, texture-heavy, or
redundant areas. A pure `top-k` policy could repeatedly select old easy frames
while missing the locally adjacent frames needed to register the next batch.
For that reason, scored selection should be evaluated only after a simpler
`last-k` baseline exists.

The agreed implementation order is:

1. `last-k`: match new images against the most recent old rig frames.
2. `top-k`: match new images against scored historical keyframes.
3. `hybrid-k`: match new images against both recent frames and scored
   keyframes.

The likely long-term default is `hybrid-k`, but it should not become the
default until `last-k`, `top-k`, and full-history matching have been compared on
representative continuation runs.

## Executable Plan

### Stage 1: Instrument Runtime By Pipeline Step

Add coarse timing around major pipeline phases:

- copy or hardlink initial session;
- render virtual views;
- feature extraction;
- seed camera/frame database rewrite;
- matching and rig verification;
- incremental mapping;
- report writing.

Success criteria:

- The JSON report includes per-stage elapsed seconds.
- The CLI summary still prints total elapsed time.
- A seeded run clearly shows where time is spent.

Rationale:

Do this before changing matching behavior. Without phase timings, we risk
optimizing the wrong part of the pipeline.

### Stage 2: Track New Image IDs As First-Class Data

After rendering and feature extraction, record the generated virtual image names
and their database `image_id`s in the report and internal pipeline state.

Success criteria:

- The code can distinguish seed images from new images without relying on name
  scans later in the pipeline.
- The report includes `new_image_count`, `new_image_ids`, or a compact equivalent
  that is useful for debugging.
- Existing full matching behavior is unchanged.

Rationale:

Scoped matching cannot be reliable until the pipeline knows exactly which images
belong to the current continuation batch.

### Stage 3: Add A Last-K Match Scope

Add a seeded-run option such as:

```text
--match-scope all|last-k
--last-k-frames N
```

Initial behavior:

- `all`: current behavior, safest fallback.
- `last-k`: match new images against new images and against the most recent
  `N` old rig frames from the seed session.

Success criteria:

- `all` remains the default until the scoped mode is proven.
- `last-k` reduces matching time on a seeded smoke run.
- The selected old frames are visible in the report.
- Reconstruction still registers the new frames on representative data.

Rationale:

This is the main efficiency lever. Matching against recent old frames should be
enough for continuous capture sequences and avoids repeatedly considering the
full historical graph.

Critical risk:

If the new sequence jumps to a distant scene area, recent-window matching may
fail to register frames that full-history matching would recover. This is why
`all` must remain available.

### Stage 4: Score Historical Keyframes

After `last-k` has baseline timings and registration quality, compute a
per-rig-frame keyframe score from the previous session. Start with conservative
score terms:

```text
keyframe_score =
  connected_frame_count
  * capped_verified_inlier_score
  * panorama_distribution_score
```

The score should be reported first before it controls matching. A good frame
should be connected to many other frames and have support distributed across
the 360-degree rig, not only concentrated in one virtual camera.

Success criteria:

- The report includes keyframe scores for prior rig frames, or a compact
  top-ranked summary.
- The scoring logic is deterministic and testable.
- Repetitive/high-texture frames do not dominate solely because of raw match
  count.

Rationale:

This prepares `top-k` matching without mixing scoring bugs with matching-scope
bugs. It also gives evidence about whether the proposed score actually selects
reasonable anchors.

### Stage 5: Add Top-K And Hybrid-K Match Scopes

Add scored matching modes after keyframe scores are visible:

```text
--match-scope top-k|hybrid-k
--top-k-frames K
--last-k-frames N
```

Initial behavior:

- `top-k`: match new images against new images and the top `K` scored old rig
  frames.
- `hybrid-k`: match new images against new images, the last `N` old rig frames,
  and the top `K` scored old rig frames.

Success criteria:

- `top-k` and `hybrid-k` are evaluated against `all` and `last-k`.
- `hybrid-k` preserves local continuity better than pure `top-k`.
- The report lists selected recent and scored anchor frames separately.
- Runtime remains bounded by `N + K`, not by full history.

Rationale:

Pure `top-k` is useful experimentally but risky as a default. `hybrid-k` is the
stronger candidate for production because recent frames protect sequential
continuity while scored keyframes provide longer-term memory.

### Stage 6: Optional Retrieval-Based Loop Candidates

Add retrieval-based loop candidates only after `last-k`, `top-k`, and
`hybrid-k` have been evaluated:

```text
--match-scope hybrid-retrieval
--retrieval-max-old-frames K
```

This mode should match new images against recent old frames, scored keyframes,
and a bounded set of old frames selected by retrieval or another cheap
similarity mechanism.

Success criteria:

- Runtime remains bounded by configured candidate counts, not by full history.
- The report lists selected retrieval frames.
- Loop closure behavior is better than pure recent or scored selection on
  non-contiguous sequences.

Rationale:

Recent frames and scored keyframes may still miss large revisits. Retrieval
candidates can recover loop closure without returning to full-history matching.

### Stage 7: Tune Mapping And Bundle Adjustment

After matching is scoped, evaluate mapper and BA settings for seeded runs:

- whether global BA is too frequent for fixed existing frames;
- whether local BA is sufficient for most continuation batches;
- whether point retriangulation/filtering dominates runtime.

Success criteria:

- Any mapper/BA change is backed by timing and reconstruction quality evidence.
- Existing frames remain fixed.
- New registered frame count and sparse point count remain acceptable.

Rationale:

Mapper tuning can help, but it is riskier than scoped matching. It should come
after the matching workload is bounded.

## Report Fields To Add

Likely useful fields:

- `elapsed_seconds_by_stage`;
- `match_scope`;
- `last_k_frames`;
- `top_k_frames`;
- `new_database_image_count`;
- `old_database_image_count`;
- `selected_old_frame_count`;
- `selected_old_frame_ids`;
- `selected_recent_frame_ids`;
- `selected_keyframe_ids`;
- `keyframe_scores`;
- `matching_pair_count` if available from COLMAP/database state.

Keep the report compact. Large lists should be capped or summarized unless they
are necessary for debugging.

## Verification Strategy

Use three levels of verification:

1. Unit tests for image/frame selection logic and report fields.
2. Small seeded smoke run with `--image-limit 1`.
3. Representative continuation run comparing:
   - total elapsed time;
   - matching elapsed time;
   - registered frames;
   - registered images;
   - sparse point count;
   - obvious registration failures.

The expected outcome is not identical sparse geometry. The expected outcome is a
useful continuation model with substantially lower runtime.

## Open Questions

- What should the first default be after implementation: keep `all`, or switch
  seeded runs to `last-k` after validation?
- What is a reasonable initial `--last-k-frames`: 10, 20, 50, or based on
  panorama count?
- Should `last-k` count old rig frames only, or should it expose image-level
  controls as well?
- What exact 360-degree distribution metric should the keyframe score use:
  active virtual camera count, yaw-bin entropy, triangulated point coverage, or
  verified match coverage?
- How should non-contiguous captures request broader matching without returning
  to full-history cost?

## Recommended Next Slice

Implement Stage 3 only: `last-k` matching as the first scoped matching policy.

This is the simplest measurable baseline. It should be implemented and
evaluated before top-scored keyframes or hybrid selection are added, because it
separates the effect of bounded matching from the harder problem of selecting
globally useful historical anchors.
