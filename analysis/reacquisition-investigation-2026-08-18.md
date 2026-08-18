# Squirrel Squirter: Visible-Squirrel Reacquisition Failure

Date: 2026-08-18

Event: `20260818-103410-613-77abc4`

Deployed revision: `a7e0826a1e09ee17ad599c4558cf6d4583a559d5`

Development branch: `manual-control`
Implementation commit: `5dfa842` (`Prevent event recording from stalling target tracking`)

## Result

Primary diagnosis: **H, specifically A + B + E, caused first by a synchronous event-recording stall.**

The squirrel did not leave the scene. Tracker 1318's confirming detector frame was followed by a **1.868-second gap between detector observations**. The newly started event synchronously rendered and encoded its retained prebuffer on the same thread that runs motion detection. Writing the 20 pre-event frames plus the event frame consumed at least 1.408 seconds of that interval. On the next detector iteration, the squirrel was still detected, but its recovered centroid was about 153 pixels from tracker 1318's retained centroid. The normal tracker rejected continuity because both its 0.9-second maximum gap and 100-pixel distance limits had been exceeded. The auto-fire reacquisition matcher then independently compared the new group with the **last observed** centroid, not a predicted position, and the candidate again fell outside 100 pixels. Later detector iterations alternated between no group and gray unconfirmed/ineligible groups. No current, confidently associated target was recovered, so the event correctly failed closed with `target_reacquisition_timeout` before aiming.

Increasing the 0.9-second grace period would not correct the initiating defect. It would only leave auto-fire waiting longer after the motion thread had already blinded itself and fragmented the track.

## Evidence boundaries

The retained `clip.avi` is an annotated review clip, not the raw detector input stream. The MOG2 background model, exact per-iteration capture timestamps, rejected contours, and numeric IDs for later gray groups were not serialized by deployed revision `a7e0826`. Therefore:

- Values marked **exact** below come from `event.json`, `classification.json`, or structured runtime logs.
- Values marked **overlay-recovered** were measured from the detector's own drawn bounding boxes and centroid markers in the retained MJPG clip. Compression limits them to approximately a few pixels.
- `unknown` means the deployed software did not retain the value. No numeric tracker ID or filter reason was invented.
- An exact raw MOG2 replay is not defensible: the clip contains overlays, only about two seconds of lead-in, and no saved subtractor state.

The unannotated `original-frame.jpg` matches clip frame 20, establishing the clip alignment used below.

## Timeline

| Evidence | Time | Finding |
|---|---:|---|
| Tracker-1318 source observation | monotonic `174790.138243523` (exact) | Confirmed, event eligible, centroid `(805.26, 344.32)`, box `(746,316,112,62)` |
| Event wall timestamp | `10:34:10.615 EDT` (exact) | Event directory/writer had begun |
| Classifier submitted | `10:34:12.023 EDT` (exact) | At least 1.408 s elapsed while beginning the event and encoding the retained frames |
| Classifier completed | `10:34:12.448 EDT` (exact) | `bird`, 83.581%, 419.01 ms |
| Next missing-target iteration / coasting | monotonic `174792.006497442`; log `10:34:12.099 EDT` (exact) | 1.868 s after the retained target observation; already over the tracker's 0.9 s maximum gap |
| Classifier held pending live target | `10:34:12.478 EDT` (exact) | Classification qualified, but no associated current target existed |
| Auto-fire decision rejected | `10:34:12.907 EDT` (exact) | `target_reacquisition_timeout` |
| Runtime failure record | monotonic `174792.990668785`; log `10:34:13.042 EDT` (exact) | 0.984 s after coasting began |
| Event finished | `10:34:13.393 EDT` (exact) | No aim, servo, settle, or valve stage occurred |

The wall timestamp and monotonic timestamp serve different purposes here. The monotonic pair proves the detector-observation gap. The event-start-to-classifier-submit wall pair isolates at least 1.408 seconds inside synchronous event startup, where deployed code wrote the retained frames before returning to detection.

## Detector iterations and candidates

The clip contains 31 frames. Frames 0-19 are the pre-event buffer. Frame 20 is the exact source frame retained for tracker 1318 and classification. The encoded clip frame rate is playback metadata and cannot be used as an exact per-iteration capture clock.

| Clip frame | Detector result | Tracker / state | Centroid | Box and area | Distance from 1318 | Area ratio | Compatibility result |
|---:|---|---|---|---|---:|---:|---|
| 20 | Candidate produced | `1318`; persistence 5; confirmed; event eligible; `small_animal_candidate` | `(805.26,344.32)` exact | `(746,316,112,62)`; area `6,944`; foreground `2,528` exact | `0` | `1.000` | Live event target |
| blind interval | No detector iteration was run | Event-start recording occupied the motion thread | unknown | unknown | unknown | unknown | Core continuity clock advanced 1.868 s |
| 21 | Candidate produced | Different/new gray unconfirmed, ineligible group; numeric ID unretained | approximately `(657,383)` overlay-recovered | approximately `(600,350,108,74)`; box area `7,992` | approximately `153.05 px` | approximately `1.151` | Fails 100 px distance; area would pass 2.5 |
| 22 | No candidate box drawn | No useful grouped candidate | unknown | unknown | n/a | n/a | Detector supplied nothing to match |
| 23 | Candidate produced | Gray unconfirmed, ineligible group; numeric ID unretained | approximately `(541,408)` | approximately `(500,386,84,46)`; box area `3,864` | approximately `271.65 px` | approximately `1.797` | Fails distance; area would pass |
| 24 | Candidate produced | Gray unconfirmed, ineligible group; numeric ID unretained | approximately `(535,409)` | approximately `(498,392,74,40)`; box area `2,960` | approximately `277.71 px` | approximately `2.346` | Fails distance; area still passes 2.5 |
| 25-30 | No candidate boxes drawn | No useful grouped candidates | unknown | unknown | n/a | n/a | Detector supplied nothing to match |

The gray annotation color is generated only when a group is both unconfirmed and not event eligible. The deployed overlay did not include tracker ID, persistence count, or the filter-reason text in a machine-readable sidecar. Frame 21's immediate gray state, combined with the absence of tracker 1318 from the recorder, proves identity continuity was lost; the exact replacement ID is unknown.

## Exact first failing condition

The first failure is in normal tracker continuity, before auto-fire's reacquisition decision:

```python
now - track.last_at <= 0.9
and distance(last_centroid, new_centroid) <= 100
```

The next observation arrived 1.868 seconds after tracker 1318's retained observation, so the maximum-gap condition failed. The overlay-recovered centroid displacement was also about 153 pixels, so the distance condition failed independently. A new track was therefore created instead of updating 1318.

Auto-fire then evaluated all current groups against tracker 1318's last box and rounded centroid `(805,344)`. Its matcher uses only:

1. Euclidean distance from the last observed centroid, limited to 100 pixels.
2. Bounding-box area ratio, limited to 2.5.

It does **not** use velocity prediction, direction, IoU, overlap, persistence, confirmation, event eligibility, or candidate category to decide geometric compatibility. Frame 21 was outside the distance gate, so it produced zero compatible groups. Zero compatible groups remain in coasting until timeout; the different-ID incompatibility path triggers only when exactly one geometrically compatible group exists.

## Detector versus reacquisition

Both subsystems contributed, but the event-start stall came first:

1. **Observation pipeline failure:** synchronous review-clip startup prevented the detector from processing frames for 1.868 seconds.
2. **Tracker fragmentation:** the next visible-squirrel candidate exceeded both normal continuity limits and received a different/new identity.
3. **Reacquisition distance rejection:** the same candidate was approximately 153 pixels from the stale last centroid, over the separate 100-pixel limit.
4. **Intermittent segmentation:** subsequent frames alternated between no useful group and new gray unconfirmed/ineligible groups, preventing restored current-target confidence.
5. **Fail-closed timeout:** no current group was accepted as tracker 1318 before the deadline.

This is not solely “the detector saw nothing,” and it is not solely “the matcher rejected a good continuous track.” It is the combination represented by diagnosis H.

## Tracker and event identity

- Old tracker: `1318`.
- Later squirrel group: different/new tracker identity, exact numeric ID not retained.
- New event ID for the same squirrel: none found in the retained event/log evidence.
- Why it was not linked: the old track was already 1.868 seconds stale and the new centroid was about 153 pixels away; deployed continuity limits were 0.9 seconds and 100 pixels.
- Why it did not confirm as a replacement: later grouping was intermittent, and the visible gray groups were unconfirmed and event ineligible.

## Motion prediction assessment

Deployed reacquisition does not predict motion. Using the reported 210.03 px/s average and the reported leftward direction, a simple constant-velocity projection from `(805,344)` gives:

| Elapsed | Leftward displacement | Predicted point |
|---:|---:|---:|
| 0.2 s | 42 px | `(763,344)` |
| 0.4 s | 84 px | `(721,344)` |
| 0.6 s | 126 px | `(679,344)` |
| 0.9 s | 189 px | `(616,344)` |

Those values show why a fixed circle around the last point is simplistic for a fast target. They do not justify adding prediction in this pass: the 1.868-second detector blind interval exceeded the intended continuity period, posture changed the segmentation centroid, and the later detector output was intermittent. Adding a wider or predicted association region without first removing the blind spot would increase wrong-object risk.

## Implemented narrow fix

Implementation commit: `5dfa842`.

1. Prebuffer annotation and event-clip encoding now run on a dedicated, serialized event-writer thread instead of blocking the motion detector at event start.
2. The writer queue is bounded at 32 live frames to avoid unlimited memory growth. Writer failures are surfaced as event recording failures.
3. Frame order remains prebuffer, event frame, then live updates. Finalization waits for queued frames before atomically completing the clip/event.
4. Failed and successful coasting attempts now retain at most 12 compact event-level diagnostic iterations in `event.json`.
5. Each diagnostic records candidate tracker ID, centroid, box, box area, distance from the last centroid, area ratio, both gate booleans, persistence, confirmation, eligibility, category, rejection reason, and the iteration decision.

No threshold was loosened. The 0.9-second grace, 100-pixel matcher radius, and 2.5 area-ratio limit remain unchanged.

## Regression coverage

Added coverage verifies:

- A deliberately blocked prebuffer renderer does not block `EventRecorder.begin` or the caller's motion-processing path.
- Prebuffer, event, and update frames retain order and are all finalized.
- Reacquisition diagnostics contain numeric gate evidence and are capped at 12 iterations.
- A fast left-moving track survives one brief segmentation gap when normal intermediate detector timing is preserved.
- Existing live-coordinate reacquisition uses the current coordinate.
- A different candidate is rejected.
- Multiple compatible candidates fail closed as ambiguous.
- A long disappearance times out.
- Continuous normal tracking remains unchanged.

Verification results:

- Focused changed-subsystem tests: `68 passed`.
- Full test suite: `335 passed`.
- Ruff on changed source/tests: passed.
- Python compilation of `src/squirrel_shooter` and `tests`: passed.
- Git whitespace check on changed files: passed.
- No physical camera, Pi CPU, GPIO, servo, valve, water, or firing test was performed.

The test uses the real failure shape and recovered numeric boundaries as a deterministic software regression. The annotated clip itself was not falsely presented as an exact raw MOG2 replay.

## Safety invariants

Unchanged and still fail closed:

- person hard veto;
- animal class allowlist and confidence threshold;
- classification and current-target freshness;
- current confirmed/event-eligible target requirement;
- native-frame calibration and camera-resolution validation;
- coordinator lock, cooldown, and rate limiting;
- servo movement and settle requirement;
- final pre-fire safety check;
- valve pulse and closure behavior;
- ambiguous association remains no fire;
- stale pre-coasting coordinates are never used for firing.

This pass is software-only evidence. It does not establish Pi scheduling performance, physical target association, unattended safety, aiming accuracy, or firing safety. A supervised deployment observation should confirm that event start no longer creates a multi-second detector gap before any physical validation is considered.

## Git and workspace

- Starting branch: `manual-control`.
- Starting local/remote revision: `a7e0826a1e09ee17ad599c4558cf6d4583a559d5`.
- Implementation commit: `5dfa842`.
- Only the two source files and three focused test files were included in the implementation commit.
- This report is committed separately on `manual-control` and the branch is pushed to `origin/manual-control`.
- Pre-existing modified and untracked owner work remains present and was not staged, cleaned, or altered.
- `Misc/` was treated as owner-only and was not inspected or touched.

## Recommended next step

Deploy the development commit through the established supervised Pi workflow, then collect one dry observation event with no actuation. Confirm in the new `event.json` that:

1. the interval between the confirming observation and the next detector iteration stays below the configured continuity gap;
2. `reacquisition_diagnostics` contains exact numeric candidate evidence if coasting occurs;
3. the target remains confirmed and current before any later physical test is authorized.
