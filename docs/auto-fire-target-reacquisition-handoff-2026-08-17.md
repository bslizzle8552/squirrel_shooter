# Auto-Fire Target Reacquisition Fix — 2026-08-17

## Outcome

The tracker/classifier timing race observed in the field is fixed in software. A qualifying asynchronous classification can now survive a brief, bounded detector dropout, but it cannot aim or fire until the same event and tracker ID are uniquely reacquired at a newly observed live position.

This pass did not access or operate the Pi, GPIO, servos, valve, camera, calibration, or water system. It is not physical validation.

## Root cause

The detector already kept an internal track alive for `motion.persistence.maximum_gap_seconds`, allowing a nearby blob to reuse the same tracker ID after a short miss. The motion runtime did not preserve that behavior for auto-fire: the first frame without the track immediately removed `(event_id, track_id)` from `_live_auto_targets`.

The classifier runs asynchronously. For the real 2026-08-17 events, its results arrived about 439–449 ms after submission:

- dog, 95.82%, event `20260817-095020-520-d8f811`, track `6371`;
- bird, 97.53%, event `20260817-095031-280-f6f292`, track `6373`.

Both event and tracker IDs were valid, and both events were still recording. A brief segmentation miss had already deleted the runtime's live target mapping. `AutoFireService._current_target()` therefore had no current target and correctly failed closed with `target_association_invalid`.

## Implementation

The existing tracker, classifier worker, auto-fire service, physical coordinator, calibration path, servo path, recording path, and valve path remain the only paths. No parallel detector or controller was added.

The runtime now maintains three bounded association states:

```text
LIVE
  | one detector miss
  v
COASTING (non-fireable, maximum 0.9 seconds)
  | unique compatible return, same event and tracker ID
  v
LIVE at newly observed coordinates

COASTING
  +-- timeout --------------------> EXPIRED / NO FIRE
  +-- different identity --------> INCOMPATIBLE / NO FIRE
  +-- multiple plausible targets -> AMBIGUOUS / NO FIRE
```

Important behavior:

- Entering coasting immediately removes the target from the live/fireable target map.
- The last coordinate is retained only as identity/reacquisition evidence. It is never supplied as an aiming coordinate.
- Reacquisition requires the same event ID and tracker ID.
- The returning target must be the sole compatible candidate.
- Compatibility is bounded by centroid distance and bounding-box area similarity.
- A qualifying classifier callback waits on a condition while the independent motion loop continues updating targets.
- Reacquisition wakes that existing classifier callback; it re-runs the normal policy using the newly observed target.
- Classification and target freshness are rechecked before engagement.
- The existing post-servo final safety validation remains authoritative.

## Configuration

The checked-in supervised field configuration now contains:

```yaml
auto_fire:
  track_loss_grace_seconds: 0.9
  reacquisition_max_centroid_distance_pixels: 100
  reacquisition_max_area_ratio: 2.5

motion:
  persistence:
    maximum_gap_seconds: 0.9
```

Validation requires the auto-fire grace to be positive, no longer than classifier freshness, and no longer than the detector's identity-preservation gap. Distance must be positive, and the area ratio must be at least 1.0.

The 0.9-second grace covers the observed 0.449-second field latency with approximately 0.45 seconds of margin while remaining far below the existing 3.0-second classification freshness limit. It does not enlarge `target_max_age_seconds`; that remains 0.75 seconds and applies to the newly observed live target.

## Diagnostics

Event-level structured logs now record:

- classifier context, event ID, tracker ID, submission/completion timestamps, and latency;
- entry into coasting, last-seen time, last centroid, and grace duration;
- successful reacquisition, elapsed time, old centroid, and new centroid;
- ambiguous, incompatible, and timed-out reacquisition failures;
- held-classification count in auto-fire status;
- the existing final acceptance/rejection reason.

There is no added per-frame informational log spam.

## Safety gates preserved

This fix did not lower or bypass:

- the dog/cat/bird allowlist;
- the strictly-greater-than-70% confidence requirement;
- the hard event-level person veto;
- unknown/error/non-model rejection;
- classification freshness;
- live target freshness;
- event ID and tracker ID identity;
- confirmed/event-eligible and person-sized motion checks;
- daylight/night restrictions;
- calibration completeness and interpolation bounds;
- camera frame-resolution matching;
- coordinator availability and busy state;
- servo move, settle, final safety check, and only-then-fire ordering;
- target-moved-during-aim rejection;
- cooldown, per-event, re-engagement, and rolling-hour limits;
- persistent rate-limit state and clock checks;
- valve closure, recording evidence, and PARK behavior.

## Tests and verification

Focused tracker/classifier/auto-fire/config verification:

```text
140 passed in 11.66s
```

The new regressions cover:

- brief dropout followed by valid reacquisition;
- no firing while classification is held during coasting;
- current reacquired coordinate used instead of the stale classifier-frame coordinate;
- grace timeout;
- incompatible/different tracker identity;
- ambiguous multiple candidates;
- stale classification after reacquisition;
- stale reacquired target;
- person veto while classification is held;
- unchanged continuously tracked behavior through the existing suite;
- existing final target-movement, geometry, night, person, coordinator, rate-limit, and persistence checks.

Full suite final run:

```text
331 passed in 16.16s
```

The first full run had one transient Windows `PermissionError` in an unrelated classifier retry test while its worker atomically replaced `classification.json`. That test passed immediately in isolation, and the complete suite then passed cleanly.

Additional verification:

- Python `compileall`: passed;
- dependency consistency (`pip check`): passed;
- Git whitespace (`git diff --check`): passed;
- repository JavaScript: untouched;
- no repository linter/type-checker is configured; `ruff` and `mypy` are not installed in this environment.

## Git

- Branch: `manual-control`
- Starting commit: `de6404848baf6cabc1d42c83dd00eec48a333ec4`
- Implementation commit: `070a1c7` (`Fix auto-fire target reacquisition race`)
- Implementation pushed: yes, to `origin/manual-control`
- Unrelated pre-existing modified/untracked Pi-data utility work was preserved and excluded from the implementation commit.
- `Misc/` was not inspected, modified, staged, or committed.

## Smallest supervised physical next step

Keep the water supply disconnected, establish a clear exclusion zone, and keep immediate power cutoff access. Deploy/restart the pushed `manual-control` revision only while attended. Move a nonliving dog/bird test image through a verified calibration area and briefly occlude it for roughly half a second while classification is running.

Success requires all of the following evidence:

1. one `auto_fire_target_coasting` log;
2. one `auto_fire_target_reacquired` log with the same event/tracker IDs;
3. the new centroid matching the returned target, not the old coordinate;
4. normal move and settle before the final safety check;
5. at most one dry valve pulse, followed by confirmed valve OFF and PARK;
6. a second trial with occlusion longer than 0.9 seconds producing `target_reacquisition_timeout` and no valve pulse.

Stop on any identity mismatch, ambiguous association, unexpected aim, missing final check, or valve behavior. Passing this attended dry test would validate one controlled Pi scenario only; it would not establish unattended auto-fire safety.
