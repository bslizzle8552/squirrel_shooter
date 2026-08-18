# Squirrel Squirter — Auto-Fire Clock Recovery Fix and Live Verification

**Date:** 2026-08-18  
**Branch:** `manual-control`  
**Fix commit:** `b81168012ea3a5733dba951d4e2cfcfb2a70ae0b` (`Recover empty auto-fire history after clock sync`)

## Result

The clock-recovery fix is deployed and behaving correctly in the field. After the owner pulled and restarted the Pi, auto-fire recovered to `IDLE` instead of remaining permanently `BLOCKED`.

The live service reported:

- auto-fire enabled: `true`
- auto-fire state: `IDLE`
- rate-limit state error: `null`
- detector state: `READY`
- camera online and camera thread alive
- detector thread alive
- classifier thread alive with no errors
- persistent rate-limit state healthy
- rolling-window shots: `0`
- remaining shots in window: `6`
- candidates evaluated: `0`
- accepted: `0`
- rejected: `0`

No movement, aiming, valve pulse, or physical firing test was requested or triggered during this verification.

## What Had Disabled Auto-Fire

The Pi started while its wall clock was wrong. When time synchronization corrected the clock forward by roughly 51 minutes, the old rate-limit protection treated that correction as unsafe and latched a permanent error:

> Auto-fire rate-limit clock is unsafe: wall clock jumped forward during runtime

Once latched, every later candidate was rejected as `safety_state_invalid` before the system could aim or fire. The reviewed evidence contained 16 such post-latch candidate rejections.

This explains why repeated classifications produced no movement or firing. It was not a human-review approval gate and was not caused by the new target-reacquisition logic.

## Dad's-Leg Event

For event `20260818-153131-300-99a55b`, track `19`, the classifier labeled the crop as `bird` at approximately 82% confidence. The event did **not** fire.

Evidence for that conclusion:

- the auto-fire decision was rejected with `safety_state_invalid`
- the decision had `accepted: false`
- no target pixel was selected
- there was no corresponding aim, servo movement, valve pulse, completed-shot record, or rolling-window shot

The crop mainly contained a leg and shoe, while the wider frame contained a person. That classifier-quality issue remains separate from this clock-recovery repair.

## Implemented Repair

The persistent rate limiter now distinguishes between an empty and a non-empty shot history:

- If the shot history is empty, a forward or backward wall-clock correction safely resets the clock baseline and emits a structured `auto_fire_rate_limit_clock_rebased` log entry.
- This recovery works both during runtime and during service startup, including after a reboot or power outage.
- If any recorded shot exists, a suspicious clock correction still fails closed and blocks auto-fire.
- Corrupt or unwritable persistent state still fails closed.

This preserves the hourly shot-limit guarantee whenever prior shots exist while avoiding permanent disablement when there is no rate-limit history to protect.

## Automated Verification

The development checkout passed:

- focused auto-fire tests: `56 passed`
- full test suite: `340 passed`
- lint checks for the changed files
- Python compilation check
- diff integrity check

New coverage includes:

- runtime forward correction with empty history
- runtime backward correction with empty history
- same-boot restart after a forward correction with empty history
- new-boot startup behind the saved wall clock with empty history
- continued fail-closed behavior for forward and backward corrections when shot history exists

## Deployment Evidence

The owner reported pulling `manual-control` and restarting the field Pi. The local and remote development branch both point to fix commit `b811680`.

The restarted service emitted the new structured recovery event at `2026-08-18T23:33:11.652+00:00`:

```json
{
  "event": "auto_fire_rate_limit_clock_rebased",
  "direction": "forward",
  "correction_seconds": 3072.182,
  "context": "service startup",
  "shot_history_count": 0
}
```

That event is direct behavioral evidence that the newly added recovery path ran on the Pi. The status API does not expose the deployed Git SHA, so the exact Pi checkout SHA was not independently read back over the API.

At approximately 19:35 EDT, more than two minutes after restart, a second live check still showed the service healthy and auto-fire `IDLE`, with zero candidates, accepted events, rejected events, or shots since restart.

## Current Safety Boundary

The system is available again, but this pass only verified software state and passive runtime health. It did not prove physical pan/tilt motion, aim accuracy, valve operation, wet firing, or correct classification of a field target.

The repair is intentionally narrow: it recovers only when the persistent shot history is empty. A suspicious clock change with recorded shots still blocks autonomous firing.

## Workspace Scope

Only the clock-recovery implementation, its tests, and this report belong to this pass. Existing unrelated modified and untracked files were preserved. `Misc/` was not inspected, modified, staged, or committed.
