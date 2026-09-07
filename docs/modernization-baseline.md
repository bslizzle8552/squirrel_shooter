# Squirrel runtime modernization baseline

Phase 0 and Phase 1 begin from development commit
`7a8cfe465ab40fb0c7010f1c82cbfe561fd4b812` on the separate
`squirrel-runtime-v2` branch/worktree. The field reference remains
`de2aee9c9c7b485f79c4c9ce36cc0a6a1cf23c85` on `v1-field-simple`.
Their merge base is `b81168012ea3a5733dba951d4e2cfcfb2a70ae0b`;
neither tip is a fast-forward replacement for the other.

Development is the implementation base because its schema-3 attempted-shot
reservation reaches durable storage **before** possible actuation, retains
uncertain attempts across restart, and separates inference, decisions, and
bounded evidence/storage work. Those are reviewed source/test properties,
not claims that the field Pi already runs them.

The initial preservation commit includes the owner's pending classifier
scene-request timeout fix, its regression, README links, and existing offline
dataset utilities/docs/tests. Before cleanup, the full preserved development
suite passed: **536 tests**. Historical reports and device evidence stay in
their original storage and in an external hash-verified snapshot; they are
not new runtime defaults or source-controlled calibration.

## Owner policy and phase boundary

The later one-class autonomous policy is confirmed squirrel, sufficient
configured confidence, current identity/observed geometry, calibrated aim,
and ordinary freshness, rate and hardware checks. A separate human detector
or person veto is **not** a future requirement. This owner decision supersedes
the forensic audit's earlier recommendations about preserving person semantics.

The operational MobileNet/VOC classifier and its existing person gates remain
available during this phase behind an explicit legacy boundary. Do not infer
that a future squirrel-only detector can publish legacy “person clear” results.
The later migration will retire that semantic path.

Preserve one camera owner, latest-frame provenance, bounded reacquisition,
non-fireable coast/lost/ambiguous state, nine-point calibration without
extrapolation, server servo limits, and one physical coordinator. Preserve
move → settle → final validation → pulse → close → cooldown → PARK, the
normally-closed active-high valve contract, and conservative attempted-shot
limits. Cleanup movement requires software evidence of successful valve
closure. Calibration editing must inhibit automatic engagement in the backend.

No Phase 0/1 change authorizes Pi deployment, restart, physical or camera
testing, model inference/training/export, the new detector/2 Hz worker,
recording redesign, or altered operational thresholds, geometry or timing.

## Reproduction and evidence

The pre-change FIELD suite passed 347 tests in the forensic audit. Retained
September 7 event provenance identifies the field revision above; it does not
prove the current Pi process's loaded bytes. In particular, the 07:24 event's
late classification followed loss/coasting and reacquisition timeout; it is
not evidence that loosening a final gate would create a current target.

The durable forensic report is
`D:\Squirrel_Squirter_ML_Rebuild\00_project_docs\SOFTWARE_SYSTEM_FORENSIC_AUDIT_2026-09-07.md`.
Phase 0 status, refs, binary tracked diff, all non-ignored original files and
their SHA-256 manifest are preserved under
`D:\Squirrel_Squirter_ML_Rebuild\09_generated_work\software_modernization_phase01_20260907`.
`Misc/` was excluded entirely. The original dirty development worktree and
clean field worktree remain separate and unchanged.

Run the offline test suite with the existing project environment, with no
hardware/model execution or dependency upgrade:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:\dev\squirrel-squirter\.venv\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider
```

Use an external, unique `--basetemp` and `--junitxml` location when preserving
validation evidence. The branch handoff records final commits, exact results,
remaining phase boundaries, and rollback instructions.

## Phase 0/1 completion

The integrated offline suite passes **592 tests** (536 preserved baseline +56
new cases). This includes cleanup exceptions/unknown valve state, backend
calibration inhibition, canonical large-object association rejection, camera
publication epochs, legacy semantic isolation, startup provenance, and the
retained September 7 late-classification trace. One existing asynchronous retry
test now waits for published persistence completion instead of file existence.

The unsupported angle-based `move_and_fire` shortcut and unused `modes.py` enum
were removed. Conditional legacy vision/detection/wrapper/diagnostic paths,
historical prediction/review data and all operational configuration values
remain. Runtime-health composition and generation-aware consumer invalidation
are deferred as described in [runtime provenance](runtime-provenance.md).

This is ready for the separately authorized Phase 2 recording work in this
worktree. It is not a deployment or physical validation result. Continue to
preserve current hardware values and the conservative pre-actuation ledger
while designing recording changes.
