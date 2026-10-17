# Final field Pi snapshot — October 17, 2026

This commit preserves the software installed in `/home/bslizzle8552/squirrel_collector_phase4_20260907` when the owner retired the field Pi. The active process was `python -B -m squirrel_shooter.collector_app`, using `collector.yaml`, the saved deer-feeder calibration and firing ledger, and the `v5-epoch148-fp32-416-r1` NCNN model. The camera was disconnected at retrieval; no new hardware acceptance is claimed.

## Source reconciliation

All 82 installed application, package, configuration-template, and script files match the staged Git content after line-ending normalization. The 83rd installed file is the historical README, retained exactly in the private archive; the repository README now points here. `deployment-manifest.json` records raw Pi SHA-256 hashes; `git-source-verification.json` records matching Git content hashes. This captures deployed calibration/control, firing history, shot recording, storage accounting, and reduced automatic recording changes. Four earlier local Phase 4.7/4.8 commits are also now included in the published runtime history.

The original development worktree remains untouched. Its unfinished calibrated-ROI changes were not deployed and are not included. Tests were recovered from the existing September recording qualification set, with fixtures aligned to the installed watch-zone API. Undeployed ROI tests and the separate historical package-builder tests are not included. No production code was altered to make tests pass.

## Final backup

Private local backup: `C:/Users/swallace/Documents/Squirrel_Squirter_Final_Pi_Backup_2026-10-17/`.

The earlier D: archive drive was unavailable. This is a complete copy of the currently remaining Squirrel project files on the Pi, not a re-verification of older external-drive archives or a full operating-system image.

- Archive: `pi-projects.tar`, 4,411,422,720 bytes.
- SHA-256: `8143b3015275945c472504bef1999ee165ca591ab37a52bd6a4980f338b8ea14`.
- Verified 13,304 regular files, 4,391,396,134 source bytes, and 12 symlinks against the Pi manifest.
- Includes active and legacy project roots, production/reference recordings, evidence, firing histories, all saved configurations/calibrations, model binaries/manifests, source backups, installed virtual environments, and both Squirrel service definitions.
- Also retained service/process metadata, installed Python/system packages, and available system journal entries.
- Pi file sizes and modification times remained unchanged through the copy; the firing database passed SQLite integrity checking.
- `Misc/` was excluded. Two transient GPIO notification FIFOs are recorded separately; they are not regular data files.

No Pi files were deleted. Both `squirrel-collector-phase4-6.service` and `squirrel-squirter.service` are inactive and disabled. There are no user timers and no Squirrel system service was found active. The Pi is ready for the owner's next project; this operation did not reimage it.

## Verification limits

Offline Python suite: 1,014 passed, 2 failed, 1 skipped. JavaScript calibration UI: 9 passed. The Windows host skipped a symlink test. The preserved snapshot has two outstanding assertions: `test_latest_frame_skips_unconsumed_sources_and_preserves_receipt_identity` expected three buffered frames but saw one; `test_runtime_night_blocks_manual_auto_evidence_preview_works_and_resumes` expected an em dash while the deployed HTML contains garbled punctuation. These do not affect the byte/hash backup proof and have not been repaired in this archival snapshot. See `validation.json`; full logs are in the private backup.

## Recovery

`pi-projects.tar` uses paths relative to `/home/bslizzle8552`. Inspect or extract into a separate recovery directory first. On Linux, `tar -xf pi-projects.tar -C /path/to/recovery` restores the project layout, including the exact source, models, recordings and private configuration. Virtual environments are Pi/Python-version-specific; the recorded dependency lists support rebuilding them. The archived service definitions describe the last installation; restoring files does not itself enable or start services.

The field configuration contains the original actuator and automatic-control settings. Keep services disabled during inspection or transfer to another purpose. The backup is historical operational evidence, not confirmation that every detector event was a physical water hit.
