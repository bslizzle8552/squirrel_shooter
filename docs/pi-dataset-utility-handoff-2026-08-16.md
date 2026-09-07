# Pi historical dataset utility — pass summary

Date: 2026-08-16

Authoritative checkout: `C:\dev\squirrel-squirter`

Branch inspected: `manual-control`

## Outcome

The repository now contains a focused, read-only Raspberry Pi extraction utility and a PC-side inventory/reporting utility. The real field pull was **not** run, and no Pi, camera, GPIO, servo, valve, or live detection behavior was touched.

### Windows password-authenticated SFTP correction

The first real operator attempt exposed a Windows OpenSSH behavior that local synthetic tests had not exercised: SFTP's `-b` option automatically enabled SSH `BatchMode=yes`, so the separate transfer connection could not request the Pi password even though the preceding SSH inventory connection and a manual interactive SFTP login succeeded. The collector now passes `BatchMode=no` before `-b`. OpenSSH uses the first supplied value, preserving password/passphrase prompting while retaining SFTP batch-file execution and fail-on-transfer-error behavior. The redundant `-q` flag was removed so future authentication diagnostics remain visible. No Pi-side change is involved.

An earlier uncommitted implementation was already present in this checkout. This pass audited it against the current repository data architecture and corrected four material gaps:

1. Append-only `events.jsonl` and `classifier.jsonl` history is now inventoried. If retention removed an event folder but its log record remains, the review index recovers one metadata-only event row instead of silently losing that history.
2. Human-reviewed samples marked `training_eligible: false` remain visible for audit but are no longer counted as usable custom-classifier training truth.
3. Service status and journal collection now use the project's actual per-user systemd service (`systemctl --user` / `journalctl --user`).
4. Retention evidence and aggregated dropped/filtered-candidate evidence are now reported more precisely in the inventory and wind/shadow report.

## Exact command to run later

From `C:\dev\squirrel-squirter` on the development PC:

```powershell
.\.venv\Scripts\python.exe -m squirrel_shooter.pi_data_pull collect --host PI_TAILSCALE_HOSTNAME_OR_IP
```

Replace `PI_TAILSCALE_HOSTNAME_OR_IP` with the Pi's current MagicDNS hostname or Tailscale IP. That is the only connection value still required.

Previously verified defaults used by the command:

- Pi user: `bslizzle8552`
- Pi repository: `/home/bslizzle8552/squirrel_shooter`
- SSH port: `22`
- service: `squirrel-squirter.service`

If the deployment moved, override the values explicitly:

```powershell
.\.venv\Scripts\python.exe -m squirrel_shooter.pi_data_pull collect --host PI_HOST --user PI_USER --remote-root /actual/repository/path
```

## Extraction utility

`src/squirrel_shooter/pi_data_pull.py` performs a PC-initiated SSH/SFTP pull. The Pi only:

- lists known data files with path, size, and modification time;
- reads requested files through SFTP;
- returns a sanitized configuration source to the PC;
- reports lightweight hostname, timezone, Git, model-file, and user-service context;
- returns a bounded service-journal excerpt.

The Pi does **not** decode video, extract frames, generate thumbnails, run inference, calculate whole-dataset hashes, compress an archive, transcode media, alter configuration, or delete/move/relabel evidence.

One second lightweight file listing detects data that changed during the copy and retransfers those files once. Missing or shorter-than-listed local copies are recorded as validation issues.

## Exact repository-defined data categories collected

- Complete retained `captures/` tree, including:
  - motion event folders and `event.json`;
  - annotated `snapshot.jpg` and `clip.avi`;
  - clean `original-frame.jpg` and exact `classifier-input.jpg`;
  - `classification.json` with detections, confidence, target, track, and review metadata;
  - manual-fire and automatic-fire event folders, zoom/full clips, snapshots, targeting evidence, and source-event relationships;
  - human-reviewed `training-dataset` samples and `manifest.jsonl`;
  - legacy `classifier/` evidence;
  - optional rejection images;
  - application, event, classifier, rejection, and session logs;
  - generated reports;
  - `auto-fire-rate-limit.json` state when present;
  - standalone/manual capture images retained at the capture root.
- Legacy repository-defined roots when present: `logs/`, `recordings/`, `data/captures/`, and `data/videos/`.
- Configured optional detector `debug/` output.
- `config/calibration_points.json`.
- Root camera test still/video artifacts named by repository rules.
- Sanitized `config/default.yaml`, including motion settings, classifier settings, inclusion zone/mask geometry, camera/night/recording settings, and retention configuration.
- Bounded 30-day per-user service journal.
- Pi hostname, timezone, collection time, Git branch/commit, user-service state, and classifier-model file metadata.

The actual capture/classifier dataset is not time-limited by the utility. It copies all history still retained in these roots.

## Data intentionally excluded

- Git objects and ordinary application source already present on the development PC.
- Virtual environments, caches, operating-system data, and unrelated files outside repository-defined data roots.
- Full classifier model binaries; path, size, and modification time are recorded instead.
- Likely secret/token/password/credential values; configuration is sanitized on the PC before it is written.
- System-journal entries older than the selected bounded window; retained project logs are copied in full.
- Separate Pi-side backup archives outside the repository-defined roots are not guessed or pulled automatically.

No original evidence is flattened, resized, recompressed, transcoded, or relabeled.

## Local snapshot and incremental behavior

Each run creates:

```text
analysis/pi_dataset_YYYY-MM-DD_HHMMSS/
  raw/
    project/                 exact Pi project-relative relationships
    config/default.sanitized.yaml
    logs/service/squirrel-squirter-journal.txt
  inventory/
    remote_files.json
    collection_context.json
    config_redactions.json
    collection_scope.json
    inventory.json
    review_index.csv
    review_index.json
    classifier_progress.md
    wind_shadow_diagnostic.md
    README.md
```

Later runs compare exact relative path, size, and modification time against the newest prior snapshot. Unchanged files are copied locally from that snapshot; only new/changed files are retransmitted. Older local snapshots are never deleted.

## PC-side inventory outputs

`inventory.json` reports:

- distinct retained/recoverable events and current event metadata files;
- images, videos, classifier decisions, total copied size, and retained time range;
- raw event-log and classifier-audit record counts;
- metadata-only events recovered after media retention;
- classifier audit action counts;
- per-predicted-label confidence statistics and high/low counts;
- accepted/rejected counts and unique event/track counts;
- discovered predicted, human, and training labels;
- unknown/no-detection/error/low-confidence/negative evidence;
- manual and automatic shot recording counts;
- day/night metadata coverage;
- retained retention/deletion evidence and its limitations.

`review_index.csv` and `review_index.json` contain review-friendly rows with event/track IDs, timestamps, prediction metadata, complete detections JSON, media paths, day/night evidence, decision/shot state, human label, training label, explicit training eligibility, target coordinates, and notes. Missing information remains empty. Event/classifier logs can restore metadata-only rows, but cannot recreate media that retention already deleted.

`classifier_progress.md` separates:

- raw retained media count;
- human-verified **and training-eligible** samples;
- conservative independent event groups;
- squirrel distance/size, position, lighting, and motion diversity when metadata permits;
- verified negatives/confusers, imbalance, obvious gaps, and required manual review.

`wind_shadow_diagnostic.md` marks each requested diagnostic `AVAILABLE NOW`, `PARTIALLY AVAILABLE`, or `NOT CURRENTLY RETAINED` from the copied evidence. Current code can retain sampled global foreground percentages, event component/group data, session aggregate contours/candidates/filter rejections, saved track IDs, FPS evidence, and periodic classifier queue/load telemetry. It does not retain a complete per-frame foreground/candidate/tracker stream or individual decisions for every rejected candidate.

## Labels/categories determinable before the real pull

Actual labels present in field data will be discovered only after collection. The current code defines these possible categories:

- MobileNet-SSD predictions: `background`, `aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, `tvmonitor`.
- Human review suggestions: `squirrel`, `person`, `car`, `rabbit`, `deer`, `other_animal`, `chipmunk`, `bird`, `raccoon`, `opossum`, `groundhog`, `fox`, `skunk`, `cat`, `dog`; safe custom human labels may also exist.
- Canonical verified negative: `background_or_false_positive`.
- Review views: `review`, `unknown`, `known`, `errors`, `false_positive`.
- Motion heuristics, not species truth: `plant_or_shadow_flicker`, `tiny_motion`, `person_sized`, `large_object`, `small_animal_candidate`, `medium_animal_candidate`, `unclassified_motion`.
- Candidate rejection reasons: `localized_lighting_change`, `tiny_motion`, `plant_or_shadow_flicker`, `below_minimum_frame_percent`, `small_motion_not_coherent`, `small_motion_insufficient_travel`.
- Global rejection reasons: `probable_ir_mode_switch`, `probable_exposure_change`, `probable_scene_obstruction`, `probable_camera_movement`, `excessive_zone_motion`, `excessive_frame_motion`, `global_motion_unclassified`.

The actual extracted label counts remain unknown because the real Pi pull was intentionally not run.

## Retention limitation

Current raw-event retention is bounded to 30 days, 1,000 complete events, and 4,096 MB, oldest complete event first. Logs and session summaries are also rotated/bounded. The utility recovers all currently retained media and can recover some older metadata from remaining logs, but it cannot reconstruct already deleted images/videos or guarantee a lifetime deletion total after log rotation.

Human-verified samples under `captures/training-dataset` are outside normal raw-event retention and are copied in full.

## Files changed for this utility

- `src/squirrel_shooter/pi_data_pull.py`
- `src/squirrel_shooter/dataset_inventory.py`
- `tests/test_pi_dataset.py`
- `docs/pi-dataset-collection.md`
- `docs/pi-dataset-utility-handoff-2026-08-16.md`
- `README.md`

Existing unrelated owner changes were preserved. `Misc/` was not searched, opened, edited, staged, cleaned, or otherwise touched during this pass.

## Validation performed

- Focused dataset tests: **7 passed**.
- Full repository suite: **317 passed**.
- Byte compilation of `src` and `tests`: passed.
- Source distribution and wheel build: passed.
- `git diff --check`: passed.
- CLI help/argument parsing: passed.
- Synthetic local validation covered current events, manual/automatic shots, human truth versus predictions, excluded training samples, orphan event recovery from rotated logs, classifier audit recovery, retention evidence, wind/shadow evidence, incremental reuse, configuration redaction, path safety, per-user service commands, and password-enabled SFTP batch option ordering/cleanup.

These are software/local-copy checks only. They do not prove SSH credentials, Tailscale reachability, field Pi storage contents, transfer duration, or physical hardware behavior.

## Ground-truth rule

**CURRENT CLASSIFIER PREDICTIONS WILL BE PRESERVED AS METADATA, NOT ASSUMED TO BE GROUND-TRUTH TRAINING LABELS.**

Only explicit human verification is treated as truth, and only currently training-eligible samples count toward usable custom-classifier progress.
