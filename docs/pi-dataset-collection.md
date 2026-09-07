# Raspberry Pi historical dataset collection

This utility makes a read-only, PC-initiated copy of the useful retained Squirrel Squirter data, then performs all inventory and analysis work on the development PC. It does not alter the Pi, run inference, decode video, extract frames, hash the full dataset, or create a Pi-side archive.

## Command to run later

From `C:\dev\squirrel-squirter` on the development PC:

```powershell
.\.venv\Scripts\python.exe -m squirrel_shooter.pi_data_pull collect --host PI_TAILSCALE_HOSTNAME_OR_IP
```

Replace only `PI_TAILSCALE_HOSTNAME_OR_IP` with the Pi's current MagicDNS hostname or Tailscale IP. That connection value is not established in the repository and was intentionally not invented.

Password authentication is supported by the Windows OpenSSH client. The collector deliberately places `BatchMode=no` before SFTP's `-b` option so the SFTP transfer connection can request a password while retaining fail-on-error batch command handling. SSH inventory/context reads and the SFTP transfer are separate read-only connections, so password-only setups may prompt more than once during a complete collection. Key/agent authentication remains optional, not required.

The established Pi account and repository path are the defaults:

- user: `bslizzle8552`
- repository: `/home/bslizzle8552/squirrel_shooter`
- SSH port: `22`

Override them explicitly if the deployment moved:

```powershell
.\.venv\Scripts\python.exe -m squirrel_shooter.pi_data_pull collect --host PI_HOST --user PI_USER --remote-root /actual/repository/path
```

Do not run the collection while troubleshooting field load. The intended time is after night mode has paused recording/classification and activity has settled.

## What is copied

The collector uses paths established by the code, configuration, documentation, and repository ignore rules. If a path does not exist on the Pi, it is skipped.

- `captures/` in full, preserving all nested relationships:
  - `events/YYYY-MM-DD/<event-id>/event.json`
  - annotated `snapshot.jpg` and `clip.avi`
  - clean `original-frame.jpg`
  - exact `classifier-input.jpg` and `classification.json`
  - manual-fire and auto-fire event folders, zoom/full clips, snapshots, and event metadata
  - `training-dataset/manifest.jsonl` and human-verified sample folders
  - legacy `classifier/` evidence
  - manual stills, optional rejection images, generated reports, application/event/classifier/rejection logs, session summaries, and the auto-fire rate-limit state
- legacy data layouts named by the repository: `logs/`, `recordings/`, `data/captures/`, and `data/videos/`
- configured optional detector `debug/` evidence
- `config/calibration_points.json`
- root-level camera test videos/stills named by the repository ignore rules
- a sanitized copy of `config/default.yaml`
- a bounded 30-day `squirrel-squirter.service` journal excerpt
- lightweight hostname, timezone, Git branch/commit, user-service state, collection time, and classifier-model file metadata

This captures the complete retained history of actual captures/classifier evidence rather than imposing a recent-time cutoff. The service journal alone is time-bounded because it may be extremely large and the retained application logs are already copied in full.

## What is deliberately excluded

- secrets: likely credential/token/password keys are redacted on the PC before configuration is written
- the full MobileNet-SSD model binaries; their path, size, and modification time are recorded instead
- Git objects, source files already on the PC, virtual environments, caches, and unrelated Pi/home-directory data
- unrelated operating-system journals

No Pi file is deleted, moved, relabeled, rewritten, compressed, resized, transcoded, rotated, or timestamp-modified.

## Local snapshot layout

Each run creates `analysis/pi_dataset_YYYY-MM-DD_HHMMSS/`:

```text
raw/
  project/                 exact remote project-relative data relationships
    captures/
    config/calibration_points.json
    ...legacy roots when present...
  config/
    default.sanitized.yaml
  logs/service/
    squirrel-squirter-journal.txt
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

To rebuild the PC-side reports without contacting the Pi:

```powershell
.\.venv\Scripts\python.exe -m squirrel_shooter.pi_data_pull inventory analysis\pi_dataset_YYYY-MM-DD_HHMMSS
```

## Incremental behavior

Before downloading, the PC asks the Pi only for relative file paths, sizes, and modification times. If the newest older local snapshot has an exact path/size/mtime match, the file is copied locally into the new timestamped snapshot instead of retransmitted. New and changed files are fetched through one SFTP batch. Nothing is deleted from an older local snapshot.

After the transfer, one second lightweight remote listing detects files that changed while the running service was being copied; only those files are fetched once more. Any missing or shorter-than-listed local file is recorded as a validation issue. No full-dataset hashes are calculated on the Pi.

## Inventory and review outputs

`inventory.json` reports total events, images, videos, classifier records, storage, retained time range, per-predicted-class confidence statistics, high/low-confidence counts, accepted/rejected counts, and unique event/track counts. It separately reports raw append-only `events.jsonl` and `classifier.jsonl` record counts, classifier audit actions, retained deletion/retention evidence, and metadata-only historical events recovered from logs after their media folders disappeared.

`review_index.csv` and `review_index.json` provide event/example rows with timestamp, event/track IDs, predicted label, confidence, media paths, day/night evidence, accepted/rejected state, shot state, human label, training label, explicit training eligibility, target pixel, and all retained detections. When an event folder has aged out but its append-only event/classifier logs remain, one metadata-only row preserves the recoverable history. Missing facts stay empty rather than being invented.

`classifier_progress.md` separates raw media counts from human-verified, training-eligible samples and conservative unique-event groups. Excluded/withdrawn samples remain visible in the review index but are not counted as usable training truth. The report summarizes retained class balance, squirrel size/position/day-night/motion metadata when available, negatives/confusers, and manual-review gaps.

`wind_shadow_diagnostic.md` marks each requested diagnostic as `AVAILABLE NOW`, `PARTIALLY AVAILABLE`, or `NOT CURRENTLY RETAINED` using evidence actually found in the copied snapshot.

## Current code-defined labels and categories

The actual labels present on the Pi will be discovered from copied metadata. Before that pull, the repository establishes these possible values:

- MobileNet-SSD predictions: `background`, `aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, `tvmonitor`.
- Human review suggestions: `squirrel`, `person`, `car`, `rabbit`, `deer`, `other_animal`, `chipmunk`, `bird`, `raccoon`, `opossum`, `groundhog`, `fox`, `skunk`, `cat`, `dog`; other safe human-entered labels may also exist.
- Canonical verified negative: `background_or_false_positive`.
- Classifier review views: `review`, `unknown`, `known`, `errors`, `false_positive`.
- Motion-only heuristic categories, not species labels: `plant_or_shadow_flicker`, `tiny_motion`, `person_sized`, `large_object`, `small_animal_candidate`, `medium_animal_candidate`, `unclassified_motion`.
- Candidate rejection reasons: `localized_lighting_change`, `tiny_motion`, `plant_or_shadow_flicker`, `below_minimum_frame_percent`, `small_motion_not_coherent`, `small_motion_insufficient_travel`.
- Global rejection reasons: `probable_ir_mode_switch`, `probable_exposure_change`, `probable_scene_obstruction`, `probable_camera_movement`, `excessive_zone_motion`, `excessive_frame_motion`, `global_motion_unclassified`.

The code automatically accepts only configured current-model labels (`person` and `car` in the current classifier review configuration). Unsupported animals such as squirrels remain predictions/unknowns until a human supplies truth.

## Retention limitation

Current raw-event retention is bounded to 30 days, 1,000 complete events, and 4,096 MB, deleting the oldest complete event first. Application logs and session summaries are also rotated/bounded. Therefore the pull can recover all history still retained, but it cannot reconstruct data already removed by earlier retention.

Human-verified samples under `captures/training-dataset` are outside raw-event retention and are copied in full. Session retention actions and retained event/classifier logs may prove that older raw events existed or were deleted, and the inventory preserves their remaining metadata, but they cannot recreate removed media. Rotated session/log files also cannot prove a lifetime deletion total.

## Ground-truth rule

**CURRENT CLASSIFIER PREDICTIONS WILL BE PRESERVED AS METADATA, NOT ASSUMED TO BE GROUND-TRUTH TRAINING LABELS.**

Only records explicitly marked human-verified are counted as training truth. Nearby frames from one event are grouped by event/session rather than treated as independent examples.
