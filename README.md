# Squirrel Squirter

Squirrel Squirter is currently a **vision-only** Raspberry Pi garden watcher:

**USB camera -> motion groups -> provisional heuristics -> snapshots/clips -> review report**

It does not recognize squirrels, aim, move anything, or control water. It imports
no GPIO, I2C, PCA9685, servo, MOSFET, solenoid, or valve driver. The existing
disabled-valve placeholder remains closed and raises an error if asked to open.

Every label produced by the watcher is a size/movement heuristic. It never outputs
a definitive `squirrel` category. Species labels belong only in the human-review
CSV after Stephen reviews the saved image and clip.

## What the watcher does

- Uses OpenCV MOG2 with configurable history, variance threshold, shadow removal,
  warmup, morphology, contour limits, and a polygon inclusion zone.
- Groups nearby, consistently moving fragments before measuring or classifying
  them. Original contours and component boxes remain in `event.json`.
- Tracks multiple candidate groups with centroid paths, persistence, duration,
  speed, direction, distance, area, component count, and grouping confidence.
- Rejects scene-wide pixel changes using actual foreground coverage in both the
  full frame and inclusion zone. A large outer rectangle around two small objects
  is not used as the measurement.
- Records an annotated JPEG and MJPG AVI with a time-based pre-event buffer and
  post-event tail.
- Appends CSV, JSON Lines, rejection, and session logs and flushes completed events.
- Preserves interrupted event folders for diagnosis on the next start.
- Deletes the oldest **complete** events first according to age, count, and storage
  limits. Active and recovered-incomplete events and logs are protected.
- Builds local HTML/Markdown reports and an editable review CSV without a database
  or web server.

The camera currently delivers approximately 9.9-10 FPS at 1280x720. The watcher
measures the real rate and uses monotonic elapsed time for warmup, cooldown,
recovery, event duration, pre-roll, and post-roll. Frame counts are used only for
consecutive-frame persistence and the minimum warmup sample. It also works when
the negotiated rate changes.

Low FPS does **not** imply infrared or night mode. Camera mode and explicit IR mode
are recorded separately as `unknown` unless configuration or actual image evidence
says otherwise. `probable_ir_mode_switch` requires a scene-wide visual transition;
frame rate alone can never produce it.

## Deploy the changes to the Raspberry Pi

The repository and GitHub remain the source of truth. Run these exact commands in
the existing Pi checkout (change only the first path if the clone is elsewhere):

```bash
cd ~/squirrel_shooter
git pull --ff-only origin main
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest
```

Expected test result for this revision: all tests pass without opening the USB
camera. Then stop any dashboard, preview, or recorder that already owns the camera
and launch the watcher:

```bash
python -m squirrel_shooter.motion_watch
```

On a desktop session, the preview supports:

- `q` - finish active events, rebuild the report, release the camera, and stop;
- `s` - save a manual annotated still;
- `e` - force the current candidate to become a test event;
- `r` - rebuild reports from the event folders already on disk.

From a normal headless SSH session, use:

```bash
python -m squirrel_shooter.motion_watch --headless
```

Press `Ctrl+C` once to stop safely. The watcher finishes active event records when
possible, releases the camera, applies retention, updates the session log, and
refreshes the report. Do not power off the Pi while it is writing an event.

To rebuild the reports later, with no camera required:

```bash
cd ~/squirrel_shooter
source .venv/bin/activate
python -m squirrel_shooter.event_report
```

## Output locations

```text
captures/
  events/
    YYYY-MM-DD/
      <timestamp-and-unique-suffix>/
        snapshot.jpg
        clip.avi
        event.json
  manual/
  rejections/                 # only when rejection snapshots are enabled
  logs/
    events.csv
    events.jsonl
    rejections.jsonl
    sessions/
      session-<id>.json
  reports/
    latest-report.html
    latest-report.md
    review.csv
```

Open `captures/reports/latest-report.html` directly in a browser; it needs no
server. Enter a human label and notes in `review.csv`. Suggested labels are
`squirrel`, `bird`, `person`, `dog`, `plant`, `shadow`, `bug`, and `unknown`, but
custom labels are allowed. Rebuilding preserves labels already entered in that
CSV. The heuristic label remains separate.

An interrupted folder keeps `.recovered-incomplete` and a recovery `event.json`.
It is intentionally excluded from normal counts and retention so Stephen can
inspect or remove it manually.

## Important configuration

All settings are in `config/default.yaml`. Defaults are conservative starting
values for 1280x720 near 10 FPS, not final garden calibration.

### Camera and measured rate

```yaml
camera:
  requested_width: 1280
  requested_height: 720
  requested_fps: 30
  camera_mode_if_known: unknown
  ir_mode_if_explicitly_detected_or_configured: unknown
  low_fps_threshold: 15.0
  reopen_after_failed_reads: 10
  reopen_delay_seconds: 2.0
```

`requested_fps` is only the USB request. `event.json` separately records actual
dimensions and `measured_camera_fps`. Do not change either mode field unless there
is direct evidence.

### Background learning and motion sensitivity

```yaml
motion:
  history: 500
  variance_threshold: 32
  min_blob_area: 300
  max_blob_area: 150000
  startup_warmup:
    seconds: 12.0
    minimum_frames: 60
  persistence:
    frames: 3
    maximum_gap_seconds: 0.7
    cooldown_seconds: 8.0
```

The first run needs time to learn the background. No normal event is confirmed
during warmup. Raising `variance_threshold`, `min_blob_area`, or persistence makes
the detector less sensitive. Cooldown and gap values are seconds, not assumed
frame counts.

### Inclusion-zone polygon

The default zone is the whole frame. To use a garden polygon, set `enabled: true`
and replace the normalized points in clockwise or counterclockwise order:

```yaml
motion:
  inclusion_zone:
    enabled: true
    polygon:
      - [0.10, 0.20]
      - [0.90, 0.20]
      - [0.82, 0.92]
      - [0.18, 0.92]
```

`[0.0, 0.0]` is top-left and `[1.0, 1.0]` is bottom-right. Keep at least three
points, all between 0 and 1.

### Fragment grouping

```yaml
motion:
  grouping:
    max_horizontal_gap_pixels: 80
    max_vertical_gap_pixels: 80
    expanded_box_margin_pixels: 40
    max_centroid_distance_pixels: 140
    direction_similarity_degrees: 50
    speed_similarity_ratio: 2.5
    maximum_components_per_group: 12
```

Increase gaps gradually if a fence or plant splits one object into several events.
Decrease them if two nearby objects are merged. Direction and speed checks prevent
nearby objects traveling differently from being merged merely because they share
a frame.

### Global-motion rejection

```yaml
motion:
  global_rejection:
    max_frame_motion_percent: 35.0
    max_zone_motion_percent: 45.0
    recovery_seconds: 3.0
    log_rejected_global_events: true
    save_debug_snapshot: false
```

Lower the thresholds if camera vibration, wind-driven plants, exposure jumps, or
obstruction are being recorded as objects. Raise them cautiously if a legitimate
person-sized object is rejected. The percentages are changed pixels, not a union
bounding rectangle.

### Event clips and provisional categories

```yaml
motion:
  event_lifecycle:
    pre_event_seconds: 2.0
    post_event_seconds: 3.0
    maximum_event_seconds: 60.0
    clip_codec: MJPG
  provisional_classification:
    tiny_max_frame_percent: 0.08
    small_animal_max_frame_percent: 1.5
    medium_animal_max_frame_percent: 5.0
    person_min_height_percent: 35.0
    large_object_min_frame_percent: 5.0
```

Tune these only after reviewing real garden clips. Categories are
`tiny_motion`, `small_animal_candidate`, `medium_animal_candidate`,
`large_object`, `person_sized`, `plant_or_shadow_flicker`, and
`unclassified_motion`.

### Retention

```yaml
retention:
  maximum_storage_megabytes: 4096
  maximum_event_age_days: 30
  maximum_event_count: 1000

logging:
  maximum_active_log_megabytes: 100
  retained_log_rotations: 5
```

Set limits below the SD card's actual free space. Logs are append-only and active;
when an active event/rejection log reaches its limit it is closed and rotated, and
only the configured number of inactive rotations is retained. The current active
log is never deleted. Old session summaries are also bounded by
`storage.max_log_files`.

## Expected false positives and rejections

The first outdoor runs are tuning sessions. False events are expected. Moving
plants, shadows, rain, insects near the lens, and fragmented masks can all look
like motion. Camera vibration, large exposure changes, scene obstruction, many
wind-driven plant regions, and a real IR-cut visual transition can produce global
rejections. Measured FPS may vary, and approximately 10 FPS is currently observed.
Low FPS alone does not indicate night mode.

## Before leaving it unattended

1. Fix the camera firmly and confirm the lens and inclusion zone show only the
   intended garden area.
2. Verify the Pi clock/time zone; event IDs and reports use local timestamps.
3. Run the full tests and a supervised 10-15 minute watcher session.
4. Press `e` during a real candidate and confirm `snapshot.jpg`, `clip.avi`, and
   `event.json` all open correctly.
5. Stop with `q` or `Ctrl+C`, rebuild the report, and confirm its links work.
6. Review `rejections.jsonl` for repeated vibration/exposure/plant rejection.
7. Check `df -h`, capture-directory permissions, retention limits, Pi temperature,
   stable USB negotiation, and reliable power.
8. Confirm no dashboard, preview, or other process is competing for the camera.
9. Confirm the disabled valve remains closed and no physical-output hardware has
   been added or energized.

The automated suite uses generated NumPy/OpenCV frames and temporary directories;
it never requires the physical USB camera:

```bash
python -m pytest
```

Real outdoor tuning still must be done on the Raspberry Pi with the actual fixed
camera, lighting, plants, storage, temperature, and USB behavior.
