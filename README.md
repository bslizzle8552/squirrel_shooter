# Squirrel Squirter

Squirrel Squirter is currently a **vision-only** Raspberry Pi garden watcher:

**one shared USB camera -> motion groups -> one lightweight event classification -> private review dashboard**

It does not recognize squirrels, aim, move anything, or control water. A small
MobileNet-SSD stress test can label common VOC objects such as `person` and `car`,
but those labels are not squirrel recognition. It imports
no GPIO, I2C, PCA9685, servo, MOSFET, solenoid, or valve driver. The existing
disabled-valve placeholder remains closed and raises an error if asked to open.

The motion watcher's labels remain size/movement heuristics. The optional object
classifier is a separate record and never outputs a definitive `squirrel`
category. Species labels belong only in human review after Stephen inspects the
saved image and clip.

## One camera, two consumers

`python -m squirrel_shooter.app` is the normal entry point. One lock-protected
camera runtime opens `/dev/video*` exactly once, measures the negotiated stream,
and publishes raw frames to the motion processor. The motion processor publishes
annotated frames back to the same runtime for the dashboard's MJPEG stream.
The dashboard stream waits for watcher-annotated frames rather than silently
falling back to raw camera JPEGs. The blue outline is the inclusion-zone boundary;
group and component boxes are drawn over it whenever the watcher sees a candidate.
If a candidate briefly drops out between frames, the live view holds its last box
for the tracker's existing short gap window (0.7 seconds by default) and labels it
`last seen`. This affects only the display and adds no classifier inference.

The dashboard never constructs or reads a `VideoCapture`, and the motion processor
never opens a camera. Camera reconnects happen only inside the shared runtime.
Stopping the complete app finishes event files, flushes logs, rebuilds reports,
stops the dashboard, and finally releases the sole camera handle.

## What the combined system does

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
- Stores each confirmed tracked group as one event folder containing its snapshot,
  clip, and metadata. Manual and standalone pictures remain separate captures.
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
python -m squirrel_shooter.classifier_setup
python -m pytest
```

The setup command downloads the pinned, MIT-licensed MobileNet-SSD definition,
weights, and license (about 23 MB total) and verifies every SHA-256 checksum before
installing them under the ignored `models/` directory. Expected test result for
this revision: **81 passed** without opening the USB camera. Then stop any old
dashboard, preview, recorder, or watcher process that already owns the camera and
start the complete system:

```bash
python -m squirrel_shooter.app
```

On a desktop session, the preview supports:

- `q` - finish active events, rebuild the report, release the camera, and stop;
- `s` - save a manual annotated still;
- `e` - force the current candidate to become a test event;
- `r` - rebuild reports from the event folders already on disk.

From a normal headless SSH session, use:

```bash
python -m squirrel_shooter.app --headless
```

The headless command still runs both motion/event processing and the dashboard.
It only disables the local OpenCV preview window.

`motion_watch` remains as a compatibility command and now uses this same shared
runtime. Do not start it alongside `squirrel_shooter.app`. To intentionally run
motion processing without HTTP, use:

```bash
python -m squirrel_shooter.motion_watch --headless --no-dashboard
```

Press `Ctrl+C` once to stop safely. The watcher finishes active event records when
possible, releases the camera, applies retention, updates the session log, and
refreshes the report. Do not power off the Pi while it is writing an event.

## Open the dashboard through Tailscale

The default listener is `0.0.0.0:5000`, which makes it reachable through the Pi's
Tailnet address while the complete app is running. In another Pi SSH session:

```bash
cd ~/squirrel_shooter
source .venv/bin/activate
tailscale status
tailscale ip -4
ss -ltnp | grep ':5000'
curl --fail http://127.0.0.1:5000/api/status
```

From the PC on the same Tailnet, open either:

```text
http://PI_TAILSCALE_IP:5000
http://PI_MAGICDNS_HOSTNAME:5000
```

Replace the placeholders with the values shown by Tailscale. Do not add router
port forwarding, a public reverse proxy, or public DNS for this dashboard.

Useful routes are:

- `/` - compact live feed, four essential readings, and the five newest events;
- `/events` - live paginated archive of all saved event pictures and clips;
- `/classifier-review` - needs-review, unknown, known, error, and false-positive event views;
- `/captures` - standalone and manual picture archive;
- `/video_feed` - shared annotated MJPEG stream;
- `/api/status` and `/api/health` - camera/motion health and counters;
- `/api/events` - recent completed events;
- `/reports/latest` - latest local HTML review report.

There are no remote aiming, movement, valve, or water controls.

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
        classifier-input.jpg  # exact crop submitted to the object model
        classification.json   # model output, visible label, and human decision
  classifier/                 # legacy evidence preserved during automatic migration
  manual/
  rejections/                 # only when rejection snapshots are enabled
  logs/
    events.csv
    events.jsonl
    classifier.jsonl          # append-only classification and review audit
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
is direct evidence. The `camera.reopen_*` values remain compatibility fallbacks;
the combined app uses the explicit `shared_camera` reconnect settings below.

### Shared runtime and dashboard

```yaml
shared_camera:
  reconnect_enabled: true
  maximum_consecutive_read_failures: 10
  reconnect_delay_seconds: 2.0
  consumer_wait_timeout_seconds: 1.0
  annotated_frame_stale_seconds: 3.0

runtime:
  headless: false
  shutdown_timeout_seconds: 10.0

dashboard:
  enabled: true
  host: 0.0.0.0
  port: 5000
  stream_fps: 8.0
  jpeg_quality: 82
  status_refresh_interval_seconds: 3.0
```

The dashboard stream is capped below the currently observed camera rate to leave
CPU time for detection and event recording. Change `host` or `port` in YAML, or
temporarily override them with `--host` and `--port`. Setting `dashboard.enabled`
to `false` is equivalent to `--no-dashboard`.

### Background learning and motion sensitivity

```yaml
motion:
  history: 500
  variance_threshold: 32
  min_blob_area: 500
  max_blob_area: 150000
  startup_warmup:
    seconds: 12.0
    minimum_frames: 60
  persistence:
    frames: 5
    maximum_gap_seconds: 0.7
    cooldown_seconds: 8.0
```

The first run needs time to learn the background. No normal event is confirmed
during warmup. Raising `variance_threshold`, `min_blob_area`, or persistence makes
the detector less sensitive. Cooldown and gap values are seconds, not assumed
frame counts.

### Candidate event filter

Weak motion remains visible on the dashboard for diagnosis, but these defaults
prevent common noise from becoming a saved event:

```yaml
motion:
  candidate_filter:
    enabled: true
    minimum_frame_percent: 0.15
    ignore_tiny_motion: true
    ignore_plant_or_shadow_flicker: true
    require_coherent_small_motion: true
    small_motion_minimum_travel_pixels: 10.0
```

Small candidates must travel consistently before they can create an event. A
gray box marked `FILTERED` is still useful tuning evidence; the dashboard shows
the specific reason. If real squirrels are being filtered, lower
`small_motion_minimum_travel_pixels` gradually before disabling coherent-motion
filtering. This is motion filtering, not animal or object recognition.

### Low-load event classifier stress test

```yaml
classifier:
  enabled: true
  event_frame_number: 1
  crop_margin_percent: 30.0
  detection_confidence: 0.25
  auto_accept_confidence: 0.60
  auto_accept_labels: [person, car]
  worker_queue_capacity: 1
```

Only a confirmed motion event can submit work. Each event submits exactly once,
using frame 1 by default (frame 2 is also allowed). The group box is expanded by
30 percent to retain useful context and copied into a bounded one-item queue. A
single background worker performs OpenCV DNN inference, so the motion loop never
waits for the normal classifier path. The bounded queue also prevents inference
work from accumulating when the Pi is busy.

The motion event is always saved first. Classification can label it, but can never
discard it. `person` and `car` detections at or above 60 percent are labeled
automatically and remain editable. Weaker person/car detections go to Needs Review.
Every other model class and no detection go to Unknown, where a rabbit, squirrel,
or other unsupported animal remains visible for the normal 30-day event retention.
A high-confidence `dog`, for example, remains Unknown because this MVP only trusts
person and car as visible labels; the raw dog suggestion is still recorded.

Human actions are Confirm Person, Confirm Car, Keep Unknown, and False Positive.
Only a human can mark an event false positive. Model-load and queue errors appear
as Classification unavailable in the Errors view and can be retried from the exact
saved input after the model is available. Existing automatic person/car labels can
also be corrected back to Unknown.

Every attempt records the submitted frame number, crop and source boxes, model,
all detections, confidence, inference time, automatic/manual decision, error, and
exact submitted crop. `captures/logs/classifier.jsonl` is append-only, while each
event's `classification.json` reflects the newest decision. The recent-clips and
event-archive cards read that same file, so their headline becomes Person, Car,
Unknown, Classification unavailable, or False Positive. The original motion label
such as `small_animal_candidate` remains visible only as secondary diagnostic data.
This classifier
uses the VOC label set, which includes people, cars, birds, cats, and dogs but not
squirrels or rabbits.

On first start after upgrading, legacy `captures/classifier/pending`, `accepted`,
and `rejected` records are copied into their matching event folders. Old files are
not deleted. A previous classifier rejection becomes Unknown rather than being
lost or treated as proof that nothing useful was present.

Watch `/api/status` during the supervised stress test. It reports classifier queue
depth, completed count, last inference time, and last error. If motion processing
rate drops materially or the queue fills, set `classifier.enabled: false`; the
motion/event system continues unchanged.

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
8. Confirm no separately started `motion_watch`, `web_dashboard`, preview, or
   recorder process is competing with the combined app for the camera.
9. Confirm the disabled valve remains closed and no physical-output hardware has
   been added or energized.

The automated suite uses generated NumPy/OpenCV frames and temporary directories;
it never requires the physical USB camera:

```bash
python -m pytest
```

Real outdoor tuning still must be done on the Raspberry Pi with the actual fixed
camera, lighting, plants, storage, temperature, and USB behavior.

## Known limitations

- The dashboard is private by deployment convention, not an authentication layer;
  keep it inside Tailscale and do not expose port 5000 publicly.
- A camera reconnect resets its short-term FPS smoothing. MOG2 continues safely,
  but a changed camera angle may cause warmup or global-motion rejection.
- Browser MJPEG display and MJPG AVI playback depend on the client's codec support.
- Heuristic categories still require human review and are not species recognition.
