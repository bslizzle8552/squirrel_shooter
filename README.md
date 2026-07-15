# Squirrel Squirter

Squirrel Squirter is a Raspberry Pi 4 garden-camera project. The current build is
strictly the robot's **eyes**:

**camera -> motion detection -> debug visualization -> event snapshots -> diagnostics**

It does not recognize squirrels, aim, move servos, access GPIO/I2C/PCA9685, or
control water. The disabled valve placeholder still defaults to closed and refuses
to open.

## What works

- One shared camera service owns the physical camera; every browser and the vision
  worker consume shared frames.
- Lightweight OpenCV MOG2 detection runs at a configurable reduced resolution.
- Blur, shadow removal, morphological cleanup, contour sizing, rectangular ROI,
  multi-frame persistence, cooldown, and frame-wide lighting reset are configurable.
- The live MJPEG feed shows boxes, centers, areas, decision reasons, detector state,
  processing FPS, blob count, persistence, and the ROI.
- Startup remains in **LEARNING** with events suppressed until the background model
  is ready.
- Accepted motion saves one annotated JPEG, subject to cooldown, into the existing
  capture gallery.
- The read-only dashboard and APIs report camera, detector, storage, event, uptime,
  and Pi-temperature health. Stale camera frames are hidden rather than presented
  as live.
- JSON logs explain startup, shutdown, camera details/failures, detector decisions,
  warm-up, cooldown, lighting resets, snapshots, retention, and exceptions.
- Optional rate-limited debug images are off by default.
- Old event captures, debug images, and log files are deleted oldest-first according
  to configured limits.

## Raspberry Pi deployment

For a new Pi, install system packages and clone once:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip v4l-utils libgl1 libglib2.0-0
git clone https://github.com/bslizzle8552/squirrel_shooter.git ~/squirrel_shooter
cd ~/squirrel_shooter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For each deployment to the existing Pi checkout:

```bash
cd ~/squirrel_shooter
git pull --ff-only origin main
source .venv/bin/activate
python -m pip install -e .
python -m squirrel_shooter.web_dashboard
```

If the repository was cloned somewhere else, change only the first `cd`. Keep that
terminal open; `Ctrl+C` stops both workers and releases the camera cleanly.

In a second SSH session, verify the private listener and read-only health APIs:

```bash
cd ~/squirrel_shooter
source .venv/bin/activate
ss -ltnp | grep ':5000'
curl --fail http://127.0.0.1:5000/api/status
curl --fail http://127.0.0.1:5000/api/health
curl --fail http://127.0.0.1:5000/api/recent-events
tailscale ip -4
```

From a device on the same Tailnet, open:

```text
http://PI_TAILSCALE_IP:5000
```

Do not expose port 5000 through router forwarding or a public proxy.

## Camera checks on the Pi

The physical camera is expected to be attached only to the Pi. Identify and verify
it there before running the dashboard:

```bash
cd ~/squirrel_shooter
source .venv/bin/activate
v4l2-ctl --list-devices
ls -l /dev/video*
python -m squirrel_shooter.camera_diagnostic
```

Only one process should own the USB camera. The existing preview and recorder remain
available when the dashboard is stopped:

```bash
python -m squirrel_shooter.camera_preview
python -m squirrel_shooter.camera_capture --seconds 10
```

## Configuration

All detector tuning lives in `config/default.yaml`; there are no hidden detection
thresholds in the worker.

```yaml
camera:
  device_index: 0
  requested_width: 1280
  requested_height: 720
  requested_fps: 30
  output_directory: captures

motion:
  enabled: true
  processing_width: 640
  learning_frames: 90
  history: 300
  variance_threshold: 25
  detect_shadows: true
  blur_kernel: 7
  morphology_kernel: 5
  open_iterations: 1
  close_iterations: 2
  min_blob_area: 450
  max_blob_area: 90000
  persistence_frames: 3
  persistence_max_distance: 80
  cooldown_seconds: 8
  lighting_change_percent: 65
  recent_event_limit: 100
  roi:
    enabled: false
    x: 0.0
    y: 0.0
    width: 1.0
    height: 1.0
```

ROI values are normalized from `0.0` to `1.0`, so they do not depend on the camera
resolution. Blob areas and persistence distance are measured in the internally
processed image, whose width is `processing_width`.

Optional debug outputs are individually configurable under
`motion.debug_outputs`. Their defaults are all `false`. Storage limits and health
staleness thresholds are under `storage` and `health`. Set `logging.level: DEBUG`
temporarily when every rejected frame-level candidate is needed in the JSON logs.

### Outdoor starting ranges

The checked-in values are safe initial estimates, not camera-specific calibration.
For the first daylight garden test, adjust one value at a time:

- `learning_frames`: 90-180 (3-6 seconds at 30 FPS)
- `variance_threshold`: 25-40; higher rejects more subtle pixel change
- `min_blob_area`: 450-1,500 at 640-pixel processing width
- `max_blob_area`: 60,000-120,000; lower it if sky/shadow changes dominate
- `persistence_frames`: 3-5
- `persistence_max_distance`: 60-120 pixels
- `cooldown_seconds`: 8-15
- `lighting_change_percent`: 55-70

Enable and tighten the ROI as soon as the fixed garden framing is known. Windy
foliage, rain, insects near the lens, moving shadows, automatic exposure changes,
and night infrared behavior will require real-camera tuning.

## Read-only APIs

- `/api/status` - dashboard summary
- `/api/health` - worker, frame, event, storage, and error counters
- `/api/recent-events` - newest accepted/rejected candidate decisions

There are no control endpoints or manual capture button.

## Tests

The automated suite never opens a real camera. It uses generated frames, synthetic
masks, and injected services/failures:

```bash
cd ~/squirrel_shooter
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest
```

Real verification still belongs on the Raspberry Pi: actual resolution/FPS,
lighting warm-up, outdoor false positives, ROI geometry, long-running CPU and
temperature, USB-camera recovery, snapshot permissions, storage retention, and
Tailscale browser behavior.
