# Squirrel Shooter

Squirrel Shooter is a DIY Raspberry Pi garden sprinkler deterrent. The long-term
sequence is:

**see → detect → aim → dry-fire → spray**

Tonight's version deliberately stops at **see**. It helps us bring up a Raspberry
Pi 4, verify an outdoor UVC USB camera, preview it, and record short annotated
test videos. It is a clean base for later phases without pretending the turret or
water system is ready.

## What works now

- YAML camera configuration with a conservative initial target of 1280×720 at 30 FPS
- Live OpenCV preview with timestamp, actual frame resolution, and measured FPS
- `q` to quit the preview and `s` to save an annotated JPEG
- Headless MJPG/AVI test recording for a Pi without a desktop window
- A diagnostic that lists `/dev/video*` devices and reads one frame
- Safe interface placeholders for detection, motion, modes, and valve control
- Camera-independent tests for configuration, filenames, timing, and fail-safe defaults

There is no squirrel detection, servo movement, GPIO, I2C, PCA9685, MOSFET, or
valve output in this version. Water control is intentionally not implemented.
The placeholder valve controller defaults to closed and refuses to open.

## Raspberry Pi setup

These commands assume current 64-bit Raspberry Pi OS with Python 3.11 or newer.
Open a terminal on the Pi and install the small set of system tools OpenCV and the
camera checks need:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip v4l-utils libgl1 libglib2.0-0
```

Clone and enter the repository:

```bash
git clone https://github.com/bslizzle8552/squirrel_shooter.git
cd squirrel_shooter
```

Create an isolated Python environment, activate it, and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Every new terminal session needs these two commands before running the project:

```bash
cd squirrel_shooter
source .venv/bin/activate
```

## Connect and identify the USB camera

Plug the camera directly into the Pi for the first test. Then run:

```bash
python -m squirrel_shooter.camera_diagnostic
```

The diagnostic lists Linux video nodes, tries the configured camera index, reads
one frame, and reports the mode the camera actually accepted.

If `/dev/video0` is not the UVC camera, inspect the devices without guessing:

```bash
v4l2-ctl --list-devices
ls -l /dev/video*
```

A single USB camera can expose more than one `/dev/video*` node. Use the node that
supports video capture. For example, `/dev/video2` normally corresponds to
`device_index: 2`. Edit `config/default.yaml`, change `device_index`, and rerun
the diagnostic. For more detail about one candidate device:

```bash
v4l2-ctl --device=/dev/video2 --all
v4l2-ctl --device=/dev/video2 --list-formats-ext
```

If the Pi reports a permission error, add your user to the video group, then log
out and back in:

```bash
sudo usermod -aG video "$USER"
```

## Live preview

From a Raspberry Pi desktop session, run:

```bash
python -m squirrel_shooter.camera_preview
```

- Press `q` to quit.
- Press `s` to save the current annotated frame.
- Still images go to `captures/camera-still-<timestamp>.jpg`.

The overlay shows the local timestamp, frame resolution, and measured processing
FPS. Camera settings are requests: the diagnostic or overlay may reveal that the
camera negotiated a different mode.

## Headless recording

SSH sessions and Raspberry Pi OS Lite do not have a preview window. Record a
10-second annotated test video instead:

```bash
python -m squirrel_shooter.camera_capture --seconds 10
```

The result goes to
`captures/camera-test-<timestamp>.avi` relative to the repository root. The AVI
uses the widely supported MJPG codec to keep the first Pi test uncomplicated.
Copy the file to another computer or open it locally with VLC.

To change the capture location, edit `output_directory` in
`config/default.yaml`. Relative paths are resolved from the directory where you
run the command, so run project commands from the repository root.

## Configuration

The initial camera target is intentionally **720p**, even though the camera is
rated for 1080p. This lowers USB, CPU, storage, and thermal pressure while we
prove the complete capture path on the Pi 4.

```yaml
camera:
  device_index: 0
  requested_width: 1280
  requested_height: 720
  requested_fps: 30
  output_directory: captures
```

Each camera command also accepts a different file with
`--config path/to/settings.yaml`.

## Run the tests

Tests do not open a camera or touch hardware:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The source layout is intentionally small:

```text
config/                         Camera settings
docs/ROADMAP.md                 Staged build and safety plan
src/squirrel_shooter/           Python package and camera commands
tests/                          Camera-independent tests
captures/                       Local output, created at runtime and ignored by Git
```

See [docs/ROADMAP.md](docs/ROADMAP.md) for the staged path from camera validation
through conservative water testing. Servos and the valve remain later-phase work;
they must not be connected to runnable code until their power and safety setup is
ready.
