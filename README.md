# Squirrel Squirter

Squirrel Squirter is a DIY Raspberry Pi garden sprinkler deterrent. The long-term
sequence is:

**see → detect → aim → dry-fire → spray**

The current version deliberately stops at **see**. It brings up a UVC USB camera,
offers local camera diagnostics and test recording, and adds a private web
dashboard for the live feed and saved JPEG captures. Detection, aiming, GPIO,
servos, and water control are not active.

## What works now

- YAML camera configuration with a conservative 1280×720 target at 30 FPS
- One shared background camera service for the dashboard and future detection work
- Private Flask dashboard with an MJPEG live stream and camera/Pi status
- Recent-capture gallery plus a paginated captures page
- Live OpenCV preview and headless MJPG/AVI test recording
- A diagnostic that lists `/dev/video*` devices and reads one frame
- Safe interface placeholders for detection, motion, modes, and valve control
- Camera-independent tests for configuration, filenames, dashboard routes, gallery
  safety, and single-camera ownership

There is no squirrel detection, tracking, servo movement, GPIO, I2C, PCA9685,
MOSFET, or valve output in this version. The placeholder valve controller defaults
to closed and refuses to open. The dashboard has no capture or hardware-control
buttons.

## Raspberry Pi setup

These commands assume 64-bit Raspberry Pi OS with Python 3.11 or newer. On the Pi,
install the system tools used by OpenCV and the camera checks:

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

Every new terminal session needs the repository and environment activated before
running a project command:

```bash
cd squirrel_shooter
source .venv/bin/activate
```

## Run the private dashboard through Tailscale

The dashboard listens on all Pi interfaces at port 5000, but it is intended to be
opened only from another device on the same Tailnet. Do not configure router port
forwarding for it.

### Pull and launch on the Pi

The exact Pi username and existing checkout location are not stored in this
repository. Replace the uppercase placeholders below with their real values. If
you used the clone commands above, `PI_REPOSITORY_PATH` is typically
`~/squirrel_shooter`.

From another device connected to the Tailnet:

```bash
ssh PI_USER@PI_TAILSCALE_HOSTNAME
cd PI_REPOSITORY_PATH
git pull --ff-only
source .venv/bin/activate
python -m pip install -e .
python -m squirrel_shooter.web_dashboard
```

The exact dashboard start command is:

```bash
python -m squirrel_shooter.web_dashboard
```

It runs Flask without debug mode on `0.0.0.0:5000`. Leave that terminal open while
using the dashboard. Press `Ctrl+C` to stop it; the shared camera is then released.

In a second SSH session on the Pi, confirm the process is listening on every
network interface at the required private dashboard port:

```bash
ss -ltnp | grep ':5000'
```

The listener should show `0.0.0.0:5000`. This does not make the dashboard public
by itself. Keep router port forwarding, public DNS, and internet-facing reverse
proxies disabled.

### Find the Pi's Tailscale name or IP

On the Pi, either before starting the dashboard or in a second SSH session:

```bash
tailscale status
tailscale ip -4
```

`tailscale status` shows Tailnet machine names and addresses. `tailscale ip -4`
prints the Pi's Tailscale IPv4 address. You can also find the same machine name and
address in the Tailscale admin console or client application.

### Open the dashboard

On a desktop or phone connected to the same Tailnet, open one of:

```text
http://PI_TAILSCALE_HOSTNAME:5000
http://PI_TAILSCALE_IP:5000
```

If the hostname does not resolve, use the IP printed by `tailscale ip -4`. A camera
failure does not stop the server: the page will show **Camera offline** and remain
available for status and saved captures. The offline detail explains whether the
configured device failed or camera capture was intentionally disabled because the
command was launched somewhere other than Raspberry Pi hardware.

From another device on the same Tailnet, these read-only checks confirm the page
and status endpoint are reachable:

```bash
curl --fail http://PI_TAILSCALE_HOSTNAME:5000/
curl --fail http://PI_TAILSCALE_HOSTNAME:5000/api/status
```

Open the page in a browser to verify that the live image updates continuously.
The stream is MJPEG, so a successful `/video-feed` request remains open while
frames arrive. If the page and status work but the stream does not, check the
camera device and make sure no other process owns it.

## Capture gallery directory

The dashboard reads the `camera.output_directory` value from `config/default.yaml`.
The default is:

```yaml
camera:
  output_directory: captures
```

Relative paths are resolved from the directory where the command is launched, so
start the dashboard from the repository root. With the default configuration the
gallery reads `PI_REPOSITORY_PATH/captures`. It lists `.jpg` and `.jpeg` files only,
newest first. New files appear after a page refresh; AVI recordings and other file
types are ignored. To compare the Pi directory with the gallery without creating
test images, run:

```bash
find captures -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) -printf '%T@ %p\n' | sort -nr | head -12
```

Refresh the landing page and confirm those newest files appear in the same order.
If `camera.output_directory` was changed, use that configured directory instead of
`captures`.

## Connect and identify the USB camera

Plug the camera directly into the Pi for the first test, activate the virtual
environment, and run:

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
supports video capture. For example, `/dev/video2` corresponds to
`device_index: 2`. Edit `config/default.yaml`, change `device_index`, and rerun the
diagnostic. For more detail about a candidate device:

```bash
v4l2-ctl --device=/dev/video2 --all
v4l2-ctl --device=/dev/video2 --list-formats-ext
```

If the Pi reports a permission error, add the current user to the video group,
then log out and back in:

```bash
sudo usermod -aG video "$USER"
```

Stop any other process using the selected camera before starting the dashboard.
Only one process can reliably own a UVC camera at a time.

## Live preview

From a Raspberry Pi desktop session, run:

```bash
python -m squirrel_shooter.camera_preview
```

- Press `q` to quit.
- Press `s` to save the current annotated frame.
- Still images go to `captures/camera-still-<timestamp>.jpg`.

The overlay shows the local timestamp, frame resolution, and measured FPS. Camera
settings are requests: the diagnostic or overlay may reveal that the camera
negotiated a different mode.

## Headless recording

SSH sessions and Raspberry Pi OS Lite do not have a preview window. Record a
10-second annotated test video instead:

```bash
python -m squirrel_shooter.camera_capture --seconds 10
```

The result goes to `captures/camera-test-<timestamp>.avi` relative to the
repository root. The AVI uses the widely supported MJPG codec.

## Configuration

The initial camera target is intentionally **720p**, even though a camera may
support 1080p. This lowers USB, CPU, storage, and thermal pressure while proving
the complete capture path on a Raspberry Pi 4.

```yaml
camera:
  device_index: 0
  requested_width: 1280
  requested_height: 720
  requested_fps: 30
  output_directory: captures
```

Every camera command, including the dashboard, accepts another file with
`--config path/to/settings.yaml`.

## Run the tests

The tests do not open a real camera or touch hardware:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The web tests inject offline and generated-frame camera services. They do not
enumerate or open a development computer webcam.

## Development machine Git workflow

GitHub is the source of truth. Make and test changes in the development checkout,
then review and push them before updating the Pi. Do not maintain a separate copy
of source changes on the Pi.

```bash
git status --short --branch
git diff --check
git diff --stat
python -m pip install -e ".[test]"
python -m pytest
git add README.md pyproject.toml src/squirrel_shooter tests
git commit -m "Add private Raspberry Pi camera dashboard"
git push origin "$(git branch --show-current)"
```

After connecting to the Pi, confirm it pulled the expected GitHub revision before
launching the dashboard:

```bash
ssh PI_USER@PI_TAILSCALE_HOSTNAME
cd PI_REPOSITORY_PATH
git pull --ff-only
git status --short --branch
git log -1 --oneline
source .venv/bin/activate
python -m pip install -e .
python -m squirrel_shooter.web_dashboard
```

The source layout remains intentionally small:

```text
config/                         Camera settings
docs/ROADMAP.md                 Staged build and safety plan
src/squirrel_shooter/           Python package and camera commands
src/squirrel_shooter/templates/ Flask page templates
src/squirrel_shooter/static/    Dashboard styles
tests/                          Camera-independent tests
captures/                       Local output, created at runtime and ignored by Git
```

See [docs/ROADMAP.md](docs/ROADMAP.md) for the staged path from camera validation
through conservative water testing. Servos and the valve remain later-phase work;
they must not be connected to runnable code until their power and safety setup is
ready.
