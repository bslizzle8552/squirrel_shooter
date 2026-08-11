# Raspberry Pi performance verification

This procedure measures the production workload without weakening detection,
video, recording, or physical-control safety. Development tests and Windows
profiling do not establish Raspberry Pi CPU, temperature, throttling, camera,
servo, GPIO, valve, or wet-fire behavior.

The production service has one 1280x720 camera owner at a requested 15 FPS.
Expensive consumers run at independent rates:

- motion detection and annotated automatic-event clips: 10 FPS
- sampled raw manual-fire pre-roll and replay: 12 FPS
- one shared live MJPEG representation: at most 8 FPS, only with a viewer
- classifier: once after a qualified event completes, never per camera frame
- OpenCV native workers: one
- aggregate telemetry and session persistence: every 30 seconds

The MJPEG worker is lazy and blocks when no viewers exist. Hidden browser tabs,
the dashboard review mode, and an explicitly paused stream detach from MJPEG.
Manual-fire pre-roll retains sampled raw frames without idle JPEG/video encoding.
Headless motion overlays are rendered only for an automatic event or a viewer.
The 2-second pre-roll, 5-second post-roll, crop/zoom replay, and full-frame
evidence clip remain enabled.

## Named runtime threads

The steady-state application uses `camera-capture`, `motion-detect`,
`classifier`, and `dashboard-http`. `mjpeg-encoder` starts on first demand but
blocks without a viewer. A single `manual-recorder` executor worker is created
on accepted FIRE and performs evidence collection and encoding outside the
physical-control lock. Flask request workers are short-lived. Manual control,
cooldown, calibration, servo commands, and PARK do not run polling threads.

On Linux, the application writes these names to each task's `comm` field so
`top -H` and `ps -T` can identify them. Linux may truncate names to 15
characters.

## 1. Deploy the exact revision

The Pi checkout must already be on `manual-control`. From that existing checkout:

```bash
cd ~/squirrel_shooter
git branch --show-current
./pull-and-start.sh
git rev-parse HEAD
systemctl --user show squirrel-squirter.service --property MainPID,ActiveState,SubState --no-pager
```

`pull-and-start.sh` stops the service, fast-forwards the branch currently checked
out on the Pi, refreshes the editable installation, runs the full software test
suite, and starts the service only after those steps succeed. Stop if the first
command is not `manual-control`; do not switch or merge branches as part of this
measurement.

Resolve and validate the one service process:

```bash
PYTHON_PID="$(systemctl --user show squirrel-squirter.service --property MainPID --value)"
printf 'Python PID: %s\n' "$PYTHON_PID"
test "$PYTHON_PID" -gt 0
pgrep -af 'python.*squirrel_shooter'
ps -p "$PYTHON_PID" -o pid,ppid,comm,etime,%cpu,%mem,rss,vsz
```

There must be one production application process. Do not start
`squirrel_shooter.app`, `motion_watch`, or another development server beside the
user service.

## 2. Keep one telemetry window open

In a second SSH session, follow the structured 30-second samples:

```bash
journalctl --user -u squirrel-squirter.service -f -o cat | grep --line-buffered 'runtime_performance'
```

For recent samples without following:

```bash
journalctl --user -u squirrel-squirter.service --since '-15 minutes' -o cat | grep 'runtime_performance'
```

Each sample includes capture and detection FPS, capture/detector/annotation
average time, estimated camera and motion thread CPU, pre-roll buffer state,
dashboard viewer/encode/egress metrics, classifier queue/inference state, manual
recording state, and CPU temperature when the Pi exposes it.

## 3. Use one repeatable measurement block

Run this block at every checkpoint and save its output with the stage name and
wall-clock time:

```bash
date --iso-8601=seconds
ps -eo pid,comm,%cpu,%mem --sort=-%cpu | head -15
ps -p "$PYTHON_PID" -o pid,comm,etime,%cpu,%mem,rss,vsz
ps -T -p "$PYTHON_PID" -o pid,tid,comm,%cpu --sort=-%cpu
vcgencmd measure_temp
vcgencmd get_throttled
curl --fail --silent http://127.0.0.1:5000/api/health | python -m json.tool
```

Also inspect interactively at least once per operating mode:

```bash
top
top -H -p "$PYTHON_PID"
```

Exit `top` with `q`. Record the raw `get_throttled` hexadecimal value rather
than relying on a verbal interpretation. A fresh `0x0` is the desired baseline;
nonzero current or historical flags require investigation together with
temperature, power, cooling, and workload evidence.

## 4. Execute the staged validation

Do not combine stages. Let each state settle, then run the complete measurement
block.

1. **Startup:** keep every browser closed. Measure immediately after the PID is
   established and verify the camera and detector become alive.
2. **Two-minute headless:** leave all camera pages closed and do not move or
   fire. At two minutes, measure. `dashboard_viewers` and
   `dashboard_stream_fps` should be zero.
3. **Ten-minute headless:** continue unchanged until ten minutes after startup,
   then measure again. Compare RSS, CPU, temperature, throttling, capture FPS,
   detection FPS, and average stage times with the two-minute sample. Long-run
   memory should settle rather than grow linearly.
4. **One dashboard viewer:** open exactly one dashboard in OBSERVE mode, wait
   two minutes, then measure. Expect one viewer and no more than the configured
   8 FPS shared encode rate. Record `dashboard_last_jpeg_bytes` and
   `dashboard_estimated_egress_mbps` alongside `tailscaled` CPU.
5. **Manual-control page:** close the dashboard, open only `/manual-control`,
   leave AIM/FIRE idle for two minutes, then measure. Its visible one-second
   state synchronization remains intentional; it shares the same MJPEG encoder
   and camera.
6. **Click-to-shoot/AIM view:** enter the existing non-firing target-aim mode,
   wait two minutes without moving, then measure. Confirm it did not create a
   second application/camera process or a second encoder worker.
7. **Calibration view:** enter calibration, make no calibration changes, wait
   two minutes, then measure. The UI may draw browser overlays, but the backend
   must still use the same stream and one-second state cadence.
8. **One manual move:** with water disconnected or otherwise made safe under
   the established field procedure, issue one ordinary D-pad/keyboard move.
   Measure during/just after movement. Confirm there is no sustained servo CPU
   after the synchronous movement completes.
9. **One supervised FIRE plus recording:** only after the existing dry and
   safety checks, perform one accepted FIRE. Do not alter the 0.40-second pulse,
   10-second backend cooldown, move-settle-fire-cooldown ordering, calibration,
   servo limits, park behavior, control token, or normally-closed/LOW-safe valve
   behavior. Observe `manual_recording_active`, `manual_recording_queued`, and
   `manual-recorder`; measure during the roughly seven-second evidence window.
   Afterward verify the zoom replay, full-frame replay, snapshot, and `event.json`.
   A rejected or cooldown-blocked FIRE must not start a recording.
10. **Viewer teardown:** close every relevant browser tab (do not merely switch
    to another app), wait 15 seconds, and measure. The viewer count and stream
    FPS should return to zero; the dashboard encoded-frame counter should stop.
11. **Second ten-minute headless:** remain idle for ten more minutes and measure.
    Compare with stages 2 and 3 to detect leaked viewers, workers, memory, or
    post-event work.

For hidden-tab behavior, optionally leave a page open, background the tab for 30
seconds, and confirm its viewer detaches; foregrounding should reconnect and
refresh immediately.

## 5. Interpret the telemetry

Use deltas between stages, not one instantaneous number:

- high `capture_read_average_ms` or `capture_thread_cpu_percent` with no viewer
  points toward USB/MJPG capture/decode or camera negotiation;
- high `detector_average_ms` and `motion_thread_cpu_percent` points toward MOG2,
  global lighting measurements, blur/morphology, mask, and contour work;
- annotation time should be near zero in quiet, zero-viewer headless operation;
- dashboard encode time and estimated egress should rise only with a viewer;
- classifier queue/inference should change only after a completed qualified
  event;
- manual recording should be inactive and unqueued at idle, then return there
  after the evidence worker finishes;
- growing RSS across both ten-minute headless periods, without corresponding
  retained event frames, is a memory-regression signal.

Preserve the measurements, exact Git SHA, Pi model, cooling configuration,
ambient conditions, power supply, camera mode, and number of viewers with the
test record. Only this physical run can decide whether the optimized Pi 4 is
thermally adequate in the field.
