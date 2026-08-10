# Raspberry Pi performance verification

The production service still captures one shared 1280x720 camera at the requested 30 FPS. Expensive consumers run independently:

- motion detection and annotated automatic-event clips: `motion.target_fps` (10 FPS)
- manual-fire rolling buffer and replay: `manual_control.recording.target_fps` (12 FPS)
- live MJPEG: `dashboard.stream_fps` (8 FPS), only while viewers are connected
- classifier: once after a qualified event completes, never on every camera frame
- OpenCV native image workers: `runtime.opencv_threads` (1)

The dashboard encoder starts lazily for the first viewer, encodes one shared frame, and blocks when the viewer count returns to zero. Manual-fire pre-roll remains continuously available, but only its sampled frames are JPEG-encoded. The 2-second pre-roll, 5-second post-roll, crop/zoom, and full-frame evidence clip remain enabled.

## Named runtime threads

The main long-running threads are `camera-capture`, `motion-detect`, `classifier`, `dashboard-http`, and the on-demand `mjpeg-encoder`. An accepted FIRE uses `manual-recorder` while its evidence is collected and written. Flask request workers are short-lived. Manual control, cooldown, calibration, servo commands, and PARK do not have polling threads.

On Linux, the application writes these names to each task's `comm` field so `top -H` and `ps -T` can identify them. Linux truncates task names after 15 characters.

## Deploy and establish the idle baseline

From the Pi checkout:

```bash
cd "$PI_REPOSITORY_PATH"
./pull-and-start.sh
```

Resolve the service's Python process and inspect the system immediately after startup:

```bash
PYTHON_PID="$(systemctl --user show squirrel-squirter.service --property MainPID --value)"
printf 'Python PID: %s\n' "$PYTHON_PID"
ps -eo pid,comm,%cpu,%mem --sort=-%cpu | head -15
vcgencmd measure_temp
vcgencmd get_throttled
```

For the first 15 minutes, leave every dashboard/live-video page closed and do not move or fire. Check at 5 minutes and again at 15 minutes:

```bash
ps -p "$PYTHON_PID" -o pid,comm,etime,%cpu,%mem
ps -T -p "$PYTHON_PID" -o pid,tid,comm,%cpu --sort=-%cpu
top -H -p "$PYTHON_PID"
vcgencmd measure_temp
vcgencmd get_throttled
curl -s http://127.0.0.1:5000/api/health | python -m json.tool
```

In `top`, press `H` if threads are not already displayed and `c` to toggle the command/name display. Exit with `q`.

For the no-viewer baseline, `/api/health` should report `dashboard_viewers: 0`, `dashboard_stream_fps: 0`, and an unchanged `dashboard_frames_encoded` count between samples. Camera FPS should remain near the camera's negotiated rate; processing and pre-roll rates should settle near their configured targets.

## Compare viewer and recording load

Open exactly one live dashboard page, wait two minutes, and repeat the commands above. Confirm `dashboard_viewers: 1`, with dashboard stream FPS at or below 8 FPS. Close the page, wait 15 seconds, and confirm the viewer count and stream FPS return to zero and the encoded-frame counter stops increasing.

Finally, perform one supervised accepted FIRE using the established physical safety procedure. During the approximately 7-second evidence window, inspect `top -H` and `/api/health`; `manual_recording.active` should be true while `manual-recorder` is working. Afterward, confirm the zoom replay, full-frame replay, snapshot, and `event.json` were written. A rejected or cooldown-blocked FIRE must not start recording.

Record CPU and temperature for four states: idle with no viewer, one viewer, ordinary detection with no recording, and an active manual recording. Development tests cannot establish Pi CPU percentages or validate physical servo, GPIO, MOSFET, solenoid, water, cooling, or power behavior.
