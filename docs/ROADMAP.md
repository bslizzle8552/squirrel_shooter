# Squirrel Squirter roadmap

The intended sequence is **see -> detect -> aim -> dry-fire -> spray**. Each phase
must be proven before the next phase can control physical hardware.

## Completed foundation: camera and private dashboard

- Validate the Pi USB camera at a conservative 720p target.
- Own the camera in one shared service.
- Provide local diagnostics, recordings, an MJPEG feed, and a private read-only
  Tailnet dashboard.

## Current phase: motion vision and diagnostics

- Detect motion locally with MOG2, cleanup, configurable size limits, persistence,
  cooldown, lighting resets, and a rectangular garden ROI.
- Make detector decisions visible in the stream, structured logs, health APIs, and
  bounded recent-event data.
- Save only annotated accepted-event snapshots and bound local storage.
- Tune and validate against real outdoor footage before advancing.

## Future phase: pan/tilt dry-fire

- Add the PCA9685 only after a separate servo power supply is ready.
- Establish conservative mechanical limits and a repeatable neutral position.
- Map image-space targets to calibrated positions with the valve disconnected.

## Future phase: conservative water testing

- Add the MOSFET and 12 V solenoid only after electrical review.
- Default the valve closed at startup, shutdown, exceptions, and signal loss.
- Use short, manually supervised pulses with hard duration and cooldown limits.

Squirrel recognition, tracking, calibration, aiming, GPIO, servos, and water remain
outside the current vision phase.
