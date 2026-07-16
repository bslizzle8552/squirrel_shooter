# Squirrel Squirter roadmap

The intended sequence is **see -> detect -> aim -> dry-fire -> spray**. Each phase
must be proven before the next phase can control physical hardware.

## Completed foundation: camera and private dashboard

- Validate the Pi USB camera at a conservative 720p target.
- Own the camera in one shared service.
- Provide local diagnostics, recordings, an MJPEG feed, and a private read-only
  Tailnet dashboard.

## Current phase: unattended vision-only watcher and review

- Detect motion locally with MOG2, cleanup, polygon inclusion, time-based recovery,
  persistence, multi-blob grouping, and multi-candidate tracking.
- Reject scene-wide motion from pixel coverage and visual evidence without treating
  low FPS as infrared evidence.
- Save annotated pre/post-roll clips, snapshots, complete event JSON, append-only
  logs, crash-recovery markers, bounded storage, and local review reports.
- Use only provisional size/movement categories; species recognition is explicitly
  a human-review step.
- Tune and validate against real outdoor footage before advancing.

## Future phase: pan/tilt dry-fire

- Add the PCA9685 only after a separate servo power supply is ready.
- Establish conservative mechanical limits and a repeatable neutral position.
- Map image-space targets to calibrated positions with the valve disconnected.

## Future phase: conservative water testing

- Add the MOSFET and 12 V solenoid only after electrical review.
- Default the valve closed at startup, shutdown, exceptions, and signal loss.
- Use short, manually supervised pulses with hard duration and cooldown limits.

Species recognition, calibration, aiming, GPIO, I2C, PCA9685, servos, MOSFET,
solenoid, and water control remain outside the current vision phase.
