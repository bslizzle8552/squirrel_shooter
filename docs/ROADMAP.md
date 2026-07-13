# Squirrel Shooter roadmap

The intended sequence is **see → detect → aim → dry-fire → spray**. Each phase
must be proven before the next phase is allowed to control physical hardware.

## Phase 1: Pi + USB camera validation

- Set up Raspberry Pi OS, Python, and the project environment.
- Identify the UVC camera and confirm reliable 720p capture at a useful frame rate.
- Verify both desktop preview and headless recording.

## Phase 2: Camera recording and garden footage collection

- Collect daytime, evening, weather, shadow, plant-motion, and squirrel footage.
- Review framing and choose the fixed camera position.
- Keep recordings local and establish sensible retention limits.

## Phase 3: Motion detection inside a defined garden zone

- Define an inclusion zone so roads, neighbors, and irrelevant movement are ignored.
- Begin with lightweight local motion detection rather than cloud or heavy ML services.
- Measure false positives before anything can aim.

## Phase 4: Pan/tilt servo zeroing and dry-fire aiming

- Add the PCA9685 only after a proper separate servo power supply is available.
- Establish conservative mechanical limits and a repeatable neutral position.
- Test aiming with the valve disconnected and water disabled.

## Phase 5: Calibration from camera pixels to pan/tilt position

- Map positions in the camera image to safe pan and tilt targets.
- Clamp every target to the established mechanical and garden-zone limits.
- Verify repeatability with dry-fire tests only.

## Phase 6: Valve control and conservative water testing

- Add the MOSFET and 12 V solenoid only after electrical review.
- Default the valve to closed/off at startup, shutdown, exceptions, and signal loss.
- Use short, manually supervised pulses with hard duration and cooldown limits.

## Phase 7: Event snapshots and a simple private dashboard

- Save useful local event snapshots and basic metadata.
- Add a small private status/history view only after the core behavior is dependable.
- Keep remote control out of scope until authentication and physical safety are designed.
