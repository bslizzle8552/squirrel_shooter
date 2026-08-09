# Manual-control handoff

Branch: `manual-control`. Do not merge into `main` without explicit authorization.

## Hardware configuration

- MOSFET signal: BCM GPIO24.
- Raspberry Pi physical pin 18: BCM GPIO24.
- Raspberry Pi physical pin 20: GND.
- Software configuration must use `gpio_pin: 24`, not physical pin number 18.
- `active_high: true` remains required so the safe closed command is LOW.
- `valve.enabled: true` remains checked in for supervised manual operation.
- The normal physically verified 0.25-second calibration/test pulse is restored.
- The backend-enforced cooldown remains 10 seconds.
- CENTER remains pan 85 degrees / tilt 85 degrees.
- PARK is configured separately as pan 85 degrees / tilt 88 degrees. This is a 3-degree offset in the existing backend `up` direction (`up` increases commanded tilt). The physical upward result must be confirmed under supervision before relying on it with water; tune only `pan_tilt.park_tilt` if the installed linkage needs a different absolute command.

Supervised hardware commissioning is underway. The owner has physically verified the manual web controls, CENTER and manual servo movement, dry-fire operation, and a wet-fire shot at 0.25 seconds. Power wiring, the BCM GPIO24 MOSFET signal, the normally closed solenoid, and the water supply are functioning. All nine physical garden calibration records now contain real camera pixels and water-impact-tested commanded pan/tilt values.

The temporary 3.0-second demonstration setting is no longer active.

## Two-device calibration workflow

The manual-control page now supports a desktop/laptop calibration console and a phone field remote at the same time. Both browsers use the same `ManualControlService` instance and backend-authoritative state.

Desktop/laptop responsibilities:

- View the shared camera stream.
- Select the active point in the 3 by 3 calibration grid.
- Click the center of that point's physical block in the live image.
- Confirm the backend-stored native camera pixel X/Y and marker.
- Monitor saved/unsaved state, progress, and stored values.
- Save or update the active point only after the physical water hit is satisfactory.

Phone responsibilities:

- Show the current backend-commanded pan and tilt angles.
- Provide large 1-degree/3-degree/5-degree step buttons, D-pad, CENTER, FIRE button, and cooldown status first. Use 1 degree for fine adjustment when 3 degrees is too large.
- Show the active calibration point as small informational text.
- Leave point selection and saving to the desktop console.

Both browsers request the current backend state once per second. This keeps pan/tilt, active point, valve state, and cooldown timing synchronized without excessive Raspberry Pi traffic. A refresh during cooldown resumes from the backend's remaining time.

The calibration field sequence remains available under `EDIT CALIBRATION`: select a point and click its block on desktop, aim and test-fire from the phone as many times as needed, adjust while the 10-second cooldown runs, then save from desktop. Firing never automatically saves or advances a point.

## Calibration interface

The manual-control page uses the existing backend calibration store for nine physical garden blocks:

```text
1  2  3
4  5  6
7  8  9
```

In `EDIT CALIBRATION` mode, the desktop live image retains its calibration behavior. The browser sends its click position and rendered image dimensions; the server accounts for responsive scaling, aspect ratio, and `object-fit: contain` letterboxing, then maps the click through the camera runtime's negotiated width and height to a native frame pixel. The backend stores that X/Y on the currently active point. A marker and coordinate readout are rendered from backend state and return after refresh.

The grid distinguishes `Not started`, `Pixel set`, `Aim saved` for legacy partial records, and `Calibrated`. Pixel-only points do not turn green and do not increase `Calibration: X / 9`. A point is green and complete only when pixel X/Y and saved pan/tilt all exist. Targeting is enabled only when the store contains exactly points 1 through 9, all complete, with nine distinct pixels and usable grid geometry.

The active point is server-side state, not browser-local state. Selecting a point on one browser changes the point reported to every browser and survives page refreshes for the life of the server process.

Selecting a point shows its stored point number, pixel X/Y, pan, and tilt. Clicking the image creates or updates the pixel half of that same point without saving aim or advancing the active point. Re-clicking replaces its pixel coordinates and never creates a duplicate.

The desktop calibration card has a prominent full-width `SAVE AIM` button. Once aim exists for that point, its label becomes `UPDATE AIM`. The control uses the backend's active point and current backend-commanded pan/tilt values; a stale point, pixel, or angle value from an older browser page is not trusted. It preserves the active point's latest backend-stored pixel coordinates and adds or updates pan/tilt. Updating writes the same record without creating a duplicate. Success feedback names the point and shows saved Pan/Tilt while the pixel coordinates remain visible.

Calibration cannot be saved until a pixel has been selected and CENTER or another real servo command has occurred. The startup 85°/85° display remains an uncommanded reference.

## Desktop click-to-aim

The camera card now has two explicit desktop modes:

- `AIM TARGET` is available only with a valid complete nine-point grid. A click is mapped to a native camera pixel, interpolated, clamped to existing servo limits, and passed through `ManualControlService` for serialized movement and settling. It does not write the calibration store, open the valve, or start cooldown.
- `EDIT CALIBRATION` retains the prior behavior: a click updates only the selected calibration point's pixel. This explicit mode is the only camera-click path that can modify calibration data.

Interpolation is local and piecewise linear. The 3 by 3 grid forms four quadrilateral cells: `1-2-5-4`, `2-3-6-5`, `4-5-8-7`, and `5-6-9-8`. Each cell is split along its top-left to bottom-right diagonal, producing eight triangles. Barycentric weights inside the containing triangle interpolate both pan and tilt. This reproduces every saved calibration point exactly and handles perspective-distorted camera geometry without fitting a complex model.

Only clicks inside the union of those eight triangles are accepted. A click outside that calibrated region returns `OUTSIDE CALIBRATED AREA`; there is no extrapolation or fallback guess. Final interpolated commands are still clamped to pan 30-150 degrees and tilt 70-150 degrees.

The backend stores the last requested target pixel, calculated pan/tilt, cell, and targeting status. The desktop shows the target marker, clicked X/Y, calculated pan/tilt, and `MOVING`, `SETTLING`, `AIM READY`, `FIRING`, `PARKING`, or `PARKED` status as applicable. Both desktop and phone continue polling the same backend commanded position.

## Firing safety

The existing FIRE button remains the only firing action. A target click never calls the valve or `move_and_fire()`. After `AIM READY`, the operator manually presses FIRE; the pulse starts from the current commanded position and does not perform a pre-fire movement.

The FIRE button retains the physically verified 0.25-second pulse and backend-enforced 10-second cooldown. Refreshes, double taps, and repeated POSTs cannot bypass it. FIRE is rejected rather than queued while movement or settling holds the coordinator. Servo movement remains allowed during cooldown.

Servo movement and valve firing remain mutually exclusive. The implemented manual sequence is:

```text
click -> interpolate -> move -> settle -> wait -> manual FIRE
      -> valve ON for 0.25s -> valve OFF -> cooldown starts -> PARK move -> PARKED
```

PARK uses the same service lock and movement helper as every other servo command. It begins only after `pulse_valve()` has returned the valve to LOW/OFF. Cooldown is timestamped when the pulse ends, so the PARK movement safely occurs during cooldown. PARK updates only commanded pan/tilt state; it never writes a calibration point or target pixel. Clean shutdown retains the existing valve-cleanup-first and configured servo PARK behavior. Startup still initializes at an uncommanded 85/85 reference and does not move the servos.

## Nine-block calibration procedure

The nine blocks replace the earlier painted-X marker concept; the calibration geometry and point numbering are unchanged.

1. On desktop, select the active calibration point.
2. Click the center of the corresponding physical block in the live camera image.
3. Confirm the backend reports the native camera pixel X/Y and restores its marker after refresh.
4. Go outside with the phone and manually aim using the D-pad.
5. FIRE the normal 0.25-second pulse and observe the water landing point.
6. Adjust aim while the backend-enforced 10-second cooldown runs; fire again only when ready.
7. Once the water hits the block, press `SAVE AIM` on desktop.
8. `SAVE AIM` preserves the pixel and records the current backend-commanded pan/tilt; the point becomes green and fully calibrated. Repeating the process for that point uses `UPDATE AIM`.
9. Select the next point and repeat through all nine blocks.

## First physical verification

No new click-to-aim or 85/88 PARK behavior is claimed as physically verified by automated tests.

1. First click-to-aim test: disconnect or shut off the water supply, open the desktop page, confirm `Calibration: 9 / 9`, select `AIM TARGET`, and click directly on one known calibration block. Confirm the target marker and X/Y, verify the reported aim matches that point's saved pan/tilt, observe movement followed by `AIM READY`, and confirm the valve/MOSFET never energizes. Then try one click clearly outside the calibrated block region and confirm it is rejected without movement.
2. First manual fire after interpolated aim: with the area supervised and water restored, click a known point, wait for `AIM READY` and complete servo stillness, then press the existing FIRE button once. Confirm the shot occurs from that aim, lasts 0.25 seconds, and a repeated FIRE remains blocked by the 10-second cooldown.
3. First PARK verification: perform a dry-fire first and observe that the valve is OFF before the servo moves to pan 85 / tilt 88. Confirm the nozzle ends approximately 2-3 degrees physically upward from CENTER. If 88 moves the installed nozzle the wrong way or too far, stop before wet testing and tune the configurable `pan_tilt.park_tilt`; CENTER stays 85/85 and the servo limits must not change.

Automatic firing on click, autonomous engagement, tracking, prediction, bursts, and guard-mode firing remain intentionally unimplemented.

## Repository boundary

`Misc/` is owner-only storage and is off-limits to normal coding work. Do not inspect, modify, stage, or commit anything beneath it. Preserve all existing modified and untracked files there.
