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

Supervised hardware commissioning is underway. The owner has physically verified the manual web controls, CENTER and manual servo movement, dry-fire operation, and a wet-fire shot at 0.25 seconds. Power wiring, the BCM GPIO24 MOSFET signal, the normally closed solenoid, and the water supply are functioning.

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
- Provide the large 3-degree/5-degree step buttons, D-pad, CENTER, FIRE button, and cooldown status first.
- Show the active calibration point as small informational text.
- Leave point selection and saving to the desktop console.

Both browsers request the current backend state once per second. This keeps pan/tilt, active point, valve state, and cooldown timing synchronized without excessive Raspberry Pi traffic. A refresh during cooldown resumes from the backend's remaining time.

The intended field sequence is: select a point and click its block on desktop, aim and test-fire from the phone as many times as needed, adjust while the 10-second cooldown runs, then save from desktop. Firing never automatically saves or advances a point.

## Calibration interface

The manual-control page uses the existing backend calibration store for nine physical garden blocks:

```text
1  2  3
4  5  6
7  8  9
```

The desktop live image is clickable. The browser sends its click position and rendered image dimensions; the server accounts for responsive scaling, aspect ratio, and `object-fit: contain` letterboxing, then maps the click through the camera runtime's negotiated width and height to a native frame pixel. The backend stores that X/Y on the currently active point. A marker and coordinate readout are rendered from backend state and return after refresh.

The grid distinguishes `Not started`, `Pixel set`, `Aim saved` for legacy partial records, and `Calibrated`. Pixel-only points do not turn green and do not increase `Calibration: X / 9`. A point is green and complete only when pixel X/Y and saved pan/tilt all exist. At 9/9 the page says physical record collection is complete while explicitly stating that interpolation is not ready.

The active point is server-side state, not browser-local state. Selecting a point on one browser changes the point reported to every browser and survives page refreshes for the life of the server process.

Selecting a point shows its stored point number, pixel X/Y, pan, and tilt. Clicking the image creates or updates the pixel half of that same point without saving aim or advancing the active point. Re-clicking replaces its pixel coordinates and never creates a duplicate.

The desktop calibration card has a prominent full-width `SAVE AIM` button. Once aim exists for that point, its label becomes `UPDATE AIM`. The control uses the backend's active point and current backend-commanded pan/tilt values; a stale point, pixel, or angle value from an older browser page is not trusted. It preserves the active point's latest backend-stored pixel coordinates and adds or updates pan/tilt. Updating writes the same record without creating a duplicate. Success feedback names the point and shows saved Pan/Tilt while the pixel coordinates remain visible.

Calibration cannot be saved until a pixel has been selected and CENTER or another real servo command has occurred. The startup 85°/85° display remains an uncommanded reference.

## Firing safety

The calibration FIRE button retains the backend-enforced 10-second cooldown. Refreshes, double taps, and repeated POSTs cannot bypass it. Servo movement remains allowed during cooldown.

Servo movement and valve firing remain mutually exclusive. All future point-and-click work must use the existing shared pipeline:

```text
target -> move -> settle -> fire -> cooldown
```

Do not implement interpolation or a parallel servo, valve, calibration, or firing path.

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

Interpolation, homography, polynomial fitting, pixel-to-pan/tilt mapping, point-and-click firing, and autonomous firing remain intentionally unimplemented.

## Repository boundary

`Misc/` is owner-only storage and is off-limits to normal coding work. Do not inspect, modify, stage, or commit anything beneath it. Preserve all existing modified and untracked files there.
