# Manual-control handoff

Branch: `manual-control`. Do not merge into `main` without explicit authorization.

## Hardware configuration

- MOSFET signal: BCM GPIO24.
- Raspberry Pi physical pin 18: BCM GPIO24.
- Raspberry Pi physical pin 20: GND.
- Software configuration must use `gpio_pin: 24`, not physical pin number 18.
- `active_high: true` remains required so the safe closed command is LOW.
- `valve.enabled: false` remains the checked-in default until supervised hardware testing.

No GPIO, MOSFET, solenoid, water flow, or servo direction was physically tested during this software pass because the Pi was powered off.

## Two-device calibration workflow

The manual-control page now supports a desktop/laptop calibration console and a phone field remote at the same time. Both browsers use the same `ManualControlService` instance and backend-authoritative state.

Desktop/laptop responsibilities:

- View the shared camera stream.
- Select the active point in the 3 by 3 calibration grid.
- Monitor saved/unsaved state, progress, and stored values.
- Save or update the active point only after the physical water hit is satisfactory.

Phone responsibilities:

- Show the current backend-commanded pan and tilt angles.
- Provide the large 3-degree/5-degree step buttons, D-pad, CENTER, FIRE button, and cooldown status first.
- Show the active calibration point as small informational text.
- Leave point selection and saving to the desktop console.

Both browsers request the current backend state once per second. This keeps pan/tilt, active point, valve state, and cooldown timing synchronized without excessive Raspberry Pi traffic. A refresh during cooldown resumes from the backend's remaining time.

The intended field sequence is: select a point on desktop, aim and test-fire from the phone as many times as needed, adjust while the 10-second cooldown runs, then save from desktop. Firing never automatically saves or advances a point.

## Calibration interface

The manual-control page uses the existing backend calibration store for exactly nine physical points:

```text
1  2  3
4  5  6
7  8  9
```

The page shows unsaved, active, and saved states; saved points are green. Progress is displayed as `Calibration: X / 9`. At 9/9 the page says physical record collection is complete while explicitly stating that interpolation is not ready.

The active point is server-side state, not browser-local state. Selecting a point on one browser changes the point reported to every browser and survives page refreshes for the life of the server process.

Selecting a point shows its stored point number, pixel X/Y, pan, and tilt. Null pixel values are expected until camera-image selection is implemented. Saving an existing point overwrites that point and never creates a duplicate. The page gives a prominent save/update confirmation with the stored angles.

SAVE uses the backend's active point and current backend-commanded pan/tilt values. A stale point or angle value from an older browser page is not trusted. Pixel values remain null for now.

Calibration still cannot be saved until CENTER or another real servo command has occurred. The startup 85°/85° display remains an uncommanded reference.

## Firing safety

The calibration FIRE button retains the backend-enforced 10-second cooldown. Refreshes, double taps, and repeated POSTs cannot bypass it. Servo movement remains allowed during cooldown.

Servo movement and valve firing remain mutually exclusive. All future point-and-click work must use the existing shared pipeline:

```text
target -> move -> settle -> fire -> cooldown
```

Do not implement interpolation or a parallel servo, valve, calibration, or firing path.

## Remaining supervised hardware verification

1. Keep the 12 V solenoid supply disconnected and confirm startup causes no servo movement.
2. Press CENTER and verify physical 85-degree/85-degree alignment.
3. Verify one small UP, DOWN, LEFT, and RIGHT command from the phone, stopping immediately if a direction is wrong or hardware binds.
4. Verify the existing pan and tilt limits conservatively.
5. Confirm physical pin 18, BCM GPIO24, is LOW at startup while `valve.enabled` is false.
6. Enable the valve only for a supervised test with an immediate power shutoff available.
7. With 12 V valve power still disconnected, verify one 0.25-second GPIO signal and the full 10-second backend cooldown on both phone and desktop.
8. Confirm refreshes and repeated FIRE requests cannot bypass cooldown while servo aiming still works.
9. Connect the safely directed water system and perform one supervised short pulse.
10. Tune pulse duration and settling delay conservatively if needed.
11. For each of the nine targets, select on desktop, aim/test from phone, and save on desktop only after the water hit is correct.

Physical testing remains required before treating servo direction, endpoint clearance, GPIO behavior, MOSFET switching, solenoid operation, or water placement as verified.

## Repository boundary

`Misc/` is owner-only storage and is off-limits to normal coding work. Do not inspect, modify, stage, or commit anything beneath it. Preserve all existing modified and untracked files there.
