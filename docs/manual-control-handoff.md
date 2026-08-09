# Manual-control handoff

Branch: `manual-control`. Do not merge into `main` without explicit authorization.

## Hardware configuration

- MOSFET signal: BCM GPIO24.
- Raspberry Pi physical pin 18: BCM GPIO24.
- Raspberry Pi physical pin 20: GND.
- Software configuration must use `gpio_pin: 24`, not physical pin number 18.
- `active_high: true` remains required so the safe closed command is LOW.
- `valve.enabled: true` remains checked in for supervised manual operation.
- The checked-in pulse is temporarily 3.0 seconds for one supervised demonstration shot, followed by the unchanged backend-enforced 10-second cooldown.
- Return the pulse to the physically verified 0.25 seconds immediately after the demonstration.

Supervised hardware commissioning is underway. The owner has physically verified the manual web controls, CENTER and manual servo movement, dry-fire operation, and a wet-fire shot at 0.25 seconds. Power wiring, the BCM GPIO24 MOSFET signal, the normally closed solenoid, and the water supply are functioning.

The temporary 3.0-second wet shot has not yet been physically tested. Do not treat that demonstration duration as verified until the owner records the result.

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

## Temporary supervised demonstration

The next physical test is one single supervised demonstration FIRE command at the temporary 3.0-second duration:

1. Keep an immediate 12 V power shutoff available, aim the nozzle safely, and clear the spray area.
2. Start the updated application and confirm there is no solenoid actuation at startup; the active-high BCM GPIO24 output must remain LOW/OFF until FIRE.
3. Open `/manual-control`, confirm the FIRE control reports `READY`, and press FIRE exactly once.
4. Observe the single 3.0-second demonstration pulse and confirm the solenoid releases afterward.
5. Confirm the page enters the 10-second cooldown and that refreshes or repeated FIRE requests do not bypass it. Servo aiming may remain available during cooldown.
6. Record the result, then return `manual_control.fire_pulse_seconds` to the previously verified 0.25 seconds before normal calibration or test operation.

The temporary 3.0-second setting is not a new operating default and must not remain in place after the demonstration.

## Repository boundary

`Misc/` is owner-only storage and is off-limits to normal coding work. Do not inspect, modify, stage, or commit anything beneath it. Preserve all existing modified and untracked files there.
