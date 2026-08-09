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

## Calibration interface

The manual-control page uses the existing backend calibration store for exactly nine physical points:

```text
1  2  3
4  5  6
7  8  9
```

The page shows unsaved, active, and saved states; saved points are green. Progress is displayed as `Calibration: X / 9`. At 9/9 the page says physical record collection is complete while explicitly stating that interpolation is not ready.

Selecting a point shows its stored point number, pixel X/Y, pan, and tilt. Null pixel values are expected until camera-image selection is implemented. Saving an existing point overwrites that point and never creates a duplicate. The page gives a prominent save/update confirmation with the stored angles.

Calibration still cannot be saved until CENTER or another real servo command has occurred. The startup 85°/85° display remains an uncommanded reference.

## Firing safety

The calibration FIRE button retains the backend-enforced 10-second cooldown. Refreshes, double taps, and repeated POSTs cannot bypass it. Servo movement remains allowed during cooldown.

Servo movement and valve firing remain mutually exclusive. All future point-and-click work must use the existing shared pipeline:

```text
target -> move -> settle -> fire -> cooldown
```

Do not implement interpolation or a parallel servo, valve, calibration, or firing path.

## Repository boundary

`Misc/` is owner-only storage and is off-limits to normal coding work. Do not inspect, modify, stage, or commit anything beneath it. Preserve all existing modified and untracked files there.
