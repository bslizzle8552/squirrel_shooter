# Manual-control handoff

Branch: `manual-control`. Do not merge into `main` without explicit authorization.

## Hardware configuration

- MOSFET signal: BCM GPIO24.
- Raspberry Pi physical pin 18: BCM GPIO24.
- Raspberry Pi physical pin 20: GND.
- Software configuration must use `gpio_pin: 24`, not physical pin number 18.
- `active_high: true` remains required so the safe closed command is LOW.
- `valve.enabled: true` remains checked in for supervised manual operation.
- The configured manual calibration/test pulse is 0.40 seconds. The earlier 0.25-second pulse was physically verified; 0.40 seconds still requires supervised physical verification.
- The backend-enforced cooldown remains 10 seconds.
- CENTER remains pan 85 degrees / tilt 85 degrees.
- PARK is configured separately as pan 85 degrees / tilt 82 degrees so the nozzle rests at the owner-selected anti-drip angle. CENTER remains 85/85; tune only `pan_tilt.park_tilt` if the installed linkage needs a different rest command.

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
- Provide large 1-degree/3-degree/5-degree step buttons, D-pad, CENTER, PARK, FIRE button, and cooldown status first. Use 1 degree for fine adjustment when 3 degrees is too large. PARK moves to the configured 85/82 anti-drip rest position without firing or starting cooldown.
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

## Field targeting geometry diagnosis

The field report was that a click on the unmoved bottom-right red calibration block was rejected as out of range and that accepted intermediate targets aimed feet left and long. A read-only inspection of the deployed backend returned these nine complete 1280x720 native-frame records; this pass does not alter them:

| Point | Pixel X | Pixel Y | Pan | Tilt |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 446 | 172 | 84 | 79 |
| 2 | 668 | 180 | 74 | 78 |
| 3 | 893 | 192 | 61 | 75 |
| 4 | 448 | 227 | 84 | 93 |
| 5 | 640 | 246 | 70 | 88 |
| 6 | 1003 | 242 | 58 | 88 |
| 7 | 139 | 436 | 101 | 109 |
| 8 | 693 | 434 | 73 | 109 |
| 9 | 1121 | 412 | 48 | 104 |

The records match the intended perspective-distorted 3 by 3 layout. The verified outer perimeter order is `1-2-3-6-9-8-7-4`; it is simple, contains all nine anchors, and is not widened beyond the saved geometry. Backend reproduction showed that exact Point 9 `(1121, 412)` was already accepted and returned approximately pan 48 / tilt 104, while `(1121, 424)` was outside. The latter is only 12 native pixels lower and can still land on the visible lower portion of the same physical block because perimeter anchors are stored at block centers. Without an overlay, that distinction appeared to be rejection of the anchor itself.

The prior mapping also split every four-anchor cell across a fixed diagonal and used only three anchors for any one result. That arbitrary diagonal creates a discontinuous choice of local fit and does not use all four physically measured corner aims, which is a credible software cause of poor intermediate field aim. It has been replaced with inverse bilinear image-space mapping and four-corner bilinear pan/tilt interpolation inside the selected quadrilateral. This preserves every anchor within floating-point tolerance, behaves continuously across each cell, and does not extrapolate beyond the calibrated cell union.

The desktop live view now draws the exact targeting geometry: a yellow outer boundary, four blue cell outlines, and numbered markers for all nine saved anchors. AIM feedback reports native X/Y, `IN RANGE` or `OUT OF RANGE`, selected cell, `inverse bilinear`, calculated pan/tilt, and the rejection reason. The overlay and targeting use the same backend native-pixel records; they do not modify calibration.

The deployed camera reported a 1280x720 native frame. The live desktop image measured approximately 726.42x408.60 CSS pixels with `object-fit: contain`, and its parent had the same bounds. Both calibration and AIM clicks use the same `display_click_to_frame_pixel()` conversion, including scaling and any letterbox offsets. Automated checks cover native size, half size, the observed live size, and a letterboxed 800x600 display; each maps the center of Point 9 back to `(1121, 412)`.

## Desktop click-to-aim

The camera card now has two explicit desktop modes:

- `AIM TARGET` is available only with a valid complete nine-point grid. A click is mapped to a native camera pixel, interpolated, clamped to existing servo limits, and passed through `ManualControlService` for serialized movement and settling. It does not write the calibration store, open the valve, or start cooldown.
- `EDIT CALIBRATION` retains the prior behavior: a click updates only the selected calibration point's pixel. This explicit mode is the only camera-click path that can modify calibration data.

Interpolation is local and four-corner bilinear. The 3 by 3 grid forms four quadrilateral cells identified as `1-2-4-5`, `2-3-5-6`, `4-5-7-8`, and `5-6-8-9`, with their geometric corners consistently ordered top-left, top-right, bottom-right, bottom-left. The backend first tests cell membership in native camera pixels, inverse-maps the selected quadrilateral to local horizontal/vertical coordinates, and uses those coordinates to bilinearly interpolate pan and tilt from all four saved aims.

Only clicks inside the union of those four validated quadrilaterals are accepted. The outer display boundary follows `1-2-3-6-9-8-7-4`, and validation rejects self-intersecting, degenerate, or non-convex geometry and any anchor outside the boundary/cell union. A click outside returns `OUTSIDE CALIBRATED AREA`; there is no extrapolation or fallback guess. Final interpolated commands are still clamped to pan 30-150 degrees and tilt 70-150 degrees.

The backend stores the last requested target pixel, calculated pan/tilt, cell, in-range result, and targeting status. The desktop shows the diagnostic geometry, target marker, clicked X/Y, selected cell/method, calculated pan/tilt, reason for rejection, and `MOVING`, `SETTLING`, `AIM READY`, `FIRING`, `PARKING`, or `PARKED` status as applicable. Both desktop and phone continue polling the same backend commanded position.

## Firing safety

The existing FIRE button remains the only firing action. A target click never calls the valve or `move_and_fire()`. After `AIM READY`, the operator manually presses FIRE; the pulse starts from the current commanded position and does not perform a pre-fire movement.

The FIRE button uses the configured 0.40-second pulse and retains the backend-enforced 10-second cooldown. Refreshes, double taps, and repeated POSTs cannot bypass it. FIRE is rejected rather than queued while movement or settling holds the coordinator. Servo movement remains allowed during cooldown.

The separate PARK button is authenticated and uses the same serialized movement helper as every other servo command. It moves directly to configured pan 85 / tilt 82, never opens the valve, and never starts or clears cooldown. It is rejected if another control action already owns the coordinator.

Servo movement and valve firing remain mutually exclusive. The implemented manual sequence is:

```text
click -> interpolate -> move -> settle -> wait -> manual FIRE
      -> valve ON for 0.40s -> valve OFF -> cooldown starts -> PARK move -> PARKED
```

PARK uses the same service lock and movement helper as every other servo command. It begins only after `pulse_valve()` has returned the valve to LOW/OFF. Cooldown is timestamped when the pulse ends, so the PARK movement safely occurs during cooldown. PARK updates only commanded pan/tilt state; it never writes a calibration point or target pixel. Clean shutdown retains the existing valve-cleanup-first and configured servo PARK behavior. Startup still initializes at an uncommanded 85/85 reference and does not move the servos.

## Manual-fire recordings

Every FIRE that completes its normal valve pulse queues one evidence event. Cooldown, disabled-valve, invalid-token, busy-state, and pulse-failure rejections do not queue a successful recording. Recording is enabled under `manual_control.recording` with approximately 2 seconds before the accepted FIRE and 5 seconds after it, for a 7-second real elapsed-time target. If some or all pre-roll is unavailable, post-roll is extended so the target remains approximately 7 seconds. The timing window uses monotonic time, never a fixed frame count.

The existing `CameraService` remains the only camera owner. It samples a rolling raw-frame pre-roll at `manual_control.recording.target_fps` (12 FPS by default), independently of the requested 15 FPS camera and the on-demand dashboard encoder; it does not open a second camera or continuously JPEG-encode pre-roll. After `pulse_valve()` has returned with the valve closed and cooldown has been timestamped, a single background worker collects post-roll and writes the clips. Crop/encoding exceptions are logged and recorded as `recording_failed` metadata when possible, but never escape back into valve cleanup, cooldown, PARK, or the control API.

For a click-to-aim shot that is still at `AIM READY`, the crop center is the verified native camera pixel selected by the operator. If the operator subsequently moves with the D-pad, fires from an arbitrary manual angle, or fires from PARK, there is no verified inverse pan/tilt-to-pixel mapping; those shots use the configured fixed fallback `(640, 360)`. The implementation does not invent inverse calibration coordinates. Crop bounds clamp to the source frame, keep its aspect ratio, and resize the 2x crop back to the source playback dimensions without changing the live stream.

Completed recordings are stored with the normal event archive under `captures/events/YYYY-MM-DD/manual-fire-.../`. `manual_fire_zoom.avi` is the primary replay, `manual_fire_full.avi` retains the full field, `snapshot.jpg` is the zoomed review frame, and `event.json` contains shot angles, the configured 0.40-second pulse, crop source/center/bounds, zoom, timing, filenames, pre-roll availability, and recording status. AVI playback FPS is calculated only after the elapsed recording window completes as `captured frame count / monotonic window seconds`. Thermal throttling therefore produces fewer frames and a lower playback FPS over the same approximately 7-second clip instead of shortening it. The site navigation's `Events` page and the manual-control `Rewatch saved manual-fire videos` link both open this archive. It labels each completed recording `Manual fire` and links both clips. MJPG AVI is used because it is already the project's reliable OpenCV/Pi event format; no external H.264 dependency is added.

## Computer-aided auto-fire (default disabled)

`auto_fire.enabled` defaults to `false`; installing or restarting this revision does not arm automatic firing. When explicitly enabled, the backend can operate headlessly with no browser connected. It reuses the sole camera, motion detector, existing classifier worker, shared `ManualControlService` coordinator, valve controller, and fire recorder. The dashboard observes the mode but does not drive it, and there is no auto-fire polling thread or worker.

The configured allowlist contains the actual VOC model labels `dog` and `bird`, each requiring at least 0.75 confidence. `person` is a hard deny: any person detection rejects and latches the entire source event. Classification errors, missing or unknown results, non-model labels, low confidence, other classes, and person-sized motion candidates all fail closed. A candidate must also be a confirmed, event-eligible live track in daylight, with a classification no more than 3.0 seconds old and a matching target observation no more than 0.75 seconds old. The target must interpolate inside the calibrated area. Automatic aiming additionally requires all nine calibration pixels to have been saved with one authoritative native-frame width and height; legacy calibration files remain valid for manual AIM but cannot auto-fire until every point is re-verified. A camera fallback or reconnect at any other frame geometry is rejected before movement, and a geometry change during aiming is rejected at the final check. After the servo move and settle, the backend repeats the interpolation and immediately rechecks enabled state, daylight, the person latch, rate-limit clock health, classification freshness, and the same current event/track/target before allowing the valve pulse.

An accepted automatic sequence is `move -> settle -> final safety check -> valve ON for 0.40s -> valve OFF -> cooldown -> PARK`. It uses the same non-queueing physical-control lock as manual actions. An automatic shot sets a 5-second cooldown; a manual FIRE sets the existing 10-second cooldown. Because the cooldown state is shared, either path blocks both paths for its remaining duration. The persistent `captures/auto-fire-rate-limit.json` ledger enforces one shot per source event, a 60-second minimum re-engagement delay for the same event or tracker ID, and at most six accepted automatic shots in a rolling hour across restarts. Historical shot records are retained instead of being destructively pruned. The state also stores wall/monotonic clock evidence; a suspicious forward/backward wall-clock jump on the same boot, a backward clock across reboot, or an unavailable clock latches a fail-closed error. Startup atomically writes or replaces the current state before an enabled service can engage; a corrupt, unreadable, or unwritable state fails closed before movement or firing. If writing an accepted shot later fails, subsequent automatic attempts also fail closed.

The dashboard indicator and `/api/status` and `/api/health` report whether auto-fire is enabled, candidate/accept/reject counts, the latest decision and reason, cooldown remaining, rolling-hour usage, and persistence health. Structured logs distinguish candidates, rejections, and completed shots. Every accepted automatic pulse attempts to queue the shared-camera evidence recorder under an `auto-fire-...` event directory. If that queue operation fails after a pulse, the shot still counts against the persistent limits and further automatic engagement fails closed. A completed `event.json` records the source motion event and track, classifier label/confidence, classified and target observation times, target pixel and bounding box, calculated pan/tilt and calibration cell, interpolation/safe-bound results, acceptance reason, 0.40-second pulse, 5-second cooldown, recording status, and clip metadata.

No automated test, mock, or desktop run establishes Raspberry Pi camera, classifier timing, GPIO, servo, valve, targeting, or wet-fire behavior. Before leaving this mode enabled, perform all of the following on the deployed Pi under direct owner supervision:

1. Leave `auto_fire.enabled: false`, pull/restart the exact revision, and confirm the dashboard, `/api/status`, and `/api/health` all report disabled. Confirm ordinary motion events still receive one best-frame classification after event completion and no physical output occurs.
2. With water disconnected and auto-fire still disabled, open `EDIT CALIBRATION` at the production camera geometry and reselect/save the native pixel for every one of Points 1-9. Recheck every saved aim under the existing supervised dry/wet calibration procedure. Confirm status reports calibration frame width 1280, height 720, verified points 1-9, and `complete: true`; do not proceed on a different negotiated frame size.
3. Archive any prior rate-limit state needed for diagnosis, start from a reviewed valid state, and confirm `captures/auto-fire-rate-limit.json` is atomically writable by the service account. Verify the Pi wall clock is synchronized before enabling; a latched clock/persistence error requires investigation and a service restart, not a dashboard refresh.
4. Shut off and disconnect the water supply, establish a clear exclusion zone, keep an immediate power cutoff within reach, and enable auto-fire only for the dry test. Use a nonliving `dog` or `bird` test target whose centroid is safely inside a previously wet-verified calibration cell. Stop immediately if the reported source event/track, target pixel, frame dimensions, interpolated aim, or movement differs from the observed target.
5. Observe one complete dry sequence. Confirm move and settle finish before the final check, the valve can energize for only 0.40 seconds with no water connected, the valve is confirmed OFF before PARK, and both manual and automatic requests are blocked for the automatic 5-second cooldown.
6. With outputs kept dry and safe, exercise fail-closed cases: a printed/person test input, night mode, stale or vanished track, 640x480/current-vs-calibration geometry mismatch, outside-calibration target, busy coordinator, one-per-event rule, minimum re-engagement interval, rolling-hour cap, clock jump, and corrupt/unwritable persisted state. Confirm the rejection reason, structured log, dashboard count, and persisted record agree; no rejected case may pulse the valve.
7. Only after the dry checks pass, restore water with the exclusion zone clear, aim at one known fixed nonliving target, and allow exactly one automatic event. Confirm a single 0.40-second shot, valve closure before PARK, the 5-second cooldown, and the `auto-fire-.../event.json` and replay metadata. Keep the system attended and disable auto-fire after the test; this one wet check is not evidence for unattended operation or for every field condition.

## Nine-block calibration procedure

The nine blocks replace the earlier painted-X marker concept; the calibration geometry and point numbering are unchanged.

1. On desktop, select the active calibration point.
2. Click the center of the corresponding physical block in the live camera image.
3. Confirm the backend reports the native camera pixel X/Y and restores its marker after refresh.
4. Go outside with the phone and manually aim using the D-pad.
5. FIRE the configured 0.40-second pulse and observe the water landing point.
6. Adjust aim while the backend-enforced 10-second cooldown runs; fire again only when ready.
7. Once the water hits the block, press `SAVE AIM` on desktop.
8. `SAVE AIM` preserves the pixel and records the current backend-commanded pan/tilt; the point becomes green and fully calibrated. Repeating the process for that point uses `UPDATE AIM`.
9. Select the next point and repeat through all nine blocks.

## First physical verification

No new click-to-aim, 85/82 PARK, or manual-fire recording behavior is claimed as physically verified by automated tests.

1. Pull/restart the existing `manual-control` deployment with the water supply disconnected or shut off. Open the desktop page and confirm `Calibration: 9 / 9`.
2. Select `AIM TARGET`. Confirm the yellow boundary, four blue cells, and all markers 1-9 appear. In particular, marker 9 must sit at the stored center of the bottom-right red block.
3. Click exactly on marker 9. Confirm native X/Y is `1121 / 412`, status is `IN RANGE`, cell is `5-6-8-9`, calculated aim is approximately pan 48 / tilt 104, movement ends at `AIM READY`, and the valve/MOSFET never energizes.
4. Repeat the dry check on Point 1 `(446,172 -> 84/79)`, Point 5 `(640,246 -> 70/88)`, and the other six numbered markers. Stop if any exact marker is rejected or fails to reproduce its saved aim.
5. Click a clearly interior location in each of the four blue cells. Confirm it reports the expected cell, moves and settles only, and produces no valve activity. Do not judge water accuracy yet.
6. Click just outside the yellow perimeter. Confirm `OUT OF RANGE`, a visible reason, and no movement. A click below Point 9 may correctly be outside even when it is still visually on the lower part of the boundary block; the yellow line is authoritative and is intentionally not expanded.
7. Only after those dry checks pass, restore water under supervision. Click one known anchor, wait for `AIM READY` and complete servo stillness, then press the separate FIRE button once. Confirm the shot lasts 0.40 seconds and repeated FIRE remains blocked by the 10-second cooldown.
8. Perform a separate dry-fire PARK check and observe that the valve is OFF before movement to pan 85 / tilt 82. Confirm this owner-selected rest angle reduces dripping without binding or overtravel; CENTER stays 85/85 and servo limits remain unchanged.
9. Wait at least 5 seconds after one accepted FIRE, then open the event archive. Confirm the `Manual fire` card opens a zoom replay centered on the clicked target and a separate full-field clip. Repeat once after a D-pad adjustment and confirm the metadata reports `configured_fixed_fallback`; tune only `manual_control.recording.crop_center_x/y` if that fallback does not cover the installed garden target area.

Automatic firing on click, prediction, bursts, and guard-mode firing remain intentionally unimplemented. The only computer-aided engagement path is the default-disabled, one-shot, qualified-event mode described above.

## Repository boundary

`Misc/` is owner-only storage and is off-limits to normal coding work. Do not inspect, modify, stage, or commit anything beneath it. Preserve all existing modified and untracked files there.
