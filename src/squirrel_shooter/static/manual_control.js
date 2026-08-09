(function () {
  'use strict';
  var cfg = window.SS_MANUAL_CONTROL;
  if (!cfg) { return; }

  var selectedStep = cfg.initial.default_step;
  var selectedCalibrationPoint = 1;
  var requestPending = false;
  var control = cfg.initial;
  var toastTimer = null;
  var els = {
    pan: document.getElementById('pan-angle'),
    tilt: document.getElementById('tilt-angle'),
    state: document.getElementById('control-state'),
    fire: document.getElementById('fire-button'),
    fireStatus: document.getElementById('fire-status'),
    servoNote: document.getElementById('servo-note'),
    positionNote: document.getElementById('position-note'),
    valveNote: document.getElementById('valve-note'),
    point: document.getElementById('calibration-point'),
    save: document.getElementById('save-calibration'),
    savedCount: document.getElementById('saved-count'),
    calibrationComplete: document.getElementById('calibration-complete'),
    calibrationButtons: document.querySelectorAll('[data-calibration-point]'),
    calibrationConfirmation: document.getElementById('calibration-confirmation'),
    calibrationConfirmationTitle: document.getElementById('calibration-confirmation-title'),
    calibrationConfirmationPan: document.getElementById('calibration-confirmation-pan'),
    calibrationConfirmationTilt: document.getElementById('calibration-confirmation-tilt'),
    detailPoint: document.getElementById('calibration-detail-point'),
    detailPixelX: document.getElementById('calibration-detail-pixel-x'),
    detailPixelY: document.getElementById('calibration-detail-pixel-y'),
    detailPan: document.getElementById('calibration-detail-pan'),
    detailTilt: document.getElementById('calibration-detail-tilt'),
    toast: document.getElementById('manual-toast')
  };

  function showToast(message) {
    els.toast.textContent = message;
    els.toast.classList.add('visible');
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(function () { els.toast.classList.remove('visible'); }, 3200);
  }

  function busyState(state) {
    return state === 'MOVING' || state === 'SETTLING' || state === 'FIRING';
  }

  function calibrationRecord(next, point) {
    return next.calibration_points.find(function (record) { return record.point === point; }) || null;
  }

  function storedPixel(value) {
    return value === null || value === undefined ? 'Not recorded (null)' : value;
  }

  function renderCalibration(next) {
    var savedCount = next.calibration_points.length;
    var selected = calibrationRecord(next, selectedCalibrationPoint);
    els.savedCount.textContent = 'Calibration: ' + savedCount + ' / 9';
    els.calibrationComplete.hidden = savedCount !== 9;
    els.calibrationButtons.forEach(function (button) {
      var point = Number(button.dataset.calibrationPoint);
      var saved = Boolean(calibrationRecord(next, point));
      var active = point === selectedCalibrationPoint;
      button.classList.toggle('saved', saved);
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
      button.querySelector('.calibration-point-state').textContent = saved ? 'Saved' : 'Not saved';
    });
    els.point.value = selectedCalibrationPoint;
    els.save.textContent = (selected ? 'Update point ' : 'Save point ') + selectedCalibrationPoint;
    els.detailPoint.textContent = selectedCalibrationPoint;
    els.detailPixelX.textContent = selected ? storedPixel(selected.pixel_x) : 'Not recorded (null)';
    els.detailPixelY.textContent = selected ? storedPixel(selected.pixel_y) : 'Not recorded (null)';
    els.detailPan.textContent = selected ? selected.pan + '°' : 'Not saved';
    els.detailTilt.textContent = selected ? selected.tilt + '°' : 'Not saved';
  }

  function showCalibrationConfirmation(record, updated) {
    els.calibrationConfirmationTitle.textContent = 'Point ' + record.point + (updated ? ' updated' : ' saved');
    els.calibrationConfirmationPan.textContent = record.pan;
    els.calibrationConfirmationTilt.textContent = record.tilt;
    els.calibrationConfirmation.hidden = false;
  }

  function render(next) {
    control = next;
    els.pan.textContent = next.pan;
    els.tilt.textContent = next.tilt;
    els.state.textContent = next.state;
    var remaining = Math.max(0, Math.ceil(next.cooldown_remaining_seconds));
    var busy = requestPending || busyState(next.state);
    var movementDisabled = !next.servo_available || busy;
    document.querySelectorAll('[data-direction]').forEach(function (button) { button.disabled = movementDisabled; });
    els.fire.disabled = !next.valve_available || busy || remaining > 0;
    els.fireStatus.textContent = !next.valve_available ? 'VALVE NOT CONFIGURED' : (remaining > 0 ? 'READY IN ' + remaining + 's' : (busy ? next.state : 'READY'));
    els.servoNote.hidden = next.servo_available;
    els.positionNote.hidden = next.position_commanded;
    els.valveNote.hidden = next.valve_available;
    els.save.disabled = requestPending || !next.servo_available || !next.position_commanded;
    renderCalibration(next);
  }

  async function requestJson(url, body) {
    requestPending = true;
    render(control);
    try {
      var response = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json', 'X-Control-Token': cfg.token},
        body: JSON.stringify(body || {})
      });
      var payload = await response.json();
      if (payload.control) { render(payload.control); }
      if (!response.ok) { throw new Error(payload.error || 'Control request failed'); }
      return payload;
    } finally {
      requestPending = false;
      render(control);
    }
  }

  async function move(direction) {
    if (requestPending) { return; }
    try { await requestJson(cfg.urls.move, {direction: direction, step: selectedStep}); }
    catch (error) { showToast(error.message); }
  }

  document.querySelectorAll('[data-step]').forEach(function (button) {
    button.addEventListener('click', function () {
      selectedStep = Number(button.dataset.step);
      document.querySelectorAll('[data-step]').forEach(function (other) {
        var selected = other === button;
        other.classList.toggle('selected', selected);
        other.setAttribute('aria-pressed', String(selected));
      });
    });
  });
  document.querySelectorAll('[data-direction]').forEach(function (button) {
    button.addEventListener('click', function () { move(button.dataset.direction); });
  });
  els.calibrationButtons.forEach(function (button) {
    button.addEventListener('click', function () {
      selectedCalibrationPoint = Number(button.dataset.calibrationPoint);
      els.calibrationConfirmation.hidden = true;
      renderCalibration(control);
    });
  });
  els.fire.addEventListener('click', async function () {
    if (requestPending || els.fire.disabled) { return; }
    try { await requestJson(cfg.urls.fire, {}); showToast('Valve pulse complete. Cooldown started.'); }
    catch (error) { showToast(error.message); }
  });
  els.save.addEventListener('click', async function () {
    try {
      var updating = Boolean(calibrationRecord(control, selectedCalibrationPoint));
      var payload = await requestJson(cfg.urls.calibration, {point: selectedCalibrationPoint, pixel_x: null, pixel_y: null});
      showCalibrationConfirmation(payload.calibration_point, updating);
      showToast('Point ' + payload.calibration_point.point + (updating ? ' updated' : ' saved') + ' — pan ' + payload.calibration_point.pan + '°, tilt ' + payload.calibration_point.tilt + '°.');
    } catch (error) { showToast(error.message); }
  });
  document.addEventListener('keydown', function (event) {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) { return; }
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName)) { return; }
    var directions = {ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right'};
    if (directions[event.key]) { event.preventDefault(); move(directions[event.key]); }
  });

  async function refresh() {
    try {
      var response = await fetch(cfg.urls.status, {cache: 'no-store'});
      if (response.ok) { var payload = await response.json(); render(payload.control); }
    } catch (_error) { /* Keep the last known state during a brief network interruption. */ }
  }
  render(control);
  window.setInterval(refresh, 250);
}());
