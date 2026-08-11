(function () {
  'use strict';
  var cfg = window.SS_MANUAL_CONTROL;
  if (!cfg) { return; }

  var selectedStep = cfg.initial.default_step;
  var selectedCalibrationPoint = cfg.initial.active_calibration_point;
  var cameraMode = cfg.initial.targeting.enabled ? 'aim' : 'calibration';
  var requestPending = false;
  var refreshPending = false;
  var stateRevision = 0;
  var control = cfg.initial;
  var toastTimer = null;
  var els = {
    pan: document.getElementById('pan-angle'),
    tilt: document.getElementById('tilt-angle'),
    state: document.getElementById('control-state'),
    fire: document.getElementById('fire-button'),
    park: document.getElementById('park-button'),
    fireStatus: document.getElementById('fire-status'),
    camera: document.getElementById('calibration-camera'),
    image: document.getElementById('calibration-image'),
    geometryOverlay: document.getElementById('calibration-geometry-overlay'),
    marker: document.getElementById('calibration-marker'),
    markerLabel: document.getElementById('calibration-marker-label'),
    targetMarker: document.getElementById('target-marker'),
    cameraModeButtons: document.querySelectorAll('[data-camera-mode]'),
    cameraModeNote: document.getElementById('camera-mode-note'),
    targetingStatus: document.getElementById('targeting-status'),
    targetingError: document.getElementById('targeting-error'),
    targetPixelX: document.getElementById('target-pixel-x'),
    targetPixelY: document.getElementById('target-pixel-y'),
    targetPan: document.getElementById('target-pan'),
    targetTilt: document.getElementById('target-tilt'),
    targetRange: document.getElementById('target-range'),
    targetCell: document.getElementById('target-cell'),
    targetMethod: document.getElementById('target-method'),
    pixelStatus: document.getElementById('pixel-selection-status'),
    servoNote: document.getElementById('servo-note'),
    positionNote: document.getElementById('position-note'),
    valveNote: document.getElementById('valve-note'),
    activePoint: document.getElementById('active-calibration-point'),
    point: document.getElementById('calibration-point'),
    save: document.getElementById('save-calibration'),
    saveAction: document.getElementById('save-aim-action'),
    savePoint: document.getElementById('save-aim-point'),
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
  var streamSrc = els.image.getAttribute('src');
  var refreshController = null;
  var REFRESH_TIMEOUT_MS = 8000;

  function showToast(message) {
    els.toast.textContent = message;
    els.toast.classList.add('visible');
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(function () { els.toast.classList.remove('visible'); }, 3200);
  }

  function busyState(state) {
    return state === 'MOVING' || state === 'SETTLING' || state === 'FIRING' || state === 'PARKING';
  }

  function calibrationRecord(next, point) {
    return next.calibration_points.find(function (record) { return record.point === point; }) || null;
  }

  function storedPixel(value) {
    return value === null || value === undefined ? 'Not recorded (null)' : value;
  }

  function frameLayout(next) {
    var frameWidth = Number(next.camera_frame_width);
    var frameHeight = Number(next.camera_frame_height);
    var rect = els.image.getBoundingClientRect();
    if (!(frameWidth > 0 && frameHeight > 0 && rect.width > 0 && rect.height > 0)) { return null; }
    var scale = Math.min(rect.width / frameWidth, rect.height / frameHeight);
    var renderedWidth = frameWidth * scale;
    var renderedHeight = frameHeight * scale;
    return {
      frameWidth: frameWidth,
      frameHeight: frameHeight,
      rect: rect,
      renderedWidth: renderedWidth,
      renderedHeight: renderedHeight,
      offsetX: (rect.width - renderedWidth) / 2,
      offsetY: (rect.height - renderedHeight) / 2
    };
  }

  function renderPixelMarker(next, selected) {
    if (cameraMode !== 'calibration' || !selected || !selected.pixel_selected) {
      els.marker.hidden = true;
      els.pixelStatus.textContent = 'Point ' + selectedCalibrationPoint + ' pixel: not selected';
      return;
    }
    els.pixelStatus.textContent = 'Point ' + selectedCalibrationPoint + ' pixel selected: X ' + selected.pixel_x + ', Y ' + selected.pixel_y;
    var layout = frameLayout(next);
    if (!layout) { els.marker.hidden = true; return; }
    var cameraRect = els.camera.getBoundingClientRect();
    var markerX = layout.rect.left - cameraRect.left + layout.offsetX + ((selected.pixel_x + 0.5) / layout.frameWidth) * layout.renderedWidth;
    var markerY = layout.rect.top - cameraRect.top + layout.offsetY + ((selected.pixel_y + 0.5) / layout.frameHeight) * layout.renderedHeight;
    els.marker.style.left = markerX + 'px';
    els.marker.style.top = markerY + 'px';
    els.markerLabel.textContent = 'Point ' + selectedCalibrationPoint;
    els.marker.hidden = false;
  }

  function renderTargetMarker(next) {
    var target = next.targeting;
    if (cameraMode !== 'aim' || target.pixel_x === null || target.pixel_y === null) {
      els.targetMarker.hidden = true;
      return;
    }
    var layout = frameLayout(next);
    if (!layout) { els.targetMarker.hidden = true; return; }
    var cameraRect = els.camera.getBoundingClientRect();
    var markerX = layout.rect.left - cameraRect.left + layout.offsetX + ((target.pixel_x + 0.5) / layout.frameWidth) * layout.renderedWidth;
    var markerY = layout.rect.top - cameraRect.top + layout.offsetY + ((target.pixel_y + 0.5) / layout.frameHeight) * layout.renderedHeight;
    els.targetMarker.style.left = markerX + 'px';
    els.targetMarker.style.top = markerY + 'px';
    els.targetMarker.hidden = false;
  }

  function svgElement(name, attributes) {
    var element = document.createElementNS('http://www.w3.org/2000/svg', name);
    Object.keys(attributes).forEach(function (key) { element.setAttribute(key, attributes[key]); });
    return element;
  }

  function geometryPoints(points) {
    return points.map(function (point) { return point.pixel_x + ',' + point.pixel_y; }).join(' ');
  }

  function renderCalibrationGeometry(next) {
    var geometry = next.targeting.geometry;
    var layout = frameLayout(next);
    if (!geometry || !layout || !window.matchMedia('(min-width: 821px)').matches) {
      els.geometryOverlay.hidden = true;
      return;
    }
    var cameraRect = els.camera.getBoundingClientRect();
    els.geometryOverlay.setAttribute('viewBox', '0 0 ' + layout.frameWidth + ' ' + layout.frameHeight);
    els.geometryOverlay.style.left = (layout.rect.left - cameraRect.left + layout.offsetX) + 'px';
    els.geometryOverlay.style.top = (layout.rect.top - cameraRect.top + layout.offsetY) + 'px';
    els.geometryOverlay.style.width = layout.renderedWidth + 'px';
    els.geometryOverlay.style.height = layout.renderedHeight + 'px';
    els.geometryOverlay.replaceChildren();
    els.geometryOverlay.appendChild(svgElement('polygon', {
      points: geometryPoints(geometry.boundary),
      'class': 'geometry-boundary'
    }));
    geometry.cells.forEach(function (cell) {
      els.geometryOverlay.appendChild(svgElement('polygon', {
        points: geometryPoints(cell.corners),
        'class': 'geometry-cell'
      }));
    });
    geometry.anchors.forEach(function (anchor) {
      els.geometryOverlay.appendChild(svgElement('circle', {
        cx: anchor.pixel_x,
        cy: anchor.pixel_y,
        r: 13,
        'class': 'geometry-anchor'
      }));
      var label = svgElement('text', {
        x: anchor.pixel_x,
        y: anchor.pixel_y,
        'class': 'geometry-anchor-label'
      });
      label.textContent = anchor.point;
      els.geometryOverlay.appendChild(label);
    });
    els.geometryOverlay.hidden = false;
  }

  function renderCameraMode(next) {
    els.cameraModeButtons.forEach(function (button) {
      var selected = button.dataset.cameraMode === cameraMode;
      button.classList.toggle('selected', selected);
      button.setAttribute('aria-pressed', String(selected));
      button.disabled = requestPending || (button.dataset.cameraMode === 'aim' && !next.targeting.enabled);
    });
    els.cameraModeNote.textContent = cameraMode === 'aim'
      ? 'AIM TARGET: click inside the calibrated garden region to move and settle. Clicking never fires.'
      : 'EDIT CALIBRATION: select a point, then click the center of its physical block.';
    els.pixelStatus.hidden = cameraMode !== 'calibration';
    els.targetingStatus.textContent = next.targeting.status;
    els.targetPixelX.textContent = next.targeting.pixel_x === null ? '--' : next.targeting.pixel_x;
    els.targetPixelY.textContent = next.targeting.pixel_y === null ? '--' : next.targeting.pixel_y;
    els.targetPan.textContent = next.targeting.pan === null ? '--' : next.targeting.pan;
    els.targetTilt.textContent = next.targeting.tilt === null ? '--' : next.targeting.tilt;
    els.targetRange.textContent = next.targeting.in_range === true ? 'IN RANGE' : (next.targeting.in_range === false ? 'OUT OF RANGE' : '--');
    els.targetCell.textContent = next.targeting.cell ? next.targeting.cell.join('-') : '--';
    els.targetMethod.textContent = next.targeting.method || '--';
    els.targetingError.textContent = next.targeting.error || '';
    els.targetingError.hidden = !next.targeting.error;
    renderTargetMarker(next);
    renderCalibrationGeometry(next);
  }

  function renderCalibration(next) {
    selectedCalibrationPoint = next.active_calibration_point;
    var savedCount = next.completed_calibration_count;
    var selected = calibrationRecord(next, selectedCalibrationPoint);
    els.activePoint.textContent = selectedCalibrationPoint;
    els.savedCount.textContent = 'Calibration: ' + savedCount + ' / 9';
    els.calibrationComplete.hidden = !next.targeting.enabled;
    els.calibrationButtons.forEach(function (button) {
      var point = Number(button.dataset.calibrationPoint);
      var record = calibrationRecord(next, point);
      var saved = Boolean(record && record.complete);
      var pixelSelected = Boolean(record && record.pixel_selected && !record.complete);
      var aimSaved = Boolean(record && record.aim_saved && !record.complete);
      var active = point === selectedCalibrationPoint;
      button.classList.toggle('saved', saved);
      button.classList.toggle('pixel-selected', pixelSelected);
      button.classList.toggle('aim-saved', aimSaved);
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
      button.querySelector('.calibration-point-state').textContent = saved ? 'Calibrated' : (pixelSelected ? 'Pixel set' : (aimSaved ? 'Aim saved' : 'Not started'));
    });
    els.point.value = selectedCalibrationPoint;
    var updatingAim = Boolean(selected && selected.aim_saved);
    els.saveAction.textContent = updatingAim ? 'UPDATE AIM' : 'SAVE AIM';
    els.savePoint.textContent = 'Point ' + selectedCalibrationPoint + ' · current Pan/Tilt';
    els.save.setAttribute('aria-label', (updatingAim ? 'Update' : 'Save') + ' current backend aim for Point ' + selectedCalibrationPoint);
    els.save.disabled = requestPending || !next.servo_available || !next.position_commanded || !selected || !selected.pixel_selected;
    els.detailPoint.textContent = selectedCalibrationPoint;
    els.detailPixelX.textContent = selected ? storedPixel(selected.pixel_x) : 'Not recorded (null)';
    els.detailPixelY.textContent = selected ? storedPixel(selected.pixel_y) : 'Not recorded (null)';
    els.detailPan.textContent = selected && selected.aim_saved ? selected.pan + '°' : 'Not saved';
    els.detailTilt.textContent = selected && selected.aim_saved ? selected.tilt + '°' : 'Not saved';
    renderPixelMarker(next, selected);
  }

  function showCalibrationConfirmation(record, updated) {
    els.calibrationConfirmationTitle.textContent = 'Point ' + record.point + ' aim ' + (updated ? 'updated' : 'saved');
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
    els.park.disabled = movementDisabled;
    els.fire.disabled = !next.valve_available || busy || remaining > 0;
    els.fireStatus.textContent = !next.valve_available ? 'VALVE NOT CONFIGURED' : (remaining > 0 ? 'READY IN ' + remaining + 's' : (busy ? next.state : 'READY'));
    els.servoNote.hidden = next.servo_available;
    els.positionNote.hidden = next.position_commanded;
    els.valveNote.hidden = next.valve_available;
    renderCalibration(next);
    renderCameraMode(next);
  }

  async function requestJson(url, body) {
    requestPending = true;
    stateRevision += 1;
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
  els.park.addEventListener('click', async function () {
    if (requestPending || els.park.disabled) { return; }
    try { await requestJson(cfg.urls.park, {}); showToast('PARK complete at the anti-drip rest position.'); }
    catch (error) { showToast(error.message); }
  });
  els.calibrationButtons.forEach(function (button) {
    button.addEventListener('click', async function () {
      if (requestPending) { return; }
      els.calibrationConfirmation.hidden = true;
      try { await requestJson(cfg.urls.activeCalibration, {point: Number(button.dataset.calibrationPoint)}); }
      catch (error) { showToast(error.message); }
    });
  });
  els.cameraModeButtons.forEach(function (button) {
    button.addEventListener('click', function () {
      if (requestPending || button.disabled) { return; }
      cameraMode = button.dataset.cameraMode;
      render(control);
    });
  });
  els.image.addEventListener('click', async function (event) {
    if (requestPending || !window.matchMedia('(min-width: 821px)').matches) { return; }
    var rect = els.image.getBoundingClientRect();
    try {
      var click = {
        display_x: event.clientX - rect.left,
        display_y: event.clientY - rect.top,
        display_width: rect.width,
        display_height: rect.height
      };
      if (cameraMode === 'aim') {
        var aimPayload = await requestJson(cfg.urls.aim, click);
        showToast('AIM READY: X ' + aimPayload.target.pixel_x + ', Y ' + aimPayload.target.pixel_y + '; pan ' + aimPayload.target.pan + ', tilt ' + aimPayload.target.tilt + '. FIRE remains manual.');
      } else {
        var payload = await requestJson(cfg.urls.calibrationPixel, click);
      showToast('Point ' + payload.calibration_point.point + ' pixel selected — X ' + payload.calibration_point.pixel_x + ', Y ' + payload.calibration_point.pixel_y + '.');
      }
    } catch (error) { showToast(error.message); }
  });
  els.fire.addEventListener('click', async function () {
    if (requestPending || els.fire.disabled) { return; }
    try { await requestJson(cfg.urls.fire, {}); showToast('Valve pulse complete, valve OFF, PARK complete. Cooldown active.'); }
    catch (error) { showToast(error.message); }
  });
  els.save.addEventListener('click', async function () {
    try {
      var selected = calibrationRecord(control, selectedCalibrationPoint);
      var updating = Boolean(selected && selected.aim_saved);
      var payload = await requestJson(cfg.urls.calibration, {});
      showCalibrationConfirmation(payload.calibration_point, updating);
      showToast('Point ' + payload.calibration_point.point + ' aim ' + (updating ? 'updated' : 'saved') + ' — pan ' + payload.calibration_point.pan + '°, tilt ' + payload.calibration_point.tilt + '°.');
    } catch (error) { showToast(error.message); }
  });
  document.addEventListener('keydown', function (event) {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) { return; }
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName)) { return; }
    var directions = {ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right'};
    if (directions[event.key]) { event.preventDefault(); move(directions[event.key]); }
  });
  window.addEventListener('resize', function () {
    renderPixelMarker(control, calibrationRecord(control, selectedCalibrationPoint));
    renderTargetMarker(control);
    renderCalibrationGeometry(control);
  });

  async function refresh() {
    if (document.hidden || requestPending || refreshPending) { return; }
    var revision = stateRevision;
    refreshPending = true;
    var controller = new AbortController();
    var timeout = window.setTimeout(function () { controller.abort(); }, REFRESH_TIMEOUT_MS);
    refreshController = controller;
    try {
      var response = await fetch(cfg.urls.status, {cache: 'no-store', signal: controller.signal});
      if (response.ok) {
        var payload = await response.json();
        if (!requestPending && revision === stateRevision) { render(payload.control); }
      }
    } catch (_error) { /* Keep the last known state during a brief network interruption. */ }
    finally {
      window.clearTimeout(timeout);
      if (refreshController === controller) { refreshController = null; }
      refreshPending = false;
    }
  }

  function syncStreamVisibility() {
    if (document.hidden && els.image.hasAttribute('src')) {
      els.image.removeAttribute('src');
    } else if (!document.hidden && !els.image.hasAttribute('src')) {
      els.image.setAttribute('src', streamSrc);
    }
  }

  document.addEventListener('visibilitychange', function () {
    syncStreamVisibility();
    if (document.hidden && refreshController) {
      refreshController.abort();
    } else if (!document.hidden) {
      window.setTimeout(refresh, 0);
    }
  });
  syncStreamVisibility();
  render(control);
  window.setInterval(refresh, cfg.pollIntervalMs);
}());
