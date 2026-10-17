(function () {
  'use strict';

  // Preview JPEGs may be smaller than the sensor image. Map the contained image
  // through native dimensions supplied by the existing camera's status instead.
  function frameLayout(rect, width, height) {
    if (!(width > 0 && height > 0 && rect.width > 0 && rect.height > 0)) return null;
    const scale = Math.min(rect.width / width, rect.height / height);
    return {left: rect.left + (rect.width - width * scale) / 2,
      top: rect.top + (rect.height - height * scale) / 2, width, height, scale};
  }

  function nativePixel(layout, clientX, clientY) {
    if (!layout) return null;
    const x = (clientX - layout.left) / layout.scale;
    const y = (clientY - layout.top) / layout.scale;
    if (x < 0 || y < 0 || x >= layout.width || y >= layout.height) return null;
    return {pixel_x: Math.floor(x), pixel_y: Math.floor(y)};
  }

  // A tiny pure interface lets the coordinate contract be tested without a DOM.
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {frameLayout, nativePixel};
    return;
  }
  const settings = JSON.parse(document.getElementById('control-settings').textContent);
  const el = id => document.getElementById(id);
  const testMode = settings.mode === 'test';
  const preview = el('camera-preview');
  let status = null, selected = 1, pixel = null, dryAimPixel = null, connected = false;
  let pendingAction = false, awaitingResult = null, timer = null, retryTimer = null;
  let polling = false, refreshController = null, active = false, generation = 0;

  function message(text, error = false) {
    el('message').textContent = text;
    el('message').className = error ? 'error' : '';
  }

  function cameraReady() {
    return !!(preview && preview.naturalWidth > 0 && status?.camera?.online && status.camera.width > 0 && status.camera.height > 0);
  }

  function layout() {
    return cameraReady() ? frameLayout(preview.getBoundingClientRect(), status.camera.width, status.camera.height) : null;
  }

  function drawReticle() {
    const marker = el('reticle'), frame = layout();
    if (!marker) return;
    marker.hidden = !pixel || !frame;
    if (marker.hidden) return;
    const container = el('camera-frame').getBoundingClientRect();
    marker.style.left = frame.left - container.left + (pixel.pixel_x + 0.5) * frame.scale + 'px';
    marker.style.top = frame.top - container.top + (pixel.pixel_y + 0.5) * frame.scale + 'px';
  }

  function savedPoint(point = selected) {
    return status?.calibration?.points?.find(record => record.point === point);
  }

  function resultText(result) {
    if (!result) return '';
    if (typeof result === 'string') return 'Last result: ' + result;
    const outcomes = {position_commanded: 'Movement settling time elapsed',
      aimed_without_water: 'Dry aim settled; water stayed OFF', manual_pulse_closed: 'Manual pulse finished; valve closed',
      dry_run: 'Dry sequence finished; water stayed OFF', fired: 'Shot finished; valve closed and head parked',
      rejected: 'Action rejected', error: 'Control error'};
    return 'Last result: ' + (outcomes[result.outcome] || result.outcome || 'Command finished') +
      (result.pan == null ? '' : ' · PAN ' + Number(result.pan).toFixed(1) + '° / TILT ' + Number(result.tilt).toFixed(1) + '°') +
      (result.error ? ' · ' + result.error : '');
  }

  function render() {
    const enabled = connected && active && !!status?.enabled;
    const available = enabled && !pendingAction && !status?.busy;
    const camera = cameraReady();
    const record = savedPoint();
    const calibration = status?.calibration;
    const mismatch = camera && calibration?.camera_width &&
      (calibration.camera_width !== status.camera.width || calibration.camera_height !== status.camera.height);
    el('map-status').textContent = mismatch ? 'STALE: saved calibration resolution differs from the live camera. Reset and recalibrate.' :
      calibration?.error ? 'Invalid calibration: ' + calibration.error :
      calibration?.complete ? 'Calibration complete for the saved geometry. Recalibrate after moving the camera or nozzle.' :
      'Calibration INCOMPLETE: ' + (calibration?.points?.length || 0) + ' / 9 points saved for this location.';
    el('saved-points').textContent = (calibration?.points || []).map(p =>
      'Point ' + p.point + ': X ' + p.pixel_x + ', Y ' + p.pixel_y + ' | PAN ' + p.pan + '°, TILT ' + p.tilt + '°').join('\n') || 'No measured points saved.';
    el('pan-angle').textContent = status?.position?.pan == null ? '—' : Number(status.position.pan).toFixed(1) + '°';
    el('tilt-angle').textContent = status?.position?.tilt == null ? '—' : Number(status.position.tilt).toFixed(1) + '°';
    el('pixel-position').textContent = pixel ? 'X ' + pixel.pixel_x + ' / Y ' + pixel.pixel_y : 'not selected';
    el('control-status').textContent = !connected ? 'Control connection unavailable' :
      'Physical control ' + (status.enabled ? 'ON' : 'OFF') + ' · ' + status.state +
      ' · Manual water ' + (status.valve_enabled ? 'enabled' : 'OFF') +
      ' · Automatic sequence ' + (status.automatic_fire_enabled ? (status.dry_run ? 'DRY' : 'water enabled') : 'OFF');
    el('camera-status').textContent = camera ? 'Native image ' + status.camera.width + ' × ' + status.camera.height +
      ' · Existing collector preview' : 'Camera or native dimensions unavailable';
    if (el('park')) el('park').disabled = !available;
    if (!testMode) {
      const calibrating = available && !!status?.calibration_mode;
      el('selected-point').textContent = 'Point ' + selected + (record ? ' · LOCKED' : ' · empty');
      el('point-count').textContent = (status?.calibration?.points?.length || 0) + ' / 9 locked';
      document.querySelectorAll('[data-point]').forEach(button => {
        const number = Number(button.dataset.point), locked = !!savedPoint(number);
        button.disabled = !calibrating;
        button.classList.toggle('locked', locked);
        button.setAttribute('aria-pressed', String(number === selected));
        button.setAttribute('aria-label', 'Point ' + number + (locked ? ', locked; load saved aim' : ', empty'));
        button.querySelector('span').textContent = locked ? '✓' : '—';
      });
      document.querySelectorAll('[data-direction]').forEach(button => { button.disabled = !calibrating; });
      el('move-step').disabled = !calibrating;
      el('center').disabled = !calibrating;
      const reticle = status.active_reticle;
      el('fire').disabled = !(calibrating && camera &&
        (!reticle || (pixel && reticle.pixel_x === pixel.pixel_x && reticle.pixel_y === pixel.pixel_y)) &&
        status.position?.pan != null && status.position?.tilt != null &&
        status.valve_enabled && !status.error && !(status.cooldown_remaining_seconds > 0));
      el('lock').hidden = !!record;
      el('relock').hidden = !record;
      const canLock = calibrating && camera && !!pixel && status.position?.pan != null && status.position?.tilt != null;
      el('lock').disabled = !canLock || !!record;
      el('relock').disabled = !canLock || !record;
      el('complete').disabled = !calibrating || !camera || status.calibration?.points?.length !== 9;
      const incompatible = status?.calibration?.error || (status?.calibration?.camera_width && camera &&
        (status.calibration.camera_width !== status.camera.width || status.calibration.camera_height !== status.camera.height));
      el('reset').hidden = !incompatible;
      el('reset').disabled = !calibrating || !camera;
    } else {
      const valid = !!status?.calibration?.complete && !status?.calibration?.error;
      el('calibration-status').textContent = valid ? 'All nine calibration points saved.' : 'A valid complete calibration is required.';
      el('dry-aim').disabled = !(available && camera && valid && pixel);
      el('dry-sequence').disabled = el('dry-aim').disabled || status.cooldown_remaining_seconds > 0;
      const aimed = status?.last_result?.action === 'dry_aim' && status.last_result.outcome === 'aimed_without_water' &&
        status.last_result.pixel?.[0] === pixel?.pixel_x && status.last_result.pixel?.[1] === pixel?.pixel_y;
      el('fire').disabled = !(available && camera && valid && dryAimPixel && pixel &&
        dryAimPixel.pixel_x === pixel.pixel_x && dryAimPixel.pixel_y === pixel.pixel_y &&
        aimed && status.valve_enabled && !status.error && !(status.cooldown_remaining_seconds > 0));
    }
    const result = status?.last_result;
    el('last-result').textContent = resultText(result);
    if (status?.error || status?.calibration?.error) message(status.error || status.calibration.error, true);
    drawReticle();
  }

  async function refresh() {
    if (!active || polling || pendingAction) return;
    polling = true;
    const ownGeneration = generation;
    const controller = new AbortController();
    refreshController = controller;
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      const response = await fetch('/api/control', {cache: 'no-store', signal: controller.signal});
      if (!response.ok) throw Error('Control status unavailable (' + response.status + ')');
      const next = await response.json();
      if (!active || ownGeneration !== generation) return;
      const previousCamera = status?.camera;
      if (previousCamera && (next.camera?.width !== previousCamera.width || next.camera?.height !== previousCamera.height)) {
        pixel = null;
        dryAimPixel = null;
      }
      status = next;
      connected = true;
      if (awaitingResult && !next.busy && next.last_result?.action === awaitingResult) {
        const failed = ['rejected', 'error'].includes(next.last_result.outcome);
        message(failed ? next.last_result.error || 'Command rejected.' : 'Command finished.', failed);
        awaitingResult = null;
      }
      if (!testMode && next.selected_point && next.selected_point !== selected) {
        selected = next.selected_point;
        const record = savedPoint();
        pixel = record ? {pixel_x: record.pixel_x, pixel_y: record.pixel_y} : null;
      }
    } catch (error) {
      if (active && ownGeneration === generation) { connected = false; message(error.message, true); }
    } finally {
      clearTimeout(timeout);
      if (refreshController === controller) refreshController = null;
      polling = false;
      if (active) render();
    }
  }

  function schedule() {
    clearTimeout(timer);
    if (active) timer = setTimeout(poll, status?.busy ? 750 : 2000);
  }

  async function poll() {
    if (!active) return;
    await refresh();
    schedule();
  }

  async function action(name, values = {}) {
    if (!active || !connected || pendingAction || status?.busy) return false;
    pendingAction = true;
    generation += 1;
    clearTimeout(timer);
    refreshController?.abort();
    render();
    // A timed-out action may have reached the server. Never automatically retry
    // movement or a pulse: obtain fresh status before enabling another command.
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);
    try {
      const response = await fetch('/api/control/action', {method: 'POST', signal: controller.signal,
        headers: {'Content-Type': 'application/json', 'X-Control-Token': settings.token},
        body: JSON.stringify({action: name, ...values})});
      const next = await response.json();
      if (!response.ok) throw Error(next.error || 'Action rejected (' + response.status + ')');
      status = next;
      connected = true;
      awaitingResult = response.status === 202 ? name : null;
      if (name === 'select') {
        selected = values.point;
        const record = savedPoint();
        pixel = record ? {pixel_x: record.pixel_x, pixel_y: record.pixel_y} : null;
      }
      if (name === 'reticle') pixel = {pixel_x: values.pixel_x, pixel_y: values.pixel_y};
      if (name === 'reset') { pixel = null; selected = 1; }
      if (name === 'dry_aim') dryAimPixel = {...values};
      if (['park', 'center', 'move', 'fire', 'dry_sequence'].includes(name)) dryAimPixel = null;
      message(name === 'lock' ? 'Point ' + values.point + ' locked.' : response.status === 202 ? 'Command accepted; waiting for the nozzle.' : 'Command complete.');
      if (name === 'complete' && next.calibration?.complete && !next.calibration_mode) {
        suspend();
        window.location.assign('/aim-test');
      }
      return true;
    } catch (error) {
      connected = false;
      message(error.name === 'AbortError' ? 'Command response timed out. Checking status; the command will not be repeated.' : error.message, true);
      return false;
    } finally {
      clearTimeout(timeout);
      pendingAction = false;
      render();
      if (active) { await refresh(); schedule(); }
    }
  }

  if (preview) {
    preview.addEventListener('click', async event => {
      if (!active || !connected || pendingAction || status?.busy) return;
      const next = nativePixel(layout(), event.clientX, event.clientY);
      if (!next) { message('Select a point inside the camera image.'); return; }
      dryAimPixel = null;
      if (testMode) pixel = next;
      else {
        pixel = null;
        if (!await action('reticle', {point: selected, ...next})) return;
      }
      message('Pixel selected.');
      render();
    });
    preview.addEventListener('load', render);
    preview.addEventListener('error', () => {
      clearTimeout(retryTimer);
      if (active) {
        message('Preview interrupted; reconnecting…', true);
        retryTimer = setTimeout(() => { if (active) preview.src = '/preview.mjpg?t=' + Date.now(); }, 3000);
      }
    });
  }
  document.querySelectorAll('[data-point]').forEach(button => {
    button.addEventListener('click', () => action('select', {point: Number(button.dataset.point)}));
  });
  document.querySelectorAll('[data-direction]').forEach(button => {
    button.addEventListener('click', () => action('move', {direction: button.dataset.direction, step: Number(el('move-step').value)}));
  });
  for (const name of ['park', 'center', 'fire', 'complete']) {
    el(name)?.addEventListener('click', () => action(name));
  }
  el('lock')?.addEventListener('click', () => { if (pixel) action('lock', {point: selected, ...pixel, relock: false}); });
  el('relock')?.addEventListener('click', () => {
    if (pixel && window.confirm('Replace locked Point ' + selected + ' with this reticle pixel and the current commanded pan/tilt?')) {
      action('lock', {point: selected, ...pixel, relock: true});
    }
  });
  el('dry-aim')?.addEventListener('click', () => { if (pixel) action('dry_aim', {...pixel}); });
  el('dry-sequence')?.addEventListener('click', () => { if (pixel) action('dry_sequence', {...pixel}); });
  el('reset')?.addEventListener('click', () => {
    if (window.confirm('Discard all saved calibration points and start again for this camera? The calibration file will be replaced.')) {
      action('reset', {confirm: true});
    }
  });

  function suspend() {
    active = false;
    dryAimPixel = null;
    generation += 1;
    clearTimeout(timer);
    clearTimeout(retryTimer);
    refreshController?.abort();
    preview?.removeAttribute('src');
    render();
  }

  function resume() {
    if (active || document.hidden) return;
    active = true;
    connected = false;
    if (preview) preview.src = '/preview.mjpg';
    poll();
  }
  document.addEventListener('visibilitychange', () => { if (document.hidden) suspend(); else resume(); });
  window.addEventListener('pagehide', suspend);
  window.addEventListener('pageshow', resume);
  window.addEventListener('resize', drawReticle);
  resume();
}());
