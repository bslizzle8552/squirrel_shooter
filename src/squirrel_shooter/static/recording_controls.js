/* Backend deadlines own recording lifetime. Browser state is presentation only. */
(function () {
  'use strict';
  var cfg = window.SS_CONFIG || {};
  var urls = cfg.recordingUrls || {};
  var start = document.getElementById('record-30');
  var stop = document.getElementById('record-stop');
  var label = document.getElementById('record-status');
  var pending = false;
  if (!start || !stop || !label) return;
  function render(state) {
    start.disabled = pending || !state.ready || !state.enabled || !!state.last_error;
    stop.disabled = pending || !state.active || !state.manual_until;
    label.textContent = (state.active ? 'Recording' : 'Inactive') +
      ' | Manual remaining: ' + Math.ceil(state.manual_remaining_seconds || 0) + 's' +
      (state.automatic_active ? ' | Automatic reason active; STOP clears manual only' : '') +
      ' | Session: ' + (state.session_id || 'none') +
      ' | Clean status: ' + (state.status || 'idle') +
      ' | Dropped: ' + (state.queue_dropped || 0) +
      (state.error || state.last_error ? ' | Error: ' + (state.error || state.last_error) : '');
  }
  async function request(action) {
    if (pending) return;
    pending = true;
    start.disabled = stop.disabled = true;
    var controller = new AbortController();
    var timeout = setTimeout(function () { controller.abort(); }, 8000);
    try {
      var response = await fetch(urls[action], {
        method: action === 'status' ? 'GET' : 'POST', cache: 'no-store', signal: controller.signal,
        headers: action === 'status' ? {} : { 'X-Control-Token': cfg.recordingToken || '' }
      });
      var payload = await response.json();
      pending = false;
      render(payload.recording || {});
      if (!response.ok) label.textContent += ' | ' + (payload.error || 'Request rejected');
    } catch (error) {
      label.textContent = 'Recording status unavailable; backend deadlines still apply. Refresh to reconnect.';
    } finally {
      pending = false;
      clearTimeout(timeout);
    }
  }
  start.addEventListener('click', function () { request('start'); });
  stop.addEventListener('click', function () { request('stop'); });
  request('status');
  setInterval(function () { if (!document.hidden) request('status'); }, 1000);
}());
