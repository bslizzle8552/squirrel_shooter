/* Squirrel Squirter — Garden Mission Control console.
   Vanilla ES6. Polling (no WebSocket), rAF-batched rendering, recycled DOM rows. */
(function () {
  'use strict';

  var cfg = window.SS_CONFIG || {};
  var urls = cfg.urls || {};
  var ROW_H = 78; // must match --row-h in console.css
  var WINDOW = (cfg.queueWindow || 10) + 2; // small overscan
  var MAX_ITEMS = cfg.queueLimit || 100;
  var HISTORY_MAX = 48;

  /* ---------- state (plain objects, no proxies) ---------- */
  var App = {
    pane: 'review', // 'review' | 'recent'
    reviewItems: [],
    recentItems: [],
    counts: { review: 0 },
    decided: 0,
    cursor: 0,
    scrollStart: 0,
    selected: null, // { kind, data }
    mode: 'observe',
    signatures: { review: '', recent: '' },
    hydrated: { review: false, recent: false },
    history: { fps: [], temp: [], queue: [] },
    prefs: { outdoor: false, pauseStream: false },
    streamSrc: null,
    filmKey: ''
  };

  /* ---------- dom refs ---------- */
  function byId(id) { return document.getElementById(id); }
  var els = {
    body: document.body,
    queueList: byId('queue-list'),
    recentList: byId('recent-list'),
    tabCountReview: byId('tab-count-review'),
    tabCountRecent: byId('tab-count-recent'),
    queueProgress: byId('queue-progress'),
    queueProgressBar: byId('queue-progress-bar'),
    rowTemplate: byId('queue-row-template'),
    liveView: byId('live-view'),
    eventView: byId('event-view'),
    videoFrame: byId('video-frame'),
    liveStream: byId('live-stream'),
    offlineTitle: byId('offline-title'),
    cameraError: byId('camera-error'),
    notice: byId('detector-notice'),
    noticeTitle: byId('notice-title'),
    noticeDetail: byId('notice-detail'),
    hudCamPill: byId('hud-cam-pill'),
    hudCamValue: byId('hud-cam-value'),
    hudDetector: byId('hud-detector'),
    hudFps: byId('hud-fps'),
    inspectorImage: byId('inspector-image'),
    inspectorSnapshot: byId('inspector-snapshot'),
    inspectorClip: byId('inspector-clip'),
    decisionDeck: byId('decision-deck'),
    inspectorHint: byId('inspector-hint'),
    filmstrip: byId('filmstrip'),
    metaTime: byId('meta-time'),
    metaLabel: byId('meta-label'),
    metaModel: byId('meta-model'),
    metaInference: byId('meta-inference'),
    metaSource: byId('meta-source'),
    metaEvent: byId('meta-event'),
    fpsValue: byId('fps-value'),
    tempValue: byId('temp-value'),
    queueValue: byId('queue-value'),
    modeValue: byId('mode-value'),
    camValue: byId('cam-value'),
    camDot: byId('cam-dot'),
    statFps: byId('stat-fps'),
    statTemp: byId('stat-temp'),
    statQueue: byId('stat-queue'),
    sparkFps: byId('spark-fps'),
    sparkTemp: byId('spark-temp'),
    sparkQueue: byId('spark-queue'),
    toast: byId('toast'),
    outdoorToggle: byId('outdoor-toggle'),
    systemDialog: byId('system-dialog'),
    helpDialog: byId('help-dialog'),
    prefOutdoor: byId('pref-outdoor'),
    prefPauseStream: byId('pref-pause-stream'),
    sysCamera: byId('sys-camera'),
    sysDetector: byId('sys-detector'),
    sysTemp: byId('sys-temp'),
    sysUptime: byId('sys-uptime'),
    sysAccepted: byId('sys-accepted'),
    sysRejected: byId('sys-rejected'),
    sysClassifier: byId('sys-classifier'),
    sysStorage: byId('sys-storage')
  };

  var rowPool = [];
  var renderPending = false;
  var toastTimer = 0;

  /* ---------- helpers ---------- */
  function clamp(value, low, high) { return value < low ? low : (value > high ? high : value); }

  function timeOf(iso) {
    if (typeof iso !== 'string' || iso.length < 16) { return ''; }
    return iso.slice(11, 16);
  }

  function pct(value) {
    return typeof value === 'number' ? Math.round(value * 100) + '%' : '';
  }

  function suggestionLine(item) {
    if (item.review_suggestion_label) {
      return 'Maybe ' + item.review_suggestion_label + ' ' + pct(item.review_suggestion_confidence);
    }
    if (item.top_label) {
      return 'Model ' + item.top_label + ' ' + pct(item.top_confidence);
    }
    return 'No model match';
  }

  function getJSON(url, ok) {
    fetch(url, { cache: 'no-store' })
      .then(function (response) { if (!response.ok) { throw new Error('http ' + response.status); } return response.json(); })
      .then(ok)
      .catch(function () { /* next poll recovers; never blank good data */ });
  }

  function showToast(message, isError) {
    els.toast.textContent = message;
    els.toast.classList.toggle('error', !!isError);
    els.toast.classList.add('show');
    if (toastTimer) { clearTimeout(toastTimer); }
    toastTimer = setTimeout(function () { els.toast.classList.remove('show'); }, 2600);
  }

  function setText(el, text) {
    if (el.textContent === text) { return; }
    el.textContent = text;
    el.classList.remove('tick');
    void el.offsetWidth; // restart the animation
    el.classList.add('tick');
  }

  function markLoaded(scope) {
    var imgs = scope.querySelectorAll('img:not(.loaded)');
    for (var i = 0; i < imgs.length; i++) {
      (function (img) {
        if (img.complete && img.naturalWidth > 0) { img.classList.add('loaded'); return; }
        img.addEventListener('load', function () { img.classList.add('loaded'); }, { once: true });
      })(imgs[i]);
    }
  }

  function scrollXIntoView(container, el) {
    if (!container || !el) { return; }
    var c = container.getBoundingClientRect();
    var r = el.getBoundingClientRect();
    if (r.left < c.left) { container.scrollLeft -= (c.left - r.left) + 10; }
    else if (r.right > c.right) { container.scrollLeft += (r.right - c.right) + 10; }
  }

  /* ---------- sparklines ---------- */
  function pushHistory(key, value) {
    if (typeof value !== 'number' || !isFinite(value)) { return; }
    var series = App.history[key];
    series.push(value);
    if (series.length > HISTORY_MAX) { series.shift(); }
  }

  function drawSpark(canvas, series, color) {
    if (!canvas) { return; }
    var dpr = window.devicePixelRatio || 1;
    var w = 72, h = 24;
    if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
    var ctx = canvas.getContext('2d');
    if (!ctx) { return; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (series.length < 2) { return; }
    var min = Math.min.apply(null, series);
    var max = Math.max.apply(null, series);
    var span = max - min;
    if (span < 0.001) { min -= 0.5; span = 1; }
    var pad = 2;
    function xAt(i) { return pad + (i / (series.length - 1)) * (w - pad * 2); }
    function yAt(v) { return h - pad - ((v - min) / span) * (h - pad * 2); }
    ctx.beginPath();
    for (var i = 0; i < series.length; i++) {
      if (i === 0) { ctx.moveTo(xAt(i), yAt(series[i])); }
      else { ctx.lineTo(xAt(i), yAt(series[i])); }
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.4;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.stroke();
    // soft area fill under the line
    ctx.lineTo(xAt(series.length - 1), h - pad);
    ctx.lineTo(xAt(0), h - pad);
    ctx.closePath();
    ctx.fillStyle = color.replace('rgb(', 'rgba(').replace(')', ', 0.12)');
    ctx.fill();
    // last-value dot
    ctx.beginPath();
    ctx.arc(xAt(series.length - 1), yAt(series[series.length - 1]), 1.8, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }

  function drawSparks() {
    drawSpark(els.sparkFps, App.history.fps, 'rgb(94, 169, 255)');
    drawSpark(els.sparkTemp, App.history.temp, 'rgb(245, 165, 36)');
    drawSpark(els.sparkQueue, App.history.queue, 'rgb(62, 207, 130)');
  }

  /* ---------- rendering (rAF batched) ---------- */
  function requestRender() {
    if (renderPending) { return; }
    renderPending = true;
    requestAnimationFrame(function () {
      renderPending = false;
      renderQueue();
      renderRecent();
      renderInspector();
      renderPane();
    });
  }

  function renderPane() {
    els.body.dataset.pane = App.pane;
    els.tabCountReview.textContent = String(App.counts.review || 0);
    els.tabCountReview.classList.toggle('zero', (App.counts.review || 0) === 0);
    els.tabCountRecent.textContent = String(App.recentItems.length);
  }

  function updateProgress() {
    var remaining = App.counts.review || 0;
    var total = remaining + App.decided;
    if (remaining === 0) {
      els.queueProgress.textContent = App.decided > 0 ? 'All caught up — ' + App.decided + ' labeled' : 'All clear';
      els.queueProgressBar.style.width = '100%';
    } else {
      els.queueProgress.textContent = remaining + ' of ' + total + ' remaining';
      els.queueProgressBar.style.width = total > 0 ? Math.round((App.decided / total) * 100) + '%' : '0%';
    }
    setText(els.queueValue, String(remaining));
    els.statQueue.classList.toggle('warn', remaining > 50);
    renderPane();
  }

  function ensureRowPool() {
    var node;
    while (rowPool.length < WINDOW) {
      node = els.rowTemplate.content.firstElementChild.cloneNode(true);
      els.queueList.appendChild(node);
      rowPool.push(node);
    }
  }

  function emptyRow(message) {
    var li = document.createElement('li');
    li.className = 'row-empty';
    li.textContent = message;
    return li;
  }

  function allClearRow() {
    var li = document.createElement('li');
    li.className = 'row-empty all-clear';
    li.innerHTML = '<svg aria-hidden="true"><use href="#i-clear"/></svg><strong>All clear</strong><span>Queue clear — nothing needs review.</span>';
    return li;
  }

  function fillReviewRow(node, item, idx) {
    node.dataset.itemId = item.item_id;
    node.dataset.index = String(idx);
    node.classList.toggle('selected', App.pane === 'review' && idx === App.cursor);
    if (App.pane === 'review' && idx === App.cursor) { node.setAttribute('aria-current', 'true'); }
    else { node.removeAttribute('aria-current'); }
    var thumb = node.querySelector('.row-thumb');
    thumb.href = urls.reviewPage;
    var img = thumb.querySelector('img');
    if (img.getAttribute('src') !== item.image_url) {
      img.classList.remove('loaded');
      img.src = item.image_url;
    }
    img.alt = 'Classifier input for ' + (item.event_id || '');
    var meta = node.querySelector('.row-meta');
    meta.children[0].textContent = timeOf(item.classifier_timestamp);
    meta.children[1].textContent = item.display_label || 'Unclassified';
    meta.children[2].textContent = suggestionLine(item);
  }

  function renderQueue() {
    if (!App.hydrated.review) { return; } // server markup stands until first poll
    var items = App.reviewItems;
    var i, idx, node;
    var stale = els.queueList.querySelector('.row-empty');
    if (stale) { els.queueList.removeChild(stale); }
    if (!items.length) {
      for (i = 0; i < rowPool.length; i++) { rowPool[i].hidden = true; }
      els.queueList.style.paddingTop = '0px';
      els.queueList.style.paddingBottom = '0px';
      els.queueList.appendChild(allClearRow());
      return;
    }
    ensureRowPool();
    var start = clamp(App.scrollStart - 1, 0, Math.max(0, items.length - WINDOW));
    var end = Math.min(items.length, start + WINDOW);
    for (i = 0; i < rowPool.length; i++) {
      idx = start + i;
      node = rowPool[i];
      if (idx >= end) { node.hidden = true; continue; }
      node.hidden = false;
      fillReviewRow(node, items[idx], idx);
    }
    els.queueList.style.paddingTop = (start * ROW_H) + 'px';
    els.queueList.style.paddingBottom = (Math.max(0, items.length - end) * ROW_H) + 'px';
    markLoaded(els.queueList);
  }

  function buildRecentRow(item, index) {
    var row = document.createElement('li');
    row.className = 'row';
    row.dataset.kind = 'recent';
    row.dataset.eventId = item.event_id || '';
    row.dataset.index = String(index);
    var selected = App.pane === 'recent' && index === App.cursor;
    row.classList.toggle('selected', selected);
    if (selected) { row.setAttribute('aria-current', 'true'); }
    var thumb = document.createElement('a');
    thumb.className = 'row-thumb';
    if (item.snapshot_url) {
      thumb.href = item.snapshot_url;
      var img = document.createElement('img');
      img.src = item.snapshot_url;
      img.alt = 'Event ' + (item.event_id || '');
      img.loading = 'lazy';
      img.decoding = 'async';
      thumb.appendChild(img);
    } else {
      thumb.hidden = true;
    }
    var meta = document.createElement('span');
    meta.className = 'row-meta';
    var time = document.createElement('time');
    time.textContent = timeOf(item.start_timestamp);
    var label = document.createElement('strong');
    label.textContent = item.display_label || 'Unclassified';
    var motion = document.createElement('small');
    motion.textContent = 'Motion: ' + String(item.motion_label || 'unclassified motion').replace(/_/g, ' ');
    meta.appendChild(time);
    meta.appendChild(label);
    meta.appendChild(motion);
    var actions = document.createElement('span');
    actions.className = 'row-actions';
    if (item.clip_url) {
      var link = document.createElement('a');
      link.className = 'mini-btn';
      link.href = item.clip_url;
      link.title = 'Open video clip';
      link.setAttribute('aria-label', 'Open video clip');
      var icon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      var use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
      use.setAttribute('href', '#i-clip');
      icon.appendChild(use);
      link.appendChild(icon);
      actions.appendChild(link);
    }
    row.appendChild(thumb);
    row.appendChild(meta);
    row.appendChild(actions);
    return row;
  }

  function renderRecent() {
    if (!App.hydrated.recent) { return; } // server markup stands until first poll
    var items = App.recentItems;
    els.recentList.replaceChildren();
    if (!items.length) {
      els.recentList.appendChild(emptyRow('No completed grouped events yet. Filtered motion stays visible in the live feed.'));
      return;
    }
    for (var i = 0; i < items.length; i++) {
      els.recentList.appendChild(buildRecentRow(items[i], i));
    }
    markLoaded(els.recentList);
  }

  /* ---------- inspector + filmstrip ---------- */
  function renderInspector() {
    var sel = App.selected;
    if (!sel) { renderFilmstrip(); return; }
    var data = sel.data;
    if (sel.kind === 'review') {
      if (els.inspectorImage.getAttribute('src') !== data.image_url) { els.inspectorImage.src = data.image_url; }
      els.inspectorImage.alt = 'Classifier input for ' + (data.event_id || '');
      els.metaTime.textContent = data.classifier_timestamp ? data.classifier_timestamp.replace('T', ' ').slice(0, 19) : '—';
      els.metaLabel.textContent = data.display_label || 'Unclassified';
      els.metaModel.textContent = data.top_label ? (data.top_label + ' ' + pct(data.top_confidence)) : 'No class over threshold';
      els.metaInference.textContent = typeof data.latency_ms === 'number' ? Math.round(data.latency_ms) + ' ms · frame ' + (data.frame_number || 1) : 'Unavailable';
      els.metaSource.textContent = data.label_source || 'Not labeled';
      els.metaEvent.textContent = data.event_id || data.item_id || '—';
      setLink(els.inspectorSnapshot, data.event_snapshot_url);
      setLink(els.inspectorClip, data.event_clip_url);
      els.decisionDeck.hidden = false;
      els.inspectorHint.hidden = true;
    } else {
      var snap = data.snapshot_url || '';
      if (els.inspectorImage.getAttribute('src') !== snap) { els.inspectorImage.src = snap; }
      els.inspectorImage.alt = 'Event ' + (data.event_id || '');
      els.metaTime.textContent = data.start_timestamp ? data.start_timestamp.replace('T', ' ').slice(0, 19) : '—';
      els.metaLabel.textContent = data.display_label || 'Unclassified';
      els.metaModel.textContent = 'Motion: ' + String(data.motion_label || 'unclassified motion').replace(/_/g, ' ');
      els.metaInference.textContent = data.duration ? data.duration + ' s clip' : '—';
      els.metaSource.textContent = data.classification_label_source || 'Not labeled';
      els.metaEvent.textContent = data.event_id || '—';
      setLink(els.inspectorSnapshot, data.snapshot_url);
      setLink(els.inspectorClip, data.clip_url);
      els.decisionDeck.hidden = true;
      els.inspectorHint.hidden = true;
    }
    renderFilmstrip();
  }

  function setLink(link, url) {
    if (url) { link.hidden = false; link.href = url; } else { link.hidden = true; link.removeAttribute('href'); }
  }

  function renderFilmstrip() {
    var sel = App.selected;
    var ctx = sel ? sel.kind : null;
    var items = ctx === 'review' ? App.reviewItems : (ctx === 'recent' ? App.recentItems : []);
    var selectedId = sel ? (ctx === 'review' ? sel.data.item_id : sel.data.event_id) : '';
    var key = ctx + '|' + selectedId + '|' + items.map(function (item) {
      return ctx === 'review' ? item.item_id : item.event_id;
    }).join(',');
    if (key === App.filmKey) { return; }
    App.filmKey = key;
    els.filmstrip.replaceChildren();
    if (!ctx || !items.length) {
      var empty = document.createElement('span');
      empty.className = 'film-empty';
      empty.textContent = ctx ? 'No more items in this list.' : 'Select an event — its queue neighbors appear here.';
      els.filmstrip.appendChild(empty);
      return;
    }
    for (var i = 0; i < items.length; i++) {
      (function (item, idx) {
        var id = ctx === 'review' ? item.item_id : item.event_id;
        var url = ctx === 'review' ? item.image_url : item.snapshot_url;
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'film-btn' + (id === selectedId ? ' selected' : '');
        btn.dataset.index = String(idx);
        btn.setAttribute('role', 'option');
        btn.setAttribute('aria-selected', id === selectedId ? 'true' : 'false');
        btn.title = (ctx === 'review' ? (item.display_label || 'Unclassified') : ('Event ' + (item.event_id || '')));
        if (url) {
          var img = document.createElement('img');
          img.src = url;
          img.alt = '';
          img.loading = 'lazy';
          img.decoding = 'async';
          btn.appendChild(img);
        }
        var time = document.createElement('time');
        time.textContent = timeOf(ctx === 'review' ? item.classifier_timestamp : item.start_timestamp);
        btn.appendChild(time);
        els.filmstrip.appendChild(btn);
      })(items[i], i);
    }
    var current = els.filmstrip.querySelector('.film-btn.selected');
    if (current) { scrollXIntoView(els.filmstrip, current); }
  }

  /* ---------- mode / selection ---------- */
  function setMode(mode) {
    App.mode = mode;
    els.body.dataset.mode = mode;
    setText(els.modeValue, mode === 'observe' ? 'OBSERVE' : 'REVIEW');
  }

  function selectItem(kind, data) {
    App.selected = { kind: kind, data: data };
    setMode('inspect');
    var url = kind === 'review' ? data.image_url : data.snapshot_url;
    if (url) {
      var pre = new Image();
      pre.src = url;
    }
    requestRender();
  }

  function selectCursor() {
    if (App.pane === 'review' && App.reviewItems[App.cursor]) {
      selectItem('review', App.reviewItems[App.cursor]);
    } else if (App.pane === 'recent' && App.recentItems[App.cursor]) {
      selectItem('recent', App.recentItems[App.cursor]);
    }
  }

  function stepSelection(delta) {
    // ← / → while inspecting: move through the same list the selection came from
    var sel = App.selected;
    if (!sel) { return; }
    var items = sel.kind === 'review' ? App.reviewItems : App.recentItems;
    if (!items.length) { return; }
    var id = sel.kind === 'review' ? sel.data.item_id : sel.data.event_id;
    var idx = -1;
    for (var i = 0; i < items.length; i++) {
      if ((sel.kind === 'review' ? items[i].item_id : items[i].event_id) === id) { idx = i; break; }
    }
    var next = clamp((idx < 0 ? 0 : idx) + delta, 0, items.length - 1);
    if (next === idx) { return; }
    App.cursor = next;
    selectItem(sel.kind, items[next]);
  }

  function moveCursor(delta) {
    var items = App.pane === 'review' ? App.reviewItems : App.recentItems;
    if (!items.length) { return; }
    App.cursor = clamp(App.cursor + delta, 0, items.length - 1);
    if (App.pane === 'review') {
      var list = els.queueList;
      var top = App.cursor * ROW_H;
      if (top < list.scrollTop) { list.scrollTop = top; }
      else if (top + ROW_H > list.scrollTop + list.clientHeight) { list.scrollTop = top + ROW_H - list.clientHeight; }
    } else {
      var row = els.recentList.children[App.cursor];
      if (row) { scrollXIntoView(els.recentList, row); }
    }
    requestRender();
  }

  function setPane(pane) {
    if (App.pane === pane) { return; }
    App.pane = pane;
    App.cursor = 0;
    requestRender();
  }

  function backToLive() {
    App.selected = null;
    setMode('observe');
    requestRender();
  }

  /* ---------- decisions ---------- */
  function indexOfReview(itemId) {
    for (var i = 0; i < App.reviewItems.length; i++) {
      if (App.reviewItems[i].item_id === itemId) { return i; }
    }
    return -1;
  }

  function advanceAfterDecision(itemId) {
    App.decided += 1;
    var idx = indexOfReview(itemId);
    if (idx >= 0) { App.reviewItems.splice(idx, 1); }
    if (App.counts.review > 0) { App.counts.review -= 1; }
    if (App.selected && App.selected.kind === 'review' && App.selected.data.item_id === itemId) {
      if (App.reviewItems.length) {
        var next = clamp(idx < 0 ? 0 : idx, 0, App.reviewItems.length - 1);
        App.cursor = next;
        App.selected = { kind: 'review', data: App.reviewItems[next] };
      } else {
        App.selected = null;
        setMode('observe');
      }
    }
    App.cursor = clamp(App.cursor, 0, Math.max(0, App.reviewItems.length - 1));
    App.signatures.review = ''; // force list re-render after optimistic removal
    updateProgress();
    requestRender();
  }

  function decide(itemId, decision, label) {
    if (!itemId) { return; }
    var url = urls.decision.replace('ITEM_ID', encodeURIComponent(itemId)).replace('DECISION', encodeURIComponent(decision));
    var body = new URLSearchParams();
    body.set('review_token', cfg.reviewToken || '');
    body.set('format', 'json');
    if (label) { body.set('approval_label', label); }
    els.decisionDeck.classList.add('busy');
    fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: body.toString()
    })
      .then(function (response) { if (!response.ok) { throw new Error('http ' + response.status); } return response.json(); })
      .then(function (payload) {
        showToast(payload.message || 'Saved');
        advanceAfterDecision(itemId);
        pollReview();
      })
      .catch(function () { showToast('Label could not be saved — try again', true); })
      .then(function () { els.decisionDeck.classList.remove('busy'); });
  }

  /* ---------- polling ---------- */
  function pollStatus() {
    getJSON(urls.status, function (data) {
      var camera = data.camera || {};
      var detector = data.detector || {};
      var fps = typeof detector.processing_fps === 'number' ? detector.processing_fps : 0;
      setText(els.fpsValue, fps.toFixed(1));
      els.statFps.classList.toggle('low', fps > 0 && fps < 5);
      var temp = data.cpu_temperature_c;
      setText(els.tempValue, temp === null || temp === undefined ? '--' : temp.toFixed(1) + '°C');
      els.statTemp.classList.toggle('warn', typeof temp === 'number' && temp > 60 && temp <= 75);
      els.statTemp.classList.toggle('alert', typeof temp === 'number' && temp > 75);
      var state = camera.state || 'OFFLINE';
      setText(els.camValue, state);
      els.camDot.className = 'dot ' + state.toLowerCase();
      // HUD overlays
      els.hudCamPill.className = 'hud-pill cam-pill ' + state.toLowerCase();
      setText(els.hudCamValue, state);
      var dState = detector.state || '';
      setText(els.hudDetector, dState || '—');
      setText(els.hudFps, fps.toFixed(1) + ' FPS');
      els.hudDetector.classList.toggle('hot', dState === 'LEARNING' || dState === 'WARMING_UP' || dState === 'NIGHT_PAUSED');
      els.videoFrame.dataset.online = camera.online ? 'true' : 'false';
      els.offlineTitle.textContent = 'Camera ' + state.toLowerCase();
      els.cameraError.textContent = camera.error || 'No fresh frames are available. The dashboard and diagnostics remain online.';
      var learning = dState === 'LEARNING' || dState === 'WARMING_UP';
      var night = dState === 'NIGHT_PAUSED';
      els.notice.classList.toggle('learning', learning);
      els.noticeTitle.textContent = night ? 'Night vision paused' : (learning ? 'LEARNING background' : 'Motion-only diagnostics');
      els.noticeDetail.textContent = night
        ? 'The live feed remains on, but clips and classifier work are paused until daytime color returns.'
        : (learning ? 'Events are suppressed until the detector reaches READY.' : 'Labels describe size and movement; they are not person, car, or squirrel recognition.');
      els.sysCamera.textContent = state + ' · ' + ((camera.resolution && camera.resolution.label) || '');
      els.sysDetector.textContent = dState + ' · ' + fps.toFixed(1) + ' FPS';
      els.sysTemp.textContent = temp === null || temp === undefined ? 'Unavailable' : temp.toFixed(1) + ' °C';
      els.sysUptime.textContent = Math.round(data.application_uptime_seconds || 0) + 's';
      els.sysAccepted.textContent = String(detector.accepted_events != null ? detector.accepted_events : (data.total_events || 0));
      els.sysRejected.textContent = String(detector.rejected_events != null ? detector.rejected_events : 0);
      var classifier = data.classifier || {};
      els.sysClassifier.textContent = 'depth ' + (classifier.queue_depth != null ? classifier.queue_depth : 0);
      els.sysStorage.textContent = (data.total_snapshots != null ? data.total_snapshots : 0) + ' snapshots';
      // sparkline history
      pushHistory('fps', fps);
      if (typeof temp === 'number') { pushHistory('temp', temp); }
      drawSparks();
    });
  }

  function pollEvents() {
    getJSON(urls.events, function (data) {
      var events = (data.events || []).slice(0, 10);
      var signature = JSON.stringify(events.map(function (event) {
        return [event.event_id, event.display_label, event.classification_label_source, event.snapshot_url, event.clip_url];
      }));
      if (App.hydrated.recent && signature === App.signatures.recent) { return; }
      if (!App.hydrated.recent) { App.hydrated.recent = true; }
      App.signatures.recent = signature;
      App.recentItems = events;
      if (App.pane === 'recent') { App.cursor = clamp(App.cursor, 0, Math.max(0, events.length - 1)); }
      requestRender();
    });
  }

  function pollReview() {
    getJSON(urls.review + '?state=review&limit=' + MAX_ITEMS, function (data) {
      App.counts = data.counts || App.counts;
      pushHistory('queue', App.counts.review || 0);
      updateProgress();
      var items = data.items || [];
      var signature = JSON.stringify(items.map(function (item) {
        return [item.item_id, item.classification_status, item.display_label, item.top_label, item.top_confidence];
      }));
      if (App.hydrated.review && signature === App.signatures.review) { renderPane(); return; }
      if (!App.hydrated.review) {
        els.queueList.replaceChildren();
        rowPool.length = 0;
        App.hydrated.review = true;
      }
      App.signatures.review = signature;
      App.reviewItems = items;
      App.cursor = clamp(App.cursor, 0, Math.max(0, items.length - 1));
      // drop a stale selection that no longer exists
      if (App.selected && App.selected.kind === 'review') {
        var found = false;
        for (var i = 0; i < items.length; i++) {
          if (items[i].item_id === App.selected.data.item_id) { found = true; break; }
        }
        if (!found) { backToLive(); return; }
      }
      requestRender();
    });
  }

  /* ---------- preferences ---------- */
  function savePrefs() {
    try { localStorage.setItem('ss.prefs', JSON.stringify(App.prefs)); } catch (_) { /* kiosk storage may be off */ }
  }

  function loadPrefs() {
    try {
      var raw = localStorage.getItem('ss.prefs');
      if (raw) {
        var parsed = JSON.parse(raw);
        App.prefs.outdoor = !!parsed.outdoor;
        App.prefs.pauseStream = !!parsed.pauseStream;
      }
    } catch (_) { /* defaults are fine */ }
  }

  function applyPrefs() {
    els.body.classList.toggle('outdoor', App.prefs.outdoor);
    els.outdoorToggle.setAttribute('aria-pressed', App.prefs.outdoor ? 'true' : 'false');
    els.prefOutdoor.checked = App.prefs.outdoor;
    els.prefPauseStream.checked = App.prefs.pauseStream;
    if (App.prefs.pauseStream) {
      if (!App.streamSrc) { App.streamSrc = els.liveStream.src; }
      els.liveStream.removeAttribute('src');
    } else if (App.streamSrc && !els.liveStream.src) {
      els.liveStream.src = App.streamSrc;
    }
  }

  /* ---------- events (delegated) ---------- */
  function onQueueClick(event) {
    var action = event.target.closest('[data-decision]');
    var row = event.target.closest('.row');
    if (action && row) {
      event.preventDefault();
      decide(row.dataset.itemId, action.dataset.decision, action.dataset.label || null);
      return;
    }
    if (row) {
      if (event.target.closest('a')) { return; } // let native links work
      App.pane = 'review';
      App.cursor = parseInt(row.dataset.index || '0', 10) || 0;
      selectCursor();
    }
  }

  function onRecentClick(event) {
    var row = event.target.closest('.row');
    if (!row || event.target.closest('a')) { return; }
    App.pane = 'recent';
    App.cursor = parseInt(row.dataset.index || '0', 10) || 0;
    selectCursor();
  }

  function onKey(event) {
    if (event.target.matches('input, select, textarea')) { return; }
    if (els.systemDialog.open || els.helpDialog.open) { return; }
    var key = event.key;
    if (key === 'ArrowDown' || key === 'j') { event.preventDefault(); moveCursor(1); }
    else if (key === 'ArrowUp' || key === 'k') { event.preventDefault(); moveCursor(-1); }
    else if (key === 'ArrowRight') {
      event.preventDefault();
      if (App.selected) { stepSelection(1); } else { moveCursor(1); }
    }
    else if (key === 'ArrowLeft') {
      event.preventDefault();
      if (App.selected) { stepSelection(-1); } else { moveCursor(-1); }
    }
    else if (key === 'Enter') { event.preventDefault(); selectCursor(); }
    else if (key === 'q') { setPane(App.pane === 'review' ? 'recent' : 'review'); }
    else if (key === 'l' || key === 'Escape') { backToLive(); }
    else if (key === 'o') { App.prefs.outdoor = !App.prefs.outdoor; savePrefs(); applyPrefs(); }
    else if (key === 's') { els.systemDialog.showModal(); }
    else if (key === '?') { els.helpDialog.showModal(); }
    else if ((key === '1' || key === '2' || key === '3' || key === '0') && App.selected && App.selected.kind === 'review') {
      var id = App.selected.data.item_id;
      if (key === '1') { decide(id, 'approve', 'car'); }
      else if (key === '2') { decide(id, 'approve', 'person'); }
      else if (key === '3') { decide(id, 'unknown', null); }
      else { decide(id, 'false-positive', null); }
    }
  }

  function onScroll() {
    App.scrollStart = Math.floor(els.queueList.scrollTop / ROW_H);
    if (App.pane === 'review') { requestRender(); }
  }

  var resizePending = false;
  function onResize() {
    if (resizePending) { return; }
    resizePending = true;
    requestAnimationFrame(function () {
      resizePending = false;
      els.body.classList.toggle('mobile', window.innerWidth < 900);
      drawSparks();
    });
  }

  /* ---------- boot ---------- */
  function boot() {
    loadPrefs();
    applyPrefs();
    onResize();
    // stage views crossfade via body[data-mode]; the initial hidden attribute is markup-only
    els.eventView.removeAttribute('hidden');
    els.body.dataset.pane = App.pane;

    els.queueList.addEventListener('click', onQueueClick);
    els.queueList.addEventListener('scroll', onScroll, { passive: true });
    els.recentList.addEventListener('click', onRecentClick);
    els.filmstrip.addEventListener('click', function (event) {
      var btn = event.target.closest('.film-btn');
      if (!btn || !App.selected) { return; }
      var idx = parseInt(btn.dataset.index || '0', 10) || 0;
      var items = App.selected.kind === 'review' ? App.reviewItems : App.recentItems;
      if (items[idx]) {
        App.cursor = idx;
        selectItem(App.selected.kind, items[idx]);
      }
    });
    els.decisionDeck.addEventListener('click', function (event) {
      var button = event.target.closest('[data-decision]');
      if (button && App.selected && App.selected.kind === 'review') {
        decide(App.selected.data.item_id, button.dataset.decision, button.dataset.label || null);
      }
    });
    byId('inspector-live').addEventListener('click', backToLive);
    els.outdoorToggle.addEventListener('click', function () {
      App.prefs.outdoor = !App.prefs.outdoor;
      savePrefs();
      applyPrefs();
    });
    els.prefOutdoor.addEventListener('change', function () {
      App.prefs.outdoor = els.prefOutdoor.checked;
      savePrefs();
      applyPrefs();
    });
    els.prefPauseStream.addEventListener('change', function () {
      App.prefs.pauseStream = els.prefPauseStream.checked;
      savePrefs();
      applyPrefs();
    });
    byId('system-open').addEventListener('click', function () { els.systemDialog.showModal(); });
    byId('help-open').addEventListener('click', function () { els.helpDialog.showModal(); });
    document.addEventListener('keydown', onKey);
    window.addEventListener('resize', onResize);

    markLoaded(document);
    updateProgress();
    pollStatus();
    pollEvents();
    pollReview();
    setInterval(pollStatus, cfg.statusPollMs || 5000);
    setInterval(pollEvents, cfg.eventsPollMs || 2000);
    setInterval(pollReview, cfg.reviewPollMs || 3000);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
