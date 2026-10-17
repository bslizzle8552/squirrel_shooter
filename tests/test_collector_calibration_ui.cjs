'use strict';
// Run with: node --test tests/test_collector_calibration_ui.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = path.join(__dirname, '../src/squirrel_shooter/static/collector_calibration.js');
const {frameLayout, nativePixel} = require(source);

test('a reduced 16:9 preview maps to native pixels through letterboxing', () => {
  const layout = frameLayout({left: 100, top: 50, width: 800, height: 600}, 1280, 720);
  assert.deepEqual(nativePixel(layout, 500, 350), {pixel_x: 640, pixel_y: 360});
  assert.deepEqual(nativePixel(layout, 100, 125), {pixel_x: 0, pixel_y: 0});
  assert.deepEqual(nativePixel(layout, 899.9, 574.9), {pixel_x: 1279, pixel_y: 719});
  assert.equal(nativePixel(layout, 500, 100), null);
  assert.equal(nativePixel(layout, 900, 350), null);
  assert.equal(nativePixel(layout, 500, 575), null);
});

test('portrait containers reject side bars and unknown dimensions', () => {
  const layout = frameLayout({left: 0, top: 0, width: 800, height: 400}, 400, 800);
  assert.deepEqual(nativePixel(layout, 400, 200), {pixel_x: 200, pixel_y: 400});
  assert.equal(nativePixel(layout, 299, 200), null);
  assert.equal(nativePixel(layout, 500, 200), null);
  assert.equal(frameLayout({width: 800, height: 400}, 0, 0), null);
  assert.equal(nativePixel(null, 10, 10), null);
});

function harness(mode = 'calibration') {
  const elements = new Map(), docListeners = {}, windowListeners = {}, timers = new Map();
  const calls = [], navigations = [];
  let nextTimer = 1, deferReticle = null, confirmResult = true;
  const control = {enabled: true, valve_enabled: true, automatic_fire_enabled: false,
    dry_run: false, state: 'IDLE', busy: false, error: null, last_result: null,
    position: {pan: 85, tilt: 82}, calibration_mode: mode !== 'test',
    calibration: {complete: mode === 'test', points: [], error: null},
    camera: {online: true, width: 1280, height: 720}, selected_point: 1,
    active_reticle: null};
  function element(id) {
    if (elements.has(id)) return elements.get(id);
    const listeners = {};
    const item = {id, style: {}, dataset: {}, hidden: false, disabled: false, textContent: '', naturalWidth: 640,
      classList: {toggle() {}}, value: '1', attributes: {}, listeners,
      addEventListener(name, fn) { listeners[name] = fn; },
      setAttribute(name, value) { this.attributes[name] = value; },
      removeAttribute(name) { delete this[name]; },
      getBoundingClientRect() { return {left: 0, top: 0, width: 800, height: 600}; },
      querySelector() { return {textContent: ''}; }};
    elements.set(id, item);
    return item;
  }
  element('control-settings').textContent = JSON.stringify({token: 'test-token', mode});
  const doc = {hidden: false, getElementById: element, querySelectorAll: () => [],
    addEventListener(name, fn) { docListeners[name] = fn; }};
  const browserWindow = {confirm: () => confirmResult, location: {assign: value => navigations.push(value)},
    addEventListener(name, fn) { windowListeners[name] = fn; }};
  const response = () => ({ok: true, status: 200, json: async () => structuredClone(control)});
  const context = {document: doc, window: browserWindow, AbortController,
    setTimeout(fn, delay) { const id = nextTimer++; timers.set(id, {fn, delay}); return id; },
    clearTimeout(id) { timers.delete(id); },
    async fetch(url, options = {}) {
      const body = options.body ? JSON.parse(options.body) : null;
      calls.push({url, options, body});
      if (body?.action === 'reticle') {
        if (deferReticle) await deferReticle;
        control.active_reticle = {pixel_x: body.pixel_x, pixel_y: body.pixel_y};
      }
      if (body?.action === 'complete') {
        control.calibration.complete = true;
        control.calibration_mode = false;
      }
      if (body?.action === 'dry_aim') {
        control.last_result = {action: 'dry_aim', outcome: 'aimed_without_water', pixel: [body.pixel_x, body.pixel_y]};
      }
      return response();
    }};
  vm.runInNewContext(fs.readFileSync(source, 'utf8'), context, {filename: source});
  return {element, doc, docListeners, windowListeners, timers, calls, control, navigations,
    delayReticle(promise) { deferReticle = promise; },
    confirm(value) { confirmResult = value; },
    async tickPoll() {
      const [id, timer] = [...timers.entries()].find(([, value]) => [750, 2000].includes(value.delay));
      timers.delete(id);
      await timer.fn();
    }};
}

async function settle() {
  for (let i = 0; i < 4; i++) await new Promise(resolve => setImmediate(resolve));
}

test('page hide tears down the only preview and polling; visible return resumes', async () => {
  const page = harness();
  await settle();
  assert.equal(page.element('camera-preview').src, '/preview.mjpg');
  assert.equal(page.calls.filter(call => call.url === '/api/control').length, 1);
  assert.equal([...page.timers.values()].filter(timer => timer.delay === 2000).length, 1);
  page.doc.hidden = true;
  page.docListeners.visibilitychange();
  assert.equal(page.element('camera-preview').src, undefined);
  assert.equal(page.timers.size, 0);
  page.doc.hidden = false;
  page.docListeners.visibilitychange();
  await settle();
  assert.equal(page.calls.filter(call => call.url === '/api/control').length, 2);
  page.windowListeners.pagehide();
  assert.equal(page.timers.size, 0);
  assert.equal(page.element('camera-preview').src, undefined);
});

test('calibration FIRE works before choosing a pixel and waits during a pixel update', async () => {
  const page = harness();
  await settle();
  assert.equal(page.element('fire').disabled, false);
  let release;
  page.delayReticle(new Promise(resolve => { release = resolve; }));
  const click = page.element('camera-preview').listeners.click({clientX: 400, clientY: 300});
  await settle();
  assert.equal(page.element('fire').disabled, true);
  assert.deepEqual(page.calls.find(call => call.body).body,
    {action: 'reticle', point: 1, pixel_x: 640, pixel_y: 360});
  release();
  await click;
  await settle();
  assert.equal(page.element('fire').disabled, false);
  assert.equal(page.calls.some(call => call.body?.action === 'fire'), false);
});

test('dry test page requires a deliberate dry aim before enabling manual FIRE', async () => {
  const page = harness('test');
  await settle();
  await page.element('camera-preview').listeners.click({clientX: 400, clientY: 300});
  assert.equal(page.element('fire').disabled, true);
  page.element('dry-aim').listeners.click();
  await settle();
  assert.equal(page.calls.find(call => call.body)?.body.action, 'dry_aim');
  assert.equal(page.element('fire').disabled, false);
  await page.element('camera-preview').listeners.click({clientX: 450, clientY: 300});
  assert.equal(page.element('fire').disabled, true);
  assert.equal(page.calls.some(call => call.body?.action === 'fire'), false);
});

test('dry tests do not silently inherit the final calibration anchor as their target', async () => {
  const page = harness('test');
  page.control.selected_point = 9;
  page.control.calibration.points.push({point: 9, pixel_x: 1180, pixel_y: 620, pan: 95, tilt: 105});
  await settle();
  assert.equal(page.element('pixel-position').textContent, 'not selected');
  assert.equal(page.element('dry-aim').disabled, true);
  assert.equal(page.element('fire').disabled, true);
});

test('completion releases calibration resources before opening deliberate dry tests', async () => {
  const page = harness();
  await settle();
  page.element('complete').listeners.click();
  await settle();
  assert.deepEqual(page.navigations, ['/aim-test']);
  assert.equal(page.element('camera-preview').src, undefined);
  assert.equal(page.timers.size, 0);
});

test('relocking a saved point requires explicit confirmation and sends the relock flag', async () => {
  const page = harness();
  page.control.calibration.points.push({point: 1, pixel_x: 640, pixel_y: 360, pan: 85, tilt: 82});
  await settle();
  await page.element('camera-preview').listeners.click({clientX: 400, clientY: 300});
  assert.equal(page.element('lock').hidden, true);
  assert.equal(page.element('relock').hidden, false);
  page.confirm(false);
  page.element('relock').listeners.click();
  await settle();
  assert.equal(page.calls.some(call => call.body?.action === 'lock'), false);
  page.confirm(true);
  page.element('relock').listeners.click();
  await settle();
  assert.deepEqual(page.calls.find(call => call.body?.action === 'lock').body,
    {action: 'lock', point: 1, pixel_x: 640, pixel_y: 360, relock: true});
});

test('a native camera dimension change discards the selected pixel and fire readiness', async () => {
  const page = harness();
  await settle();
  await page.element('camera-preview').listeners.click({clientX: 400, clientY: 300});
  assert.equal(page.element('fire').disabled, false);
  page.control.camera.width = 1920;
  page.control.camera.height = 1080;
  await page.tickPoll();
  assert.equal(page.element('pixel-position').textContent, 'not selected');
  assert.equal(page.element('fire').disabled, true);
});
