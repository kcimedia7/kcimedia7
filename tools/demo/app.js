/**
 * Browser demo shell.
 *
 * Everything the hosted demo does happens on this page: frames are decoded by
 * the browser, the reconstruction runs on the main thread, and the library
 * lives in IndexedDB on the viewer's own device. Nothing is uploaded anywhere.
 *
 * The conversion pipeline and the renderer are the project's real ones, inlined
 * by tools/build-demo.mjs.
 */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const MAX_DIM = 640;

const state = {
  viewer: null,
  edits: defaultEdits(),
  current: null,      // { id, name, kind, frames, splatCount, buffer, thumb }
  library: [],
  busy: false,
  tab: 'convert',
  settings: { targetFrames: 28, detail: 150, arcDeg: 360 },
};

function defaultEdits() {
  return {
    translate: [0, 0, 0],
    rotate: [0, 0, 0],
    scale: 1,
    splatScale: 1,
    opacity: 1,
    exposure: 0,
    saturation: 1,
    crop: null,
    pruneBelowOpacity: 0,
    background: '#0a0c11',
  };
}

// ---------------------------------------------------------------- persistence

/**
 * IndexedDB rather than localStorage: a cloud is megabytes of binary, well past
 * what localStorage holds, and it survives a reload on the viewer's device.
 */
const DB_NAME = 'splatworks-demo';
let dbPromise = null;

function db() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains('captures')) {
        req.result.createObjectStore('captures', { keyPath: 'id' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  }).catch(() => null);   // private windows and blocked storage: run without it
  return dbPromise;
}

async function dbAll() {
  const conn = await db();
  if (!conn) return [];
  return new Promise((resolve) => {
    const req = conn.transaction('captures').objectStore('captures').getAll();
    req.onsuccess = () => resolve((req.result || []).sort((a, b) => b.createdAt - a.createdAt));
    req.onerror = () => resolve([]);
  });
}

async function dbPut(record) {
  const conn = await db();
  if (!conn) return false;
  return new Promise((resolve) => {
    const tx = conn.transaction('captures', 'readwrite');
    tx.objectStore('captures').put(record);
    tx.oncomplete = () => resolve(true);
    tx.onerror = () => resolve(false);
  });
}

async function dbDelete(id) {
  const conn = await db();
  if (!conn) return;
  await new Promise((resolve) => {
    const tx = conn.transaction('captures', 'readwrite');
    tx.objectStore('captures').delete(id);
    tx.oncomplete = resolve;
    tx.onerror = resolve;
  });
}

// ------------------------------------------------------------ frame extraction

function isVideoFile(file) {
  return file.type.startsWith('video/') || /\.(mp4|mov|m4v|webm|avi|mkv)$/i.test(file.name);
}

/** Decode a selection of files into ImageData frames, in capture order. */
async function extractFrames(files, { targetFrames, onProgress }) {
  const videos = files.filter(isVideoFile);
  const images = files.filter((f) => !isVideoFile(f));
  const frames = [];
  const total = images.length + videos.length * targetFrames;
  let done = 0;

  for (const file of images) {
    let bitmap;
    try {
      bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
    } catch {
      throw new Error(`${file.name} could not be decoded by this browser.`);
    }
    frames.push(toImageData(bitmap, bitmap.width, bitmap.height));
    bitmap.close?.();
    onProgress(++done / total, `Reading ${file.name}`);
  }

  for (const file of videos) {
    const perVideo = Math.max(2, Math.round(targetFrames / videos.length));
    const url = URL.createObjectURL(file);
    const video = document.createElement('video');
    video.preload = 'auto';
    video.muted = true;
    video.playsInline = true;
    video.src = url;
    try {
      await videoReady(video, file.name);
      const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
      if (!duration) throw new Error(`${file.name} reports no duration, so frames cannot be sampled.`);

      // Trim the ends: the first and last moments of a handheld clip are
      // usually the hand reaching for the button.
      const start = Math.min(0.08 * duration, 0.4);
      const span = Math.max(duration - 2 * start, duration * 0.5);

      for (let i = 0; i < perVideo; i++) {
        const t = start + (span * i) / Math.max(1, perVideo - 1);
        await seek(video, Math.min(t, duration - 0.02));
        frames.push(toImageData(video, video.videoWidth, video.videoHeight));
        onProgress(++done / total, `Extracting frames from ${file.name}`);
      }
    } finally {
      video.removeAttribute('src');
      video.load();
      URL.revokeObjectURL(url);
    }
  }

  if (!frames.length) throw new Error('No frames could be read from that selection.');
  return { frames, kind: videos.length ? 'video' : 'photos' };
}

function toImageData(source, width, height) {
  if (!width || !height) throw new Error('A frame arrived with no dimensions.');
  const k = Math.min(1, MAX_DIM / Math.max(width, height));
  const w = Math.max(1, Math.round(width * k));
  const h = Math.max(1, Math.round(height * k));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { alpha: false, willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

function videoReady(video, name) {
  return new Promise((resolve, reject) => {
    const fail = () => reject(new Error(`${name} could not be opened as a video by this browser.`));
    const timer = setTimeout(fail, 30000);
    video.addEventListener('loadedmetadata', () => { clearTimeout(timer); resolve(); }, { once: true });
    video.addEventListener('error', () => { clearTimeout(timer); fail(); }, { once: true });
  });
}

function seek(video, time) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Timed out seeking through the video.')), 20000);
    video.addEventListener('seeked', () => { clearTimeout(timer); resolve(); }, { once: true });
    video.addEventListener('error', () => { clearTimeout(timer); reject(new Error('The video failed while seeking.')); }, { once: true });
    video.currentTime = time;
  });
}

/** Middle frame, scaled down, as the library thumbnail. */
function thumbnailFrom(frames) {
  const frame = frames[Math.floor(frames.length / 2)];
  const canvas = document.createElement('canvas');
  const k = Math.min(1, 320 / Math.max(frame.width, frame.height));
  canvas.width = Math.round(frame.width * k);
  canvas.height = Math.round(frame.height * k);
  const src = document.createElement('canvas');
  src.width = frame.width;
  src.height = frame.height;
  src.getContext('2d').putImageData(frame, 0, 0);
  canvas.getContext('2d').drawImage(src, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.72);
}

// ------------------------------------------------------------------ conversion

async function convert(files) {
  if (state.busy) return;
  state.busy = true;
  setConvertEnabled(false);

  try {
    progress(0.02, 'Reading frames');
    const { frames, kind } = await extractFrames(files, {
      targetFrames: state.settings.targetFrames,
      onProgress: (f, label) => progress(0.02 + 0.45 * f, label),
    });

    progress(0.5, `Building gaussians from ${frames.length} frames`);
    // Yield first: reconstruction is synchronous and would otherwise freeze
    // the page before the progress bar ever paints.
    await raf();

    const chosen = pickEvenly(frames, 24);
    const { cloud } = reconstructPreview(chosen, {
      grid: state.settings.detail,
      arcDeg: state.settings.arcDeg,
    });

    progress(0.9, 'Preparing the viewer');
    await raf();

    const buffer = encodeSplatBuffer(cloud);
    const record = {
      id: `c${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`,
      name: nameFor(files, kind),
      kind,
      frames: chosen.length,
      splatCount: cloud.count,
      createdAt: Date.now(),
      thumb: thumbnailFrom(chosen),
      buffer,
    };

    progress(1, 'Done');
    await show(record);
    await dbPut(record);
    state.library = await dbAll();
    renderLibrary();
    setTab('edit');
    toast(`${cloud.count.toLocaleString()} gaussians \u00b7 saved to your library`, 'ok');
  } catch (err) {
    toast(err.message, 'error', 7000);
    progress(0, '');
  } finally {
    state.busy = false;
    setConvertEnabled(true);
  }
}

function nameFor(files, kind) {
  if (files.length === 1) return files[0].name.replace(/\.[^.]+$/, '').slice(0, 60);
  return `${kind === 'video' ? 'Video' : 'Photo'} capture ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
}

function pickEvenly(items, max) {
  if (items.length <= max) return items.slice();
  const out = [];
  for (let i = 0; i < max; i++) out.push(items[Math.round((i * (items.length - 1)) / (max - 1))]);
  return [...new Set(out)];
}

const raf = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

// ---------------------------------------------------------------------- viewer

async function show(record) {
  state.current = record;
  state.edits = defaultEdits();

  $('#empty').hidden = true;
  $('#stage-tools').hidden = false;
  $('#stage-stats').hidden = false;

  if (!state.viewer) {
    try {
      state.viewer = new SplatViewer($('#gl'));
    } catch (err) {
      $('#empty').hidden = false;
      $('#empty').innerHTML = `<div class="empty-card"><h2>This browser can't run the viewer</h2><p>${escapeHtml(err.message)}</p></div>`;
      return;
    }
  }

  // The renderer transfers the buffer to its sort worker, so hand it a copy and
  // keep the original for re-opening this capture later.
  const { count, bounds } = state.viewer.load(record.buffer.slice(0));
  state.viewer.setEdits(state.edits);
  state.viewer.camera.frame(bounds);
  state.viewer.start();

  $('#title').textContent = record.name;
  $('#subtitle').textContent = `${count.toLocaleString()} gaussians \u00b7 ${record.frames} frames \u00b7 ${record.kind === 'video' ? 'video' : 'photos'}`;
  renderEdit();
  startStats();
}

let statsTimer = null;
function startStats() {
  clearInterval(statsTimer);
  statsTimer = setInterval(() => {
    if (!state.viewer || !state.current) return;
    const s = state.viewer.stats;
    $('#stage-stats').textContent =
      `${state.current.splatCount.toLocaleString()} gaussians \u00b7 ${s.fps} fps${s.sortMs ? ` \u00b7 sort ${s.sortMs} ms` : ''}`;
  }, 500);
}

// ------------------------------------------------------------------------- UI

function progress(fraction, label) {
  $('#progress').hidden = fraction <= 0;
  $('#progress-bar').style.width = `${Math.round(fraction * 100)}%`;
  $('#progress-label').textContent = label;
  $('#progress-pct').textContent = fraction > 0 ? `${Math.round(fraction * 100)}%` : '';
}

function setConvertEnabled(on) {
  $('#convert').disabled = !on;
  $('#convert').textContent = on ? 'Choose photos or video' : 'Working\u2026';
}

function toast(message, kind = 'info', ms = 4000) {
  const node = document.createElement('div');
  node.className = `toast ${kind}`;
  node.textContent = message;
  $('#toasts').append(node);
  setTimeout(() => {
    node.style.opacity = '0';
    setTimeout(() => node.remove(), 250);
  }, ms);
}

function setTab(name) {
  state.tab = name;
  for (const button of $$('.tab')) button.classList.toggle('on', button.dataset.tab === name);
  for (const panel of $$('.panel')) panel.hidden = panel.dataset.panel !== name;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

/** Labelled range control that writes straight through to the live edits. */
function slider({ label, min, max, step, value, format, onInput }) {
  const wrap = document.createElement('label');
  wrap.className = 'ctrl';
  const head = document.createElement('span');
  head.className = 'ctrl-head';
  const name = document.createElement('span');
  name.textContent = label;
  const readout = document.createElement('span');
  readout.className = 'ctrl-value';
  readout.textContent = format(value);
  head.append(name, readout);

  const input = document.createElement('input');
  input.type = 'range';
  input.min = min; input.max = max; input.step = step; input.value = value;
  input.addEventListener('input', () => {
    const v = Number(input.value);
    readout.textContent = format(v);
    onInput(v);
  });
  wrap.append(head, input);
  return wrap;
}

function renderEdit() {
  const host = $('[data-panel="edit"]');
  host.innerHTML = '';
  if (!state.current) {
    host.innerHTML = '<p class="muted">Convert a capture first \u2014 then everything here changes the render live.</p>';
    return;
  }
  const e = state.edits;
  const push = () => state.viewer?.setEdits(e);

  const group = (title, ...nodes) => {
    const section = document.createElement('section');
    section.className = 'group';
    const h = document.createElement('h3');
    h.textContent = title;
    section.append(h, ...nodes);
    return section;
  };

  const axes = (label, target, min, max, step, format) => {
    const wrap = document.createElement('div');
    const head = document.createElement('span');
    head.className = 'axis-label';
    head.textContent = label;
    const row = document.createElement('div');
    row.className = 'axis-row';
    ['X', 'Y', 'Z'].forEach((axis, i) => {
      row.append(slider({
        label: axis, min, max, step, value: target[i], format,
        onInput: (v) => { target[i] = v; push(); },
      }));
    });
    wrap.append(head, row);
    return wrap;
  };

  host.append(
    group('Transform',
      axes('Move', e.translate, -3, 3, 0.01, (v) => v.toFixed(2)),
      axes('Rotate', e.rotate, -Math.PI, Math.PI, 0.01, (v) => `${Math.round((v * 180) / Math.PI)}\u00b0`),
      slider({
        label: 'Scale', min: 0.1, max: 4, step: 0.01, value: e.scale,
        format: (v) => `${v.toFixed(2)}\u00d7`, onInput: (v) => { e.scale = v; push(); },
      })),

    group('Appearance',
      slider({
        label: 'Gaussian size', min: 0.2, max: 3, step: 0.01, value: e.splatScale,
        format: (v) => `${v.toFixed(2)}\u00d7`, onInput: (v) => { e.splatScale = v; push(); },
      }),
      slider({
        label: 'Opacity', min: 0.1, max: 3, step: 0.01, value: e.opacity,
        format: (v) => `${v.toFixed(2)}\u00d7`, onInput: (v) => { e.opacity = v; push(); },
      }),
      slider({
        label: 'Exposure', min: -2, max: 2, step: 0.01, value: e.exposure,
        format: (v) => `${v > 0 ? '+' : ''}${v.toFixed(2)} EV`, onInput: (v) => { e.exposure = v; push(); },
      }),
      slider({
        label: 'Saturation', min: 0, max: 2, step: 0.01, value: e.saturation,
        format: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { e.saturation = v; push(); },
      })),

    group('Clean up',
      slider({
        label: 'Hide splats fainter than', min: 0, max: 0.6, step: 0.005, value: e.pruneBelowOpacity,
        format: (v) => (v === 0 ? 'off' : v.toFixed(3)),
        onInput: (v) => { e.pruneBelowOpacity = v; push(); },
      }),
      cropToggle(e, push)),
  );

  const reset = document.createElement('button');
  reset.className = 'btn';
  reset.textContent = 'Reset all edits';
  reset.addEventListener('click', () => {
    state.edits = defaultEdits();
    state.viewer?.setEdits(state.edits);
    renderEdit();
  });
  host.append(reset);
}

function cropToggle(e, push) {
  const wrap = document.createElement('div');
  const row = document.createElement('label');
  row.className = 'check';
  const box = document.createElement('input');
  box.type = 'checkbox';
  box.checked = Boolean(e.crop);
  const text = document.createElement('span');
  text.textContent = 'Crop to a box';
  row.append(box, text);
  wrap.append(row);

  const sliders = document.createElement('div');
  wrap.append(sliders);

  const paint = () => {
    sliders.innerHTML = '';
    if (!e.crop) return;
    ['X', 'Y', 'Z'].forEach((axis, i) => {
      sliders.append(slider({
        label: `${axis} min`, min: -2.5, max: 2.5, step: 0.01, value: e.crop.min[i],
        format: (v) => v.toFixed(2),
        onInput: (v) => { e.crop.min[i] = Math.min(v, e.crop.max[i]); push(); },
      }));
      sliders.append(slider({
        label: `${axis} max`, min: -2.5, max: 2.5, step: 0.01, value: e.crop.max[i],
        format: (v) => v.toFixed(2),
        onInput: (v) => { e.crop.max[i] = Math.max(v, e.crop.min[i]); push(); },
      }));
    });
  };

  box.addEventListener('change', () => {
    if (box.checked) {
      const b = state.viewer?.bounds;
      const c = b?.center || [0, 0, 0];
      const r = (b?.radius || 1) * 0.75;
      e.crop = { min: [c[0] - r, c[1] - r, c[2] - r], max: [c[0] + r, c[1] + r, c[2] + r], invert: false };
    } else {
      e.crop = null;
    }
    push();
    paint();
  });
  paint();
  return wrap;
}

function renderLibrary() {
  const host = $('[data-panel="library"]');
  host.innerHTML = '';
  if (!state.library.length) {
    host.innerHTML = '<p class="muted">Conversions you make are saved here, on this device. Nothing is uploaded.</p>';
    return;
  }
  for (const record of state.library) {
    const card = document.createElement('div');
    card.className = 'lib-item' + (state.current?.id === record.id ? ' on' : '');

    const open = document.createElement('button');
    open.className = 'lib-open';
    open.innerHTML =
      `<img src="${record.thumb}" alt="">` +
      `<span class="lib-text"><strong>${escapeHtml(record.name)}</strong>` +
      `<em>${record.splatCount.toLocaleString()} gaussians \u00b7 ${record.frames} frames</em></span>`;
    open.addEventListener('click', () => { show(record); renderLibrary(); });

    const remove = document.createElement('button');
    remove.className = 'lib-del';
    remove.title = 'Delete';
    remove.textContent = '\u00d7';
    remove.addEventListener('click', async () => {
      await dbDelete(record.id);
      state.library = await dbAll();
      if (state.current?.id === record.id) {
        state.current = null;
        state.viewer?.stop();
        $('#empty').hidden = false;
        $('#stage-tools').hidden = true;
        $('#stage-stats').hidden = true;
        $('#title').textContent = 'SplatWorks';
        $('#subtitle').textContent = 'Photos and video to gaussian splats, in your browser';
        renderEdit();
      }
      renderLibrary();
    });

    card.append(open, remove);
    host.append(card);
  }
}

// ------------------------------------------------------------------- snapshot

let downloads = null;
window.claude?.use?.('downloads').then((d) => {
  downloads = d;
  $('#snapshot').hidden = !d;
}).catch(() => {});

async function snapshot() {
  if (!state.viewer || !downloads) return;
  const dataUrl = state.viewer.snapshot();
  const blob = await (await fetch(dataUrl)).blob();
  const name = `${(state.current?.name || 'splat').replace(/[^\w.-]+/g, '_')}.png`;
  try {
    await downloads.save({ filename: name, data: blob });
  } catch (err) {
    if (err?.code !== 'declined') toast('That snapshot could not be saved.', 'error');
  }
}

// ----------------------------------------------------------------------- boot

function wire() {
  const input = $('#files');
  const drop = $('#stage');

  $('#pick-empty').addEventListener('click', () => input.click());
  input.addEventListener('change', () => {
    if (input.files.length) convert([...input.files]);
    input.value = '';
  });

  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('over');
    const files = [...e.dataTransfer.files].filter((f) => f.type.startsWith('image/') || f.type.startsWith('video/'));
    if (files.length) convert(files);
    else toast('Drop photos or a video \u2014 nothing else can be converted.', 'error');
  });

  $('#convert').addEventListener('click', () => input.click());

  for (const button of $$('.tab')) {
    button.addEventListener('click', () => setTab(button.dataset.tab));
  }

  $('#frames').addEventListener('input', (e) => {
    state.settings.targetFrames = Number(e.target.value);
    $('#frames-value').textContent = e.target.value;
  });
  $('#detail').addEventListener('input', (e) => {
    state.settings.detail = Number(e.target.value);
    $('#detail-value').textContent = e.target.value;
  });
  $('#arc').addEventListener('change', (e) => { state.settings.arcDeg = Number(e.target.value); });

  $('#fit').addEventListener('click', () => state.viewer?.camera.frame(state.viewer.bounds));
  $('#front').addEventListener('click', () => state.viewer?.camera.setView('front'));
  $('#side').addEventListener('click', () => state.viewer?.camera.setView('right'));
  $('#top').addEventListener('click', () => state.viewer?.camera.setView('top'));
  $('#spin').addEventListener('click', (e) => {
    if (!state.viewer) return;
    state.viewer.camera.autoRotate = !state.viewer.camera.autoRotate;
    e.target.classList.toggle('on', state.viewer.camera.autoRotate);
  });
  $('#snapshot').addEventListener('click', snapshot);

  $('#dismiss').addEventListener('click', () => {
    $('#note').hidden = true;
    try { localStorage.setItem('splatworks-note', '1'); } catch { /* storage may be blocked */ }
  });
}

async function boot() {
  wire();
  setTab('convert');
  renderEdit();
  try {
    if (localStorage.getItem('splatworks-note')) $('#note').hidden = true;
  } catch { /* private window: just show the note */ }
  state.library = await dbAll();
  renderLibrary();
  if (state.library.length) show(state.library[0]);
}

boot();
