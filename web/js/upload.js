import { el, clear, toast, formatBytes } from './ui.js';
import { extractFrames, isVideo } from './frames.js';
import { api } from './api.js';

/**
 * "New conversion" view: pick files, extract frames in the browser, upload.
 *
 * Capture guidance is part of the UI rather than buried in docs — the single
 * biggest determinant of splat quality is how the source was shot.
 */

const CAPTURE_TIPS = [
  'Walk a full circle around the subject, keeping it centred in frame.',
  'Overlap generously — each shot should share most of its view with the last.',
  'Keep lighting constant and avoid reflective or featureless surfaces.',
  'Move steadily; motion blur costs more detail than a lower frame count.',
];

export function renderUpload(host, { capabilities, onCreated }) {
  clear(host);

  const state = {
    files: [],
    busy: false,
    frames: null,
    previews: [],
    settings: { targetFrames: 32, detail: 160, arcDeg: 360 },
  };

  const fileInput = el('input', {
    type: 'file',
    multiple: true,
    accept: 'image/*,video/*',
    class: 'sr-only',
    onchange: () => setFiles([...fileInput.files]),
  });

  const dropzone = el('div', {
    class: 'dropzone',
    tabindex: '0',
    role: 'button',
    onclick: () => fileInput.click(),
    onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } },
    ondragover: (e) => { e.preventDefault(); dropzone.classList.add('over'); },
    ondragleave: () => dropzone.classList.remove('over'),
    ondrop: (e) => {
      e.preventDefault();
      dropzone.classList.remove('over');
      setFiles([...e.dataTransfer.files]);
    },
  },
    el('h3', {}, 'Drop photos or a video here'),
    el('p', {}, 'or click to browse — a 20–60 shot orbit, or a slow walk-around clip'));

  const summary = el('div', { class: 'preview-strip' });
  const nameInput = el('input', { type: 'text', placeholder: 'Name this capture' });

  const framesSlider = numberField('Frames to sample from video', 8, 120, 4, state.settings.targetFrames,
    (v) => { state.settings.targetFrames = v; });
  const detailSlider = numberField('Reconstruction detail', 48, 320, 8, state.settings.detail,
    (v) => { state.settings.detail = v; });
  const arcSelect = el('select', {
    onchange: () => { state.settings.arcDeg = Number(arcSelect.value); },
  },
    el('option', { value: '360' }, 'Full orbit (360°)'),
    el('option', { value: '180' }, 'Half orbit (180°)'),
    el('option', { value: '90' }, 'Quarter turn (90°)'),
    el('option', { value: '0' }, 'Single viewpoint'));

  const progressBar = el('i');
  const progressWrap = el('div', { class: 'job-progress', style: { display: 'none' } },
    el('div', { class: 'caption' }, el('span', { id: 'up-label' }, 'Working'), el('span', { id: 'up-pct' }, '0%')),
    el('div', { class: 'progress' }, progressBar));

  const submit = el('button', { class: 'primary', disabled: true, onclick: start },
    'Convert to splat');

  const reconstructs = Boolean(capabilities?.reconstructs);
  const advanced = !reconstructs
    ? el('div', { class: 'group' },
        el('p', { class: 'group-title' }, 'Preview backend settings'),
        detailSlider,
        el('label', { class: 'field' },
          el('span', { class: 'label' }, el('span', {}, 'How far the camera travelled')),
          arcSelect),
        el('p', { class: 'hint' },
          'The preview backend assumes an orbit around the subject. Tell it how far ' +
          'you actually moved and the reliefs line up better.'))
    : el('div', { class: 'group' },
        el('p', { class: 'group-title' }, 'Training' ),
        numberField('Training iterations', 1000, 30000, 500, 7000,
          (v) => { state.settings.iterations = v; }),
        el('p', { class: 'hint' }, 'More iterations sharpen detail and cost time.'));

  host.append(el('div', { class: 'page' },
    el('div', { class: 'page-head' },
      el('h1', {}, 'New conversion'),
      el('p', { class: 'sub' },
        'Frames are decoded in your browser, so any format your device can play works — ' +
        'no server-side video tooling required.')),

    capabilities?.backend === 'preview' ? previewNotice(capabilities) : reconstructionNotice(capabilities),

    el('div', { class: 'upload-grid' },
      el('div', { class: 'card' }, fileInput, dropzone, summary, progressWrap),
      el('div', { class: 'card' },
        el('h2', {}, 'Settings'),
        el('label', { class: 'field' },
          el('span', { class: 'label' }, el('span', {}, 'Name')),
          nameInput),
        framesSlider,
        advanced,
        el('div', { class: 'group' }, submit))),

    el('div', { class: 'card', style: { marginTop: '20px' } },
      el('h2', {}, 'Shooting a capture that converts well'),
      el('ul', { style: { margin: '0', paddingLeft: '20px', color: 'var(--text-dim)', fontSize: '13.5px' } },
        CAPTURE_TIPS.map((tip) => el('li', { style: { marginBottom: '5px' } }, tip))))));

  function setFiles(files) {
    const usable = files.filter((f) => f.type.startsWith('image/') || f.type.startsWith('video/')
      || /\.(jpe?g|png|webp|heic|heif|mp4|mov|m4v|webm)$/i.test(f.name));
    if (!usable.length) {
      toast('None of those files look like photos or video.', 'error');
      return;
    }
    state.files = usable;
    submit.disabled = false;
    const bytes = usable.reduce((n, f) => n + f.size, 0);
    const videos = usable.filter(isVideo).length;
    clear(summary).append(el('p', { class: 'hint', style: { width: '100%' } },
      `${usable.length} file${usable.length === 1 ? '' : 's'} · ${formatBytes(bytes)}`
      + (videos ? ` · ${videos} video${videos === 1 ? '' : 's'}` : '')));
    if (!nameInput.value) {
      nameInput.value = usable.length === 1
        ? usable[0].name.replace(/\.[^.]+$/, '')
        : `Capture — ${new Date().toLocaleDateString()}`;
    }
    framesSlider.style.display = videos ? '' : 'none';
  }

  function setProgress(fraction, label) {
    progressWrap.style.display = '';
    progressBar.style.width = `${Math.round(fraction * 100)}%`;
    progressWrap.querySelector('#up-label').textContent = label;
    progressWrap.querySelector('#up-pct').textContent = `${Math.round(fraction * 100)}%`;
  }

  async function start() {
    if (state.busy || !state.files.length) return;
    state.busy = true;
    submit.disabled = true;
    submit.textContent = 'Working…';

    try {
      setProgress(0.02, 'Reading frames');
      const { frames, kind, previews } = await extractFrames(state.files, {
        targetFrames: state.settings.targetFrames,
        // Feature matching lives or dies on resolution: SfM needs detail the
        // preview backend has no use for, so only pay the upload cost when the
        // backend will actually solve poses.
        maxDim: capabilities?.reconstructs ? 1600 : 640,
        onProgress: ({ done, total, label }) => {
          setProgress(0.02 + 0.48 * (total ? done / total : 0), label);
        },
      });

      clear(summary).append(previews.map((src) => el('img', { src, alt: '' })));

      const form = new FormData();
      form.append('name', nameInput.value.trim());
      form.append('kind', kind);
      form.append('settings', JSON.stringify({
        detail: state.settings.detail,
        arcDeg: state.settings.arcDeg,
        maxFrames: Math.max(8, state.settings.targetFrames),
        ...(state.settings.iterations ? { iterations: state.settings.iterations } : {}),
      }));
      frames.forEach((blob, i) => {
        form.append('frame', blob, `frame_${String(i + 1).padStart(5, '0')}.png`);
      });
      // Keep originals so the capture can be re-run later on a machine that has
      // COLMAP, without asking the user to upload everything again.
      for (const file of state.files) {
        if (file.size <= 256 * 1024 * 1024) form.append('source', file, file.name);
      }

      const { asset } = await api.upload(form, {
        onProgress: (f) => setProgress(0.5 + 0.5 * f, 'Uploading'),
      });

      setProgress(1, 'Queued');
      toast('Conversion queued.', 'success');
      previews.forEach((url) => URL.revokeObjectURL(url));
      onCreated?.(asset);
    } catch (err) {
      if (err.name !== 'AbortError') toast(err.message, 'error', 7000);
      progressWrap.style.display = 'none';
    } finally {
      state.busy = false;
      submit.disabled = false;
      submit.textContent = 'Convert to splat';
    }
  }
}

function reconstructionNotice(capabilities) {
  const bundled = capabilities?.backend === 'gaussian';
  return el('div', { class: 'notice', style: { borderColor: '#1f3a2a', background: '#101a14', color: '#bfe6cd' } },
    el('strong', { style: { color: '#7ee2a8' } }, 'Full reconstruction is available. '),
    bundled
      ? 'Camera poses are solved with COLMAP and the gaussians are optimised against your '
        + 'photos, so the result is a true 3D Gaussian Splat. Training runs on the CPU here, '
        + 'which is slow but real — expect minutes, not seconds.'
      : 'COLMAP and the configured trainer will solve camera poses and optimise the gaussians.',
    capabilities?.reasons?.length
      ? el('span', { style: { display: 'block', marginTop: '6px', opacity: '.85' } },
          capabilities.reasons.join(' '))
      : null);
}

function previewNotice(capabilities) {
  return el('div', { class: 'notice' },
    el('strong', {}, 'Running in preview mode. '),
    'This machine has no COLMAP + trainer install, so conversions build a fast proxy splat ',
    'from your frames rather than solving true camera poses. Everything else — the library, ',
    'the viewer, editing and export — works the same. ',
    capabilities.reasons?.length
      ? el('span', { style: { display: 'block', marginTop: '6px', opacity: '.85' } },
          capabilities.reasons.join(' '))
      : null);
}

function numberField(label, min, max, step, value, onInput) {
  const readout = el('span', { class: 'value' }, String(value));
  const input = el('input', {
    type: 'range', min, max, step, value,
    oninput: () => { readout.textContent = input.value; onInput(Number(input.value)); },
  });
  return el('label', { class: 'field' },
    el('span', { class: 'label' }, el('span', {}, label), readout),
    input);
}
