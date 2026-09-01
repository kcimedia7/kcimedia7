import {
  el, clear, toast, slider, debounce, formatBytes, formatCount, relativeTime,
  formatDuration, confirmAction,
} from './ui.js';
import { api, assetVersion } from './api.js';
import { SplatViewer } from './viewer/renderer.js';

/** How each capture kind is described in the library and detail views. */
const SOURCE_LABEL = { photos: 'Photos', video: 'Video', pano: '360 panorama' };

/**
 * Detail view: the splat on the left, everything you can do to it on the right.
 *
 * Edits are applied in the shader immediately and saved to the library on a
 * debounce, so dragging a slider is instant and nothing needs an explicit save.
 */

export function renderDetail(host, { asset, onDeleted, onChanged }) {
  clear(host);

  let current = asset;
  let viewer = null;
  let disposed = false;

  const canvas = el('canvas');
  const overlay = el('div', { class: 'stage-overlay' });
  const stats = el('div', { class: 'stage-stats' });
  const tools = el('div', { class: 'stage-tools' });
  // Nobody guesses that keyboard flight exists. It fades once the pointer is
  // over the stage, so it explains itself and then stays out of the way.
  const help = el('div', { class: 'stage-help' },
    el('kbd', {}, 'W'), el('kbd', {}, 'A'), el('kbd', {}, 'S'), el('kbd', {}, 'D'),
    el('span', {}, 'move'),
    el('kbd', {}, 'Q'), el('kbd', {}, 'E'), el('span', {}, 'up / down'),
    el('kbd', {}, 'Shift'), el('span', {}, 'faster'),
    el('span', {}, '· drag to orbit'));
  const stage = el('div', { class: 'stage' }, canvas, overlay, stats, tools, help);

  const tabButtons = {};
  const tabBody = el('div', { class: 'tab-body' });
  let activeTab = 'details';

  const title = el('input', {
    class: 'title',
    type: 'text',
    value: current.name,
    onchange: () => save({ name: title.value.trim() || 'Untitled capture' }),
  });

  const sidebar = el('aside', { class: 'sidebar' },
    el('div', { class: 'sidebar-head' },
      title,
      el('div', { class: 'tile-meta', style: { marginTop: '6px' } },
        el('span', { class: `badge ${current.status}`, id: 'status-badge' }, current.status),
        el('span', { id: 'status-note' }, current.message || ''))),
    el('div', { class: 'tabs' },
      tab('details', 'Details'),
      tab('edit', 'Edit'),
      tab('log', 'Log')),
    tabBody);

  host.append(el('div', { class: 'detail' }, stage, sidebar));

  paintTab();
  bootViewer();

  function tab(id, label) {
    const button = el('button', {
      class: id === activeTab ? 'active' : '',
      onclick: () => {
        activeTab = id;
        for (const [key, node] of Object.entries(tabButtons)) {
          node.classList.toggle('active', key === id);
        }
        paintTab();
      },
    }, label);
    tabButtons[id] = button;
    return button;
  }

  // ---------- viewer ----------

  async function bootViewer() {
    if (current.status !== 'ready' || !current.result?.splatFile) {
      showOverlayForStatus();
      return;
    }
    setOverlay(el('div', { class: 'box' },
      el('span', { class: 'spinner' }),
      el('p', { style: { margin: '10px 0 0' } }, 'Loading splat…')));

    try {
      viewer = new SplatViewer(canvas);
    } catch (err) {
      setOverlay(el('div', { class: 'box' },
        el('h2', {}, 'This browser cannot show the splat'),
        el('p', { class: 'sub' }, err.message),
        el('a', { class: 'button', href: api.exportUrl(current.id, 'ply') }, 'Download the .ply instead')));
      return;
    }

    try {
      // The version makes a re-run fetch the new model rather than the
      // hard-cached previous one.
      const res = await fetch(api.splatUrl(current.id, assetVersion(current)));
      if (!res.ok) throw new Error(`Could not load the splat (${res.status})`);
      const buffer = await res.arrayBuffer();
      if (disposed) return;

      const { count, bounds } = viewer.load(buffer);
      viewer.setEdits(current.edits);
      viewer.camera.frame(bounds);
      viewer.start();
      setOverlay(null);
      buildTools();

      const tick = () => {
        if (disposed) return;
        stats.textContent = `${formatCount(count)} gaussians · ${viewer.stats.fps} fps`
          + (viewer.stats.sortMs ? ` · sort ${viewer.stats.sortMs}ms` : '');
        setTimeout(tick, 500);
      };
      tick();
    } catch (err) {
      setOverlay(el('div', { class: 'box' },
        el('h2', {}, 'Could not load this splat'),
        el('p', { class: 'sub' }, err.message)));
    }
  }

  function buildTools() {
    clear(tools).append(
      el('button', { onclick: () => viewer.camera.frame(viewer.bounds) }, 'Fit'),
      el('button', { onclick: () => viewer.camera.setView('front') }, 'Front'),
      el('button', { onclick: () => viewer.camera.setView('right') }, 'Side'),
      el('button', { onclick: () => viewer.camera.setView('top') }, 'Top'),
      el('button', {
        onclick: (e) => {
          viewer.camera.autoRotate = !viewer.camera.autoRotate;
          e.target.textContent = viewer.camera.autoRotate ? 'Stop spin' : 'Spin';
        },
      }, 'Spin'),
      el('button', { onclick: snapshot }, 'Snapshot'));
  }

  function snapshot() {
    const url = viewer.snapshot();
    const a = el('a', { href: url, download: `${current.name.replace(/[^\w.-]+/g, '_')}.png` });
    document.body.append(a);
    a.click();
    a.remove();
  }

  function setOverlay(node) {
    clear(overlay);
    if (node) overlay.append(node);
    overlay.style.display = node ? '' : 'none';
  }

  function showOverlayForStatus() {
    if (current.status === 'failed') {
      setOverlay(el('div', { class: 'box' },
        el('h2', {}, 'Conversion failed'),
        el('p', { class: 'sub' }, current.error || 'No further detail was recorded.'),
        el('div', { class: 'button-row', style: { marginTop: '14px' } },
          el('button', { class: 'primary', onclick: reconvert }, 'Try again'))));
    } else if (current.status === 'cancelled') {
      setOverlay(el('div', { class: 'box' },
        el('h2', {}, 'Conversion cancelled'),
        el('div', { class: 'button-row', style: { marginTop: '14px' } },
          el('button', { class: 'primary', onclick: reconvert }, 'Run it again'))));
    } else {
      const bar = el('i', { style: { width: `${Math.round((current.progress || 0) * 100)}%` } });
      setOverlay(el('div', { class: 'box' },
        el('span', { class: 'spinner' }),
        el('h2', { style: { margin: '12px 0 4px' } }, current.message || 'Converting'),
        el('p', { class: 'sub' }, 'You can leave this page — it keeps running.'),
        el('div', { class: 'progress', style: { marginTop: '14px' } }, bar)));
    }
  }

  // ---------- tabs ----------

  function paintTab() {
    clear(tabBody);
    if (activeTab === 'details') tabBody.append(detailsTab());
    else if (activeTab === 'edit') tabBody.append(editTab());
    else tabBody.append(logTab());
  }

  function detailsTab() {
    const r = current.result;
    const rows = el('dl', { class: 'rows' },
      row('Status', current.status),
      row('Source', SOURCE_LABEL[current.kind] || 'Photos'),
      row('Frames used', formatCount(current.source?.frameCount)),
      row('Backend', current.backend || '—'),
      r ? row('Gaussians', formatCount(r.splatCount)) : null,
      r ? row('PLY size', formatBytes(r.plyBytes)) : null,
      r?.stats?.iterations ? row('Iterations', formatCount(r.stats.iterations)) : null,
      row('Created', relativeTime(current.createdAt)),
      row('Conversion time', formatDuration(current.startedAt, current.finishedAt)));

    const notes = el('textarea', {
      placeholder: 'Notes about this capture…',
      value: current.notes || '',
      onchange: () => save({ notes: notes.value }),
    });

    const tagsInput = el('input', {
      type: 'text',
      placeholder: 'tags, comma separated',
      value: (current.tags || []).join(', '),
      onchange: () => save({ tags: tagsInput.value.split(',').map((t) => t.trim()).filter(Boolean) }),
    });

    return el('div', {},
      current.error ? el('div', { class: 'notice error' }, current.error) : null,
      rows,

      el('div', { class: 'group' },
        el('p', { class: 'group-title' }, 'Export'),
        el('div', { class: 'button-row' },
          el('a', {
            class: 'button', href: api.exportUrl(current.id, 'ply'),
            title: 'Standard 3DGS point cloud, with your edits applied',
          }, 'Download .ply'),
          el('a', {
            class: 'button', href: api.exportUrl(current.id, 'splat'),
            title: 'Compact format read by most web viewers',
          }, 'Download .splat')),
        el('p', { class: 'hint' },
          'Exports include your edits. ',
          el('a', { href: api.exportUrl(current.id, 'ply', true) }, 'Download the unedited original'),
          '.')),

      el('div', { class: 'group' },
        el('p', { class: 'group-title' }, 'Notes'),
        notes,
        el('label', { class: 'field', style: { marginTop: '12px' } },
          el('span', { class: 'label' }, el('span', {}, 'Tags')),
          tagsInput)),

      el('div', { class: 'group' },
        el('p', { class: 'group-title' }, 'Manage'),
        el('div', { class: 'button-row' },
          el('button', { onclick: reconvert }, 'Re-run conversion'),
          el('button', { onclick: duplicate }, 'Duplicate'),
          (current.status === 'running' || current.status === 'queued')
            ? el('button', { onclick: cancel }, 'Cancel') : null),
        el('div', { class: 'button-row', style: { marginTop: '8px' } },
          el('button', { class: 'danger', onclick: remove }, 'Delete conversion'))));
  }

  function editTab() {
    if (current.status !== 'ready') {
      return el('p', { class: 'sub' }, 'Editing unlocks once the conversion finishes.');
    }
    const e = { ...current.edits };

    const push = debounce(() => save({ edits: e }, { quiet: true }), 600);
    const live = (mutate) => {
      mutate();
      viewer?.setEdits(e);
      push();
    };

    const transform = el('div', { class: 'group' },
      el('p', { class: 'group-title' }, 'Transform'),
      axisTriplet('Move', e.translate, -3, 3, 0.01, (i, v) => live(() => { e.translate[i] = v; })),
      axisTriplet('Rotate', e.rotate, -Math.PI, Math.PI, 0.01,
        (i, v) => live(() => { e.rotate[i] = v; }), (v) => `${Math.round((v * 180) / Math.PI)}°`),
      slider({
        label: 'Scale', min: 0.1, max: 4, step: 0.01, value: e.scale,
        format: (v) => `${v.toFixed(2)}×`,
        onInput: (v) => live(() => { e.scale = v; }),
      }));

    const appearance = el('div', { class: 'group' },
      el('p', { class: 'group-title' }, 'Appearance'),
      slider({
        label: 'Gaussian size', min: 0.2, max: 3, step: 0.01, value: e.splatScale,
        format: (v) => `${v.toFixed(2)}×`,
        onInput: (v) => live(() => { e.splatScale = v; }),
      }),
      slider({
        label: 'Opacity', min: 0.1, max: 3, step: 0.01, value: e.opacity,
        format: (v) => `${v.toFixed(2)}×`,
        onInput: (v) => live(() => { e.opacity = v; }),
      }),
      slider({
        label: 'Exposure', min: -2, max: 2, step: 0.01, value: e.exposure,
        format: (v) => `${v > 0 ? '+' : ''}${v.toFixed(2)} EV`,
        onInput: (v) => live(() => { e.exposure = v; }),
      }),
      slider({
        label: 'Saturation', min: 0, max: 2, step: 0.01, value: e.saturation,
        format: (v) => `${Math.round(v * 100)}%`,
        onInput: (v) => live(() => { e.saturation = v; }),
      }),
      el('label', { class: 'field' },
        el('span', { class: 'label' }, el('span', {}, 'Background')),
        el('input', {
          type: 'color', value: e.background, style: { height: '34px', padding: '2px' },
          oninput: (ev) => live(() => { e.background = ev.target.value; }),
        })));

    const cropToggle = el('input', {
      type: 'checkbox',
      checked: Boolean(e.crop),
      onchange: () => live(() => {
        e.crop = cropToggle.checked ? cropFromBounds(viewer?.bounds) : null;
        paintTab();
      }),
    });

    const cleanup = el('div', { class: 'group' },
      el('p', { class: 'group-title' }, 'Clean up'),
      slider({
        label: 'Hide splats fainter than', min: 0, max: 0.6, step: 0.005, value: e.pruneBelowOpacity,
        format: (v) => (v === 0 ? 'off' : v.toFixed(3)),
        onInput: (v) => live(() => { e.pruneBelowOpacity = v; }),
      }),
      el('label', { class: 'field', style: { display: 'flex', gap: '9px', alignItems: 'center' } },
        cropToggle, el('span', {}, 'Crop to a box')),
      e.crop ? cropControls(e, live) : null,
      el('p', { class: 'hint' },
        'Cropping and hiding are non-destructive — they change what is rendered and exported, ' +
        'never the stored reconstruction.'));

    return el('div', {},
      transform, appearance, cleanup,
      el('div', { class: 'group' },
        el('div', { class: 'button-row' },
          el('button', {
            onclick: async () => {
              push.cancel();
              await save({ resetEdits: true });
              paintTab();
              viewer?.setEdits(current.edits);
            },
          }, 'Reset all edits'),
          el('button', {
            class: 'primary',
            onclick: () => { push.flush(); toast('Edits saved.', 'success'); },
          }, 'Save now'))));
  }

  function cropControls(e, live) {
    const axes = ['X', 'Y', 'Z'];
    const nodes = axes.flatMap((axis, i) => [
      slider({
        label: `${axis} min`, min: -2.5, max: 2.5, step: 0.01, value: e.crop.min[i],
        format: (v) => v.toFixed(2),
        onInput: (v) => live(() => { e.crop.min[i] = Math.min(v, e.crop.max[i]); }),
      }),
      slider({
        label: `${axis} max`, min: -2.5, max: 2.5, step: 0.01, value: e.crop.max[i],
        format: (v) => v.toFixed(2),
        onInput: (v) => live(() => { e.crop.max[i] = Math.max(v, e.crop.min[i]); }),
      }),
    ]);

    const showClipped = el('input', {
      type: 'checkbox',
      onchange: () => { if (viewer) { viewer.showCropped = showClipped.checked; } },
    });

    return el('div', {},
      ...nodes,
      el('label', { class: 'field', style: { display: 'flex', gap: '9px', alignItems: 'center' } },
        showClipped, el('span', {}, 'Ghost the clipped splats')),
      el('label', { class: 'field', style: { display: 'flex', gap: '9px', alignItems: 'center' } },
        el('input', {
          type: 'checkbox',
          checked: Boolean(e.crop.invert),
          onchange: (ev) => live(() => { e.crop.invert = ev.target.checked; }),
        }),
        el('span', {}, 'Invert (keep what is outside)')));
  }

  function logTab() {
    const lines = current.log || [];
    if (!lines.length) return el('p', { class: 'sub' }, 'Nothing has been logged for this conversion.');
    const body = el('div', { class: 'log' },
      lines.map(({ at, line }) => el('div', { class: /error|failed/i.test(line) ? 'err' : '' },
        `${new Date(at).toLocaleTimeString()}  ${line}`)));
    setTimeout(() => { body.scrollTop = body.scrollHeight; }, 0);
    return body;
  }

  function row(label, value) {
    return el('div', { class: 'row' }, el('dt', {}, label), el('dd', {}, value ?? '—'));
  }

  // ---------- actions ----------

  async function save(patch, { quiet = false } = {}) {
    try {
      const { asset: updated } = await api.patch(current.id, patch);
      current = updated;
      onChanged?.(updated);
      if (!quiet) toast('Saved.', 'success', 1800);
    } catch (err) {
      toast(err.message, 'error');
    }
  }

  async function reconvert() {
    try {
      await api.reconvert(current.id, current.settings || {});
      toast('Queued for another run.', 'success');
    } catch (err) {
      toast(err.message, 'error');
    }
  }

  async function duplicate() {
    try {
      const { asset: copy } = await api.duplicate(current.id);
      toast('Duplicated — edit the copy freely.', 'success');
      location.hash = `#/a/${copy.id}`;
    } catch (err) {
      toast(err.message, 'error');
    }
  }

  async function cancel() {
    try {
      await api.cancel(current.id);
      toast('Cancelling…');
    } catch (err) {
      toast(err.message, 'error');
    }
  }

  async function remove() {
    if (!confirmAction(`Delete "${current.name}" and its frames? This cannot be undone.`)) return;
    try {
      await api.remove(current.id);
      toast('Deleted.', 'success');
      onDeleted?.(current.id);
    } catch (err) {
      toast(err.message, 'error');
    }
  }

  // ---------- live updates ----------

  return {
    /** Called when a server event says this asset changed. */
    update(next) {
      const wasReady = current.status === 'ready';
      current = next;
      const badge = sidebar.querySelector('#status-badge');
      const note = sidebar.querySelector('#status-note');
      if (badge) { badge.className = `badge ${next.status}`; badge.textContent = next.status; }
      if (note) note.textContent = next.message || '';

      if (!wasReady && next.status === 'ready') {
        // The conversion just finished; bring the viewer up.
        bootViewer();
        paintTab();
      } else if (current.status !== 'ready') {
        showOverlayForStatus();
        if (activeTab === 'log') paintTab();
      }
    },
    dispose() {
      disposed = true;
      viewer?.dispose();
    },
  };
}

function axisTriplet(label, values, min, max, step, onChange, format) {
  return el('div', {},
    el('span', { class: 'label', style: { fontSize: '12.5px', color: 'var(--text-dim)' } }, label),
    el('div', { class: 'triplet', style: { marginTop: '6px' } },
      ['X', 'Y', 'Z'].map((axis, i) => slider({
        label: axis, min, max, step, value: values[i],
        format: format || ((v) => v.toFixed(2)),
        onInput: (v) => onChange(i, v),
      }))));
}

function cropFromBounds(bounds) {
  const c = bounds?.center || [0, 0, 0];
  const r = (bounds?.radius || 1) * 0.75;
  return {
    min: [c[0] - r, c[1] - r, c[2] - r],
    max: [c[0] + r, c[1] + r, c[2] + r],
    invert: false,
  };
}
