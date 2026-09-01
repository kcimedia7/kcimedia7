import { el, clear, relativeTime, formatCount } from './ui.js';
import { api, assetVersion } from './api.js';

/** How each capture kind is described in the library and detail views. */
const SOURCE_LABEL = { photos: 'Photos', video: 'Video', pano: '360 panorama' };

/**
 * The library: every conversion ever run, searchable, with live status for the
 * ones still working.
 */

const SORTS = {
  newest: (a, b) => b.createdAt.localeCompare(a.createdAt),
  oldest: (a, b) => a.createdAt.localeCompare(b.createdAt),
  name: (a, b) => a.name.localeCompare(b.name),
  splats: (a, b) => (b.result?.splatCount || 0) - (a.result?.splatCount || 0),
};

export function renderLibrary(host, { assets, onOpen }) {
  clear(host);

  const state = { query: '', sort: 'newest', filter: 'all' };

  const search = el('input', {
    type: 'search',
    placeholder: 'Search names, tags and notes',
    oninput: () => { state.query = search.value.trim().toLowerCase(); paint(); },
  });

  const sortSelect = el('select', {
    style: { width: 'auto' },
    onchange: () => { state.sort = sortSelect.value; paint(); },
  },
    el('option', { value: 'newest' }, 'Newest first'),
    el('option', { value: 'oldest' }, 'Oldest first'),
    el('option', { value: 'name' }, 'By name'),
    el('option', { value: 'splats' }, 'Most gaussians'));

  const filterSelect = el('select', {
    style: { width: 'auto' },
    onchange: () => { state.filter = filterSelect.value; paint(); },
  },
    el('option', { value: 'all' }, 'All conversions'),
    el('option', { value: 'ready' }, 'Ready'),
    el('option', { value: 'working' }, 'In progress'),
    el('option', { value: 'failed' }, 'Failed'));

  const grid = el('div', { class: 'grid' });
  const countLabel = el('span', { class: 'hint', style: { margin: '0' } });

  host.append(el('div', { class: 'page' },
    el('div', { class: 'page-head' },
      el('h1', {}, 'Library'),
      el('p', { class: 'sub' },
        'Every conversion is kept here with its source frames, so you can re-open it, ' +
        'adjust it, or re-run it with different settings at any time.')),
    el('div', { class: 'library-toolbar' },
      search, filterSelect, sortSelect,
      el('span', { class: 'spacer' }),
      countLabel,
      el('a', { class: 'button primary', href: '#/new' }, 'New conversion')),
    grid));

  function paint() {
    const filtered = assets
      .filter((a) => matchesFilter(a, state.filter))
      .filter((a) => matchesQuery(a, state.query))
      .sort(SORTS[state.sort]);

    countLabel.textContent = `${filtered.length} of ${assets.length}`;
    clear(grid);

    if (!filtered.length) {
      grid.style.display = 'block';
      grid.append(el('div', { class: 'empty' },
        assets.length
          ? 'Nothing matches those filters.'
          : el('div', {},
              el('p', { style: { margin: '0 0 14px', fontSize: '15px' } },
                'No conversions yet.'),
              el('a', { class: 'button primary', href: '#/new' }, 'Convert your first capture'))));
      return;
    }
    grid.style.display = '';
    for (const asset of filtered) grid.append(tile(asset, onOpen));
  }

  paint();
  return { update(next) { assets = next; paint(); } };
}

function matchesFilter(asset, filter) {
  if (filter === 'all') return true;
  if (filter === 'ready') return asset.status === 'ready';
  if (filter === 'working') return asset.status === 'running' || asset.status === 'queued';
  if (filter === 'failed') return asset.status === 'failed' || asset.status === 'cancelled';
  return true;
}

function matchesQuery(asset, query) {
  if (!query) return true;
  const hay = [asset.name, asset.notes, ...(asset.tags || []), asset.backend || '']
    .join(' ').toLowerCase();
  return hay.includes(query);
}

function tile(asset, onOpen) {
  const thumb = el('div', { class: 'tile-thumb' });
  if (asset.result?.thumbnail) {
    thumb.style.backgroundImage = `url(${api.thumbnailUrl(asset.id, assetVersion(asset))})`;
  } else if (asset.status === 'running' || asset.status === 'queued') {
    thumb.append(el('div', { style: { display: 'grid', gap: '8px', justifyItems: 'center' } },
      el('span', { class: 'spinner' }),
      el('span', {}, asset.message || 'Working')));
  } else {
    thumb.append(el('span', {}, asset.status === 'failed' ? 'Conversion failed' : 'No preview'));
  }

  const bar = el('i', { style: { width: `${Math.round((asset.progress || 0) * 100)}%` } });

  return el('a', {
    class: 'tile',
    href: `#/a/${asset.id}`,
    onclick: (e) => {
      if (e.metaKey || e.ctrlKey || e.shiftKey) return;
      e.preventDefault();
      onOpen(asset.id);
    },
  },
    thumb,
    el('div', { class: 'tile-body' },
      el('div', { class: 'tile-name' }, asset.name),
      el('div', { class: 'tile-meta' },
        el('span', { class: `badge ${asset.status}` }, statusLabel(asset)),
        el('span', {}, SOURCE_LABEL[asset.kind] || 'Photos'),
        asset.result?.splatCount
          ? el('span', {}, `${formatCount(asset.result.splatCount)} gaussians`)
          : null,
        el('span', {}, relativeTime(asset.createdAt))),
      asset.tags?.length
        ? el('div', { class: 'tags' }, asset.tags.slice(0, 4).map((t) => el('span', { class: 'tag' }, t)))
        : null,
      (asset.status === 'running' || asset.status === 'queued')
        ? el('div', { class: 'progress', style: { marginTop: 'auto' } }, bar)
        : null));
}

function statusLabel(asset) {
  switch (asset.status) {
    case 'ready': return 'Ready';
    case 'running': return 'Converting';
    case 'queued': return 'Queued';
    case 'failed': return 'Failed';
    case 'cancelled': return 'Cancelled';
    default: return asset.status;
  }
}
