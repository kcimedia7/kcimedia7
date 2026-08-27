import { el, clear, toast } from './ui.js';
import { api, subscribe } from './api.js';
import { renderLibrary } from './library.js';
import { renderUpload } from './upload.js';
import { renderDetail } from './detail.js';

/**
 * App shell: hash routing, a single cached copy of the library, and one SSE
 * connection that keeps whatever view is mounted up to date.
 */

const view = document.getElementById('view');

const state = {
  assets: [],
  capabilities: null,
  mounted: null,      // { route, id, handle }
};

async function boot() {
  window.addEventListener('hashchange', route);

  try {
    const health = await api.health();
    state.capabilities = health.capabilities;
    paintBackendBadge(health.capabilities);
  } catch {
    paintBackendBadge(null);
  }

  try {
    const { assets } = await api.list();
    state.assets = assets;
  } catch (err) {
    toast(`Could not load the library: ${err.message}`, 'error', 8000);
  }

  subscribe(onServerUpdate);
  route();
}

function paintBackendBadge(caps) {
  const badge = document.getElementById('backend-badge');
  if (!caps) {
    badge.querySelector('.label').textContent = 'server unreachable';
    return;
  }
  badge.dataset.backend = caps.backend;
  const LABELS = {
    colmap: 'COLMAP + trainer',
    gaussian: 'full reconstruction',
    preview: 'preview backend',
  };
  const TITLES = {
    colmap: 'Full reconstruction: COLMAP solves camera poses and the configured trainer optimises the gaussians.',
    gaussian: 'Full reconstruction: pycolmap solves camera poses and gaussians are optimised against your photos on the CPU.',
    preview: 'Preview reconstruction \u2014 a fast proxy, not structure-from-motion.',
  };
  badge.querySelector('.label').textContent = LABELS[caps.backend] || caps.backend;
  badge.title = `${TITLES[caps.backend] || ''} ${(caps.reasons || []).join(' ')}`.trim();
}

/** An asset changed on the server: refresh the cache and the mounted view. */
async function onServerUpdate(event) {
  let fresh = null;
  try {
    const res = await api.get(event.assetId);
    fresh = res.asset;
  } catch {
    // Deleted while we were asking — drop it from the cache.
    state.assets = state.assets.filter((a) => a.id !== event.assetId);
    if (state.mounted?.route === 'detail' && state.mounted.id === event.assetId) goto('#/');
    else state.mounted?.handle?.update?.(state.assets);
    return;
  }

  const idx = state.assets.findIndex((a) => a.id === fresh.id);
  if (idx === -1) state.assets.unshift(fresh);
  else state.assets[idx] = fresh;

  if (state.mounted?.route === 'library') {
    state.mounted.handle.update(state.assets);
  } else if (state.mounted?.route === 'detail' && state.mounted.id === fresh.id) {
    state.mounted.handle.update(fresh);
  }

  if (event.type === 'done') toast(`"${fresh.name}" is ready.`, 'success');
  if (event.type === 'failed') toast(`"${fresh.name}" failed to convert.`, 'error', 7000);
}

function goto(hash) {
  if (location.hash === hash) route();
  else location.hash = hash;
}

function route() {
  const hash = location.hash || '#/';
  state.mounted?.handle?.dispose?.();
  state.mounted = null;
  clear(view);

  const detail = /^#\/a\/(.+)$/.exec(hash);
  if (detail) return mountDetail(decodeURIComponent(detail[1]));
  if (hash.startsWith('#/new')) return mountUpload();
  return mountLibrary();
}

function setNav(name) {
  for (const link of document.querySelectorAll('.topnav a')) {
    link.classList.toggle('active', link.dataset.nav === name);
  }
}

function mountLibrary() {
  setNav('library');
  const handle = renderLibrary(view, {
    assets: state.assets,
    onOpen: (id) => goto(`#/a/${id}`),
  });
  state.mounted = { route: 'library', handle };
}

function mountUpload() {
  setNav('new');
  renderUpload(view, {
    capabilities: state.capabilities,
    onCreated: (asset) => {
      state.assets.unshift(asset);
      goto(`#/a/${asset.id}`);
    },
  });
  state.mounted = { route: 'upload' };
}

async function mountDetail(id) {
  setNav(null);
  view.append(el('div', { class: 'page' },
    el('p', { class: 'sub' }, el('span', { class: 'spinner' }), ' Loading…')));

  let asset = state.assets.find((a) => a.id === id);
  try {
    const res = await api.get(id);
    asset = res.asset;
    const idx = state.assets.findIndex((a) => a.id === id);
    if (idx === -1) state.assets.unshift(asset);
    else state.assets[idx] = asset;
  } catch (err) {
    clear(view).append(el('div', { class: 'page' },
      el('div', { class: 'notice error' }, `That conversion could not be opened: ${err.message}`),
      el('a', { class: 'button', href: '#/' }, 'Back to the library')));
    return;
  }

  clear(view);
  const handle = renderDetail(view, {
    asset,
    onDeleted: (deletedId) => {
      state.assets = state.assets.filter((a) => a.id !== deletedId);
      goto('#/');
    },
    onChanged: (updated) => {
      const idx = state.assets.findIndex((a) => a.id === updated.id);
      if (idx !== -1) state.assets[idx] = updated;
    },
  });
  state.mounted = { route: 'detail', id, handle };
}

boot();
