import fsp from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { DB_FILE, DATA_DIR, ASSETS_DIR } from './config.js';

/**
 * The library is a small collection (hundreds of entries, each pointing at
 * files on disk), so it lives in one JSON document that is rewritten
 * atomically. Writes are serialised through a promise chain so concurrent
 * pipeline stages cannot interleave and lose an update.
 */

const EMPTY = { version: 1, assets: [] };

let cache = null;
let writeChain = Promise.resolve();

export function newId() {
  return crypto.randomBytes(9).toString('base64url');
}

export async function init() {
  await fsp.mkdir(ASSETS_DIR, { recursive: true });
  cache = await load();
  return cache;
}

async function load() {
  try {
    const raw = await fsp.readFile(DB_FILE, 'utf8');
    const doc = JSON.parse(raw);
    if (!doc || !Array.isArray(doc.assets)) return structuredClone(EMPTY);
    return doc;
  } catch (err) {
    if (err.code === 'ENOENT') return structuredClone(EMPTY);
    // A corrupt library is worth keeping a copy of rather than silently losing.
    const backup = `${DB_FILE}.corrupt-${Date.now()}`;
    try { await fsp.rename(DB_FILE, backup); } catch { /* nothing to move */ }
    console.error(`library.json was unreadable (${err.message}); moved to ${backup}`);
    return structuredClone(EMPTY);
  }
}

function db() {
  if (!cache) throw new Error('store.init() must run before the store is used');
  return cache;
}

async function flush() {
  const doc = db();
  writeChain = writeChain.then(async () => {
    await fsp.mkdir(DATA_DIR, { recursive: true });
    const tmp = `${DB_FILE}.${process.pid}.tmp`;
    await fsp.writeFile(tmp, JSON.stringify(doc, null, 2));
    await fsp.rename(tmp, DB_FILE);
  }).catch((err) => {
    console.error('failed to persist library:', err);
  });
  return writeChain;
}

export function listAssets() {
  return db().assets.map(publicView).sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export function getAsset(id) {
  return db().assets.find((a) => a.id === id) || null;
}

export function getAssetView(id) {
  const asset = getAsset(id);
  return asset ? publicView(asset) : null;
}

export async function createAsset(fields) {
  const now = new Date().toISOString();
  const asset = {
    id: fields.id || newId(),
    name: fields.name || 'Untitled capture',
    notes: '',
    tags: [],
    kind: fields.kind || 'photos',
    status: 'queued',
    stage: 'queued',
    progress: 0,
    message: 'Waiting for a worker',
    error: null,
    backend: null,
    createdAt: now,
    updatedAt: now,
    startedAt: null,
    finishedAt: null,
    source: fields.source || { files: [], frameCount: 0 },
    settings: fields.settings || {},
    result: null,
    edits: defaultEdits(),
    log: [],
    ...fields.extra,
  };
  db().assets.push(asset);
  await flush();
  return asset;
}

export async function updateAsset(id, patch) {
  const asset = getAsset(id);
  if (!asset) return null;
  Object.assign(asset, patch, { updatedAt: new Date().toISOString() });
  await flush();
  return asset;
}

export async function appendLog(id, line) {
  const asset = getAsset(id);
  if (!asset) return null;
  asset.log.push({ at: new Date().toISOString(), line: String(line).slice(0, 2000) });
  // Keep the tail; a long COLMAP run is chatty and the UI only shows recent lines.
  if (asset.log.length > 400) asset.log.splice(0, asset.log.length - 400);
  asset.updatedAt = new Date().toISOString();
  await flush();
  return asset;
}

export async function deleteAsset(id) {
  const assets = db().assets;
  const idx = assets.findIndex((a) => a.id === id);
  if (idx === -1) return false;
  assets.splice(idx, 1);
  await flush();
  await fsp.rm(assetDir(id), { recursive: true, force: true });
  return true;
}

export function assetDir(id) {
  return path.join(ASSETS_DIR, id);
}

export function assetPath(id, ...rest) {
  const dir = assetDir(id);
  const full = path.join(dir, ...rest);
  // Defence in depth: ids come from the URL, so never let one escape its folder.
  if (full !== dir && !full.startsWith(dir + path.sep)) {
    throw new Error('path escapes the asset directory');
  }
  return full;
}

export async function ensureAssetDirs(id) {
  const dirs = ['source', 'frames', 'work', 'output'];
  await Promise.all(dirs.map((d) => fsp.mkdir(assetPath(id, d), { recursive: true })));
  return assetDir(id);
}

export function defaultEdits() {
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
    background: '#0b0d12',
  };
}

/** Everything the client is allowed to see, plus derived fields it needs. */
function publicView(asset) {
  const view = { ...asset };
  view.log = asset.log.slice(-60);
  view.hasOutput = Boolean(asset.result && asset.result.splatFile);
  return view;
}

/** Test seam: drop the in-memory copy so a fresh init() re-reads from disk. */
export function _resetForTests() {
  cache = null;
  writeChain = Promise.resolve();
}
