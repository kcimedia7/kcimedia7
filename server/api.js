import fsp from 'node:fs/promises';
import fs from 'node:fs';
import path from 'node:path';
import * as store from './store.js';
import * as jobs from './jobs.js';
import { parseMultipart } from './http/multipart.js';
import {
  Router, sendJson, sendError, readJsonBody, sendFile, sanitiseFilename,
} from './http/router.js';
import { detectCapabilities, peekCapabilities } from './pipeline/backends.js';
import { stagePlan } from './pipeline/index.js';
import { decodePly, encodePly } from './pipeline/ply.js';
import { decodeSplatBuffer, encodeSplatBuffer } from './pipeline/splat.js';
import { applyEdits, normaliseEdits, isIdentity } from './edits.js';
import { MAX_UPLOAD_BYTES } from './config.js';

const PNG_MAGIC = Buffer.from([0x89, 0x50, 0x4e, 0x47]);

export function createApiRouter() {
  const r = new Router();

  r.get('/api/health', (req, res) => {
    // Answers straight away so a health check never waits on backend detection;
    // `ready` tells the client whether to ask again.
    const caps = peekCapabilities();
    sendJson(res, 200, {
      ok: true,
      ready: Boolean(caps),
      capabilities: caps,
      queue: jobs.queueState(),
      plans: {
        preview: stagePlan('preview'),
        colmap: stagePlan('colmap'),
        gaussian: stagePlan('gaussian'),
      },
    });
  });

  r.get('/api/assets', (req, res) => {
    sendJson(res, 200, { assets: store.listAssets() });
  });

  r.post('/api/assets', handleUpload);

  r.get('/api/assets/:id', (req, res, { params }) => {
    const asset = store.getAssetView(params.id);
    if (!asset) return sendError(res, 404, 'no such conversion');
    sendJson(res, 200, { asset, active: jobs.isActive(params.id) });
  });

  r.patch('/api/assets/:id', async (req, res, { params }) => {
    const asset = store.getAsset(params.id);
    if (!asset) return sendError(res, 404, 'no such conversion');
    const body = await readJsonBody(req);
    const patch = {};

    if (typeof body.name === 'string') patch.name = body.name.trim().slice(0, 160) || 'Untitled capture';
    if (typeof body.notes === 'string') patch.notes = body.notes.slice(0, 4000);
    if (Array.isArray(body.tags)) {
      patch.tags = [...new Set(body.tags
        .filter((t) => typeof t === 'string')
        .map((t) => t.trim().toLowerCase().slice(0, 32))
        .filter(Boolean))].slice(0, 24);
    }
    if (body.edits && typeof body.edits === 'object') {
      patch.edits = normaliseEdits({ ...asset.edits, ...body.edits });
    }
    if (body.resetEdits) patch.edits = store.defaultEdits();

    const updated = await store.updateAsset(params.id, patch);
    sendJson(res, 200, { asset: store.getAssetView(updated.id) });
  });

  r.delete('/api/assets/:id', async (req, res, { params }) => {
    jobs.cancel(params.id);
    const ok = await store.deleteAsset(params.id);
    if (!ok) return sendError(res, 404, 'no such conversion');
    sendJson(res, 200, { deleted: params.id });
  });

  r.post('/api/assets/:id/cancel', (req, res, { params }) => {
    const cancelled = jobs.cancel(params.id);
    sendJson(res, 200, { cancelled });
  });

  r.post('/api/assets/:id/reconvert', async (req, res, { params }) => {
    const asset = store.getAsset(params.id);
    if (!asset) return sendError(res, 404, 'no such conversion');
    if (jobs.isActive(params.id)) return sendError(res, 409, 'this conversion is already running');

    const body = await readJsonBody(req);
    const settings = { ...asset.settings, ...sanitiseSettings(body.settings || {}) };
    await store.updateAsset(params.id, {
      settings,
      status: 'queued',
      stage: 'queued',
      progress: 0,
      message: 'Waiting for a worker',
      error: null,
      result: null,
      log: [],
    });
    jobs.enqueue(params.id);
    sendJson(res, 202, { asset: store.getAssetView(params.id) });
  });

  /** Copy an entry so a variant can be edited without losing the original. */
  r.post('/api/assets/:id/duplicate', async (req, res, { params }) => {
    const asset = store.getAsset(params.id);
    if (!asset) return sendError(res, 404, 'no such conversion');

    const copy = await store.createAsset({
      name: `${asset.name} (copy)`,
      kind: asset.kind,
      source: asset.source,
      settings: asset.settings,
      extra: {
        notes: asset.notes,
        tags: [...asset.tags],
        edits: structuredClone(asset.edits),
        status: asset.status,
        stage: asset.stage,
        progress: asset.progress,
        message: asset.message,
        backend: asset.backend,
        result: asset.result ? structuredClone(asset.result) : null,
        finishedAt: asset.finishedAt,
      },
    });
    await store.ensureAssetDirs(copy.id);
    await copyTree(store.assetPath(asset.id, 'frames'), store.assetPath(copy.id, 'frames'));
    await copyTree(store.assetPath(asset.id, 'output'), store.assetPath(copy.id, 'output'));
    sendJson(res, 201, { asset: store.getAssetView(copy.id) });
  });

  r.get('/api/assets/:id/splat', (req, res, { params }) => {
    const asset = store.getAsset(params.id);
    if (!asset?.result?.splatFile) return sendError(res, 404, 'this conversion has no splat yet');
    sendFile(req, res, store.assetPath(params.id, 'output', asset.result.splatFile), {
      cache: 'private, max-age=31536000, immutable',
    });
  });

  r.get('/api/assets/:id/thumbnail', (req, res, { params }) => {
    const asset = store.getAsset(params.id);
    if (!asset?.result?.thumbnail) return sendError(res, 404, 'no thumbnail');
    sendFile(req, res, store.assetPath(params.id, 'output', asset.result.thumbnail), {
      cache: 'private, max-age=86400',
    });
  });

  /**
   * Export with edits baked in. `?raw=1` returns the untouched training output.
   */
  r.get('/api/assets/:id/export.ply', async (req, res, { params, query }) => {
    const asset = store.getAsset(params.id);
    if (!asset?.result?.plyFile) return sendError(res, 404, 'this conversion has no splat yet');
    const plyPath = store.assetPath(params.id, 'output', asset.result.plyFile);
    const filename = `${sanitiseFilename(asset.name)}.ply`;

    if (query.get('raw') === '1' || isIdentity(asset.edits)) {
      return sendFile(req, res, plyPath, { download: filename });
    }

    const cloud = decodePly(await fsp.readFile(plyPath));
    const edited = applyEdits(cloud, asset.edits);
    const buf = encodePly(edited);
    res.writeHead(200, {
      'content-type': 'application/octet-stream',
      'content-length': buf.length,
      'content-disposition': `attachment; filename="${filename}"`,
      'cache-control': 'no-store',
    });
    res.end(buf);
  });

  /** Same as above in the compact `.splat` format many web viewers accept. */
  r.get('/api/assets/:id/export.splat', async (req, res, { params, query }) => {
    const asset = store.getAsset(params.id);
    if (!asset?.result?.splatFile) return sendError(res, 404, 'this conversion has no splat yet');
    const splatPath = store.assetPath(params.id, 'output', asset.result.splatFile);
    const filename = `${sanitiseFilename(asset.name)}.splat`;

    if (query.get('raw') === '1' || isIdentity(asset.edits)) {
      return sendFile(req, res, splatPath, { download: filename });
    }
    const cloud = decodeSplatBuffer(await fsp.readFile(splatPath));
    const buf = encodeSplatBuffer(applyEdits(cloud, asset.edits));
    res.writeHead(200, {
      'content-type': 'application/octet-stream',
      'content-length': buf.length,
      'content-disposition': `attachment; filename="${filename}"`,
      'cache-control': 'no-store',
    });
    res.end(buf);
  });

  r.get('/api/events', (req, res) => {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache, no-transform',
      connection: 'keep-alive',
      'x-accel-buffering': 'no',
    });
    res.write('retry: 3000\n\n');

    const onUpdate = (payload) => {
      res.write(`event: update\ndata: ${JSON.stringify(payload)}\n\n`);
    };
    jobs.events.on('update', onUpdate);

    // Proxies drop idle streams; a comment frame keeps this one warm.
    const beat = setInterval(() => res.write(': ping\n\n'), 20_000);
    const close = () => {
      clearInterval(beat);
      jobs.events.off('update', onUpdate);
    };
    req.on('close', close);
    req.on('error', close);
  });

  return r;
}

async function handleUpload(req, res) {
  const id = store.newId();
  await store.ensureAssetDirs(id);

  const meta = { name: '', kind: 'photos', settings: {} };
  const sourceFiles = [];
  let frameCount = 0;
  const opened = [];

  const cleanup = async () => {
    await Promise.allSettled(opened.map((s) => new Promise((resolve) => s.close(resolve))));
    await fsp.rm(store.assetDir(id), { recursive: true, force: true });
  };

  try {
    await parseMultipart(req, {
      maxBytes: MAX_UPLOAD_BYTES,
      onField(name, value) {
        if (name === 'name') meta.name = value.trim().slice(0, 160);
        else if (name === 'kind') meta.kind = value === 'video' ? 'video' : 'photos';
        else if (name === 'settings') {
          try { meta.settings = sanitiseSettings(JSON.parse(value)); } catch { /* keep defaults */ }
        }
      },
      async onFile(part) {
        if (part.name === 'frame') {
          frameCount += 1;
          const target = store.assetPath(id, 'frames', `frame_${String(frameCount).padStart(5, '0')}.png`);
          return pngSink(target, opened);
        }
        if (part.name === 'source') {
          const safe = sanitiseFilename(part.filename || `source_${sourceFiles.length + 1}`);
          const target = store.assetPath(id, 'source', `${String(sourceFiles.length + 1).padStart(3, '0')}_${safe}`);
          sourceFiles.push({ name: safe, contentType: part.contentType, stored: path.basename(target) });
          return fileSink(target, opened);
        }
        return nullSink();
      },
    });
  } catch (err) {
    await cleanup();
    return sendError(res, err.status || 400, err.message);
  }

  if (!frameCount) {
    await cleanup();
    return sendError(res, 400, 'no frames were included in the upload');
  }

  const asset = await store.createAsset({
    id,
    name: meta.name || defaultName(meta.kind, sourceFiles),
    kind: meta.kind,
    settings: meta.settings,
    source: { files: sourceFiles, frameCount },
  });

  jobs.enqueue(asset.id);
  sendJson(res, 201, { asset: store.getAssetView(asset.id) });
}

function defaultName(kind, sourceFiles) {
  if (sourceFiles.length === 1) return sourceFiles[0].name.replace(/\.[^.]+$/, '');
  const when = new Date().toLocaleString();
  return `${kind === 'video' ? 'Video' : 'Photo'} capture — ${when}`;
}

export function sanitiseSettings(input) {
  const out = {};
  const num = (key, lo, hi) => {
    const n = Number(input[key]);
    if (Number.isFinite(n)) out[key] = Math.min(hi, Math.max(lo, n));
  };
  if (['auto', 'preview', 'colmap'].includes(input.backend)) out.backend = input.backend;
  if (['sequential', 'exhaustive'].includes(input.matcher)) out.matcher = input.matcher;
  num('iterations', 100, 60_000);
  num('maxFrames', 1, 600);
  num('detail', 24, 512);
  num('arcDeg', 0, 360);
  num('splatScale', 0.1, 5);
  num('subjectThreshold', 0, 1);
  return out;
}

/** Writable sink that rejects anything that is not a PNG. */
function pngSink(target, opened) {
  const stream = fs.createWriteStream(target);
  opened.push(stream);
  let checked = false;
  let head = Buffer.alloc(0);
  return {
    async write(chunk) {
      if (!checked) {
        head = Buffer.concat([head, chunk]);
        if (head.length >= 4) {
          checked = true;
          if (!head.subarray(0, 4).equals(PNG_MAGIC)) {
            throw Object.assign(new Error('frames must be PNG images'), { status: 400 });
          }
        }
      }
      await writeChunk(stream, chunk);
    },
    end() { return closeStream(stream); },
  };
}

function fileSink(target, opened) {
  const stream = fs.createWriteStream(target);
  opened.push(stream);
  return {
    write: (chunk) => writeChunk(stream, chunk),
    end: () => closeStream(stream),
  };
}

function nullSink() {
  return { write: async () => {}, end: async () => {} };
}

function writeChunk(stream, chunk) {
  return new Promise((resolve, reject) => {
    if (stream.write(chunk)) return resolve();
    stream.once('drain', resolve);
    stream.once('error', reject);
  });
}

function closeStream(stream) {
  return new Promise((resolve, reject) => {
    stream.end((err) => (err ? reject(err) : resolve()));
  });
}

async function copyTree(from, to) {
  try {
    await fsp.cp(from, to, { recursive: true });
  } catch (err) {
    if (err.code !== 'ENOENT') throw err;
  }
}
