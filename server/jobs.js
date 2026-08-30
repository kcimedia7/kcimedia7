import { EventEmitter } from 'node:events';
import { convert } from './pipeline/index.js';
import * as store from './store.js';
import { MAX_CONCURRENT_JOBS } from './config.js';

/**
 * Conversion queue.
 *
 * Reconstruction is CPU/GPU bound and a real trainer wants a whole GPU, so jobs
 * run at a fixed concurrency rather than all at once. Progress is written back
 * to the library so a page reload during a long conversion picks up where the
 * UI left off, and mirrored to `events` for live updates.
 */

export const events = new EventEmitter();
events.setMaxListeners(0);

const queue = [];
const running = new Map(); // assetId -> AbortController
let started = false;

export function enqueue(assetId) {
  if (running.has(assetId) || queue.includes(assetId)) return false;
  queue.push(assetId);
  emit(assetId, 'queued');
  pump();
  return true;
}

export function isActive(assetId) {
  return running.has(assetId) || queue.includes(assetId);
}

export function cancel(assetId) {
  const idx = queue.indexOf(assetId);
  if (idx !== -1) {
    queue.splice(idx, 1);
    return true;
  }
  const controller = running.get(assetId);
  if (controller) {
    controller.abort();
    return true;
  }
  return false;
}

/**
 * Stop every job and wait for the running ones to actually let go.
 *
 * Aborting a conversion is not instant: the pipeline notices the signal between
 * stages, and until it does it is still writing frames and output files. Anyone
 * who tears down the data directory before that -- a test, a shutdown path --
 * races a live writer and gets ENOTEMPTY from a directory that was empty a
 * moment earlier.
 *
 * @returns {Promise<boolean>} false if a job was still running at the deadline
 */
export async function drain({ timeoutMs = 10_000 } = {}) {
  queue.length = 0;
  for (const controller of running.values()) controller.abort();
  const deadline = Date.now() + timeoutMs;
  while (running.size && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  return running.size === 0;
}

export function queueState() {
  return { running: [...running.keys()], queued: [...queue], limit: MAX_CONCURRENT_JOBS };
}

/** Re-queue anything that was mid-flight when the process last stopped. */
export async function resumeInterrupted() {
  if (started) return;
  started = true;
  for (const asset of store.listAssets()) {
    if (asset.status === 'running' || asset.status === 'queued') {
      await store.updateAsset(asset.id, {
        status: 'queued',
        stage: 'queued',
        progress: 0,
        message: 'Re-queued after the server restarted',
      });
      queue.push(asset.id);
    }
  }
  pump();
}

function pump() {
  while (running.size < MAX_CONCURRENT_JOBS && queue.length) {
    const id = queue.shift();
    void start(id);
  }
}

async function start(assetId) {
  const asset = store.getAsset(assetId);
  if (!asset) return;

  const controller = new AbortController();
  running.set(assetId, controller);

  await store.updateAsset(assetId, {
    status: 'running',
    stage: 'ingest',
    progress: 0,
    message: 'Starting',
    error: null,
    startedAt: new Date().toISOString(),
    finishedAt: null,
  });
  emit(assetId, 'started');

  // Progress fires far more often than it is worth persisting; throttle the
  // writes but always let the emitter through for live UI.
  let lastWrite = 0;
  const onProgress = (update) => {
    const patch = {
      stage: update.stage,
      progress: update.progress,
      message: update.message || update.label,
    };
    const now = Date.now();
    if (now - lastWrite > 400) {
      lastWrite = now;
      void store.updateAsset(assetId, patch);
    } else {
      Object.assign(store.getAsset(assetId) || {}, patch);
    }
    emit(assetId, 'progress', patch);
  };

  const onLog = (line) => {
    void store.appendLog(assetId, line);
    emit(assetId, 'log', { line });
  };

  try {
    await store.ensureAssetDirs(assetId);
    const result = await convert({
      framesDir: store.assetPath(assetId, 'frames'),
      workDir: store.assetPath(assetId, 'work'),
      outputDir: store.assetPath(assetId, 'output'),
      settings: { ...asset.settings, kind: asset.kind },
      onProgress,
      onLog,
      signal: controller.signal,
    });

    await store.updateAsset(assetId, {
      status: 'ready',
      stage: 'done',
      progress: 1,
      message: `${result.splatCount.toLocaleString()} gaussians`,
      backend: result.backend,
      result,
      finishedAt: new Date().toISOString(),
    });
    emit(assetId, 'done');
  } catch (err) {
    const aborted = controller.signal.aborted;
    await store.appendLog(assetId, `${aborted ? 'cancelled' : 'error'}: ${err.message}`);
    await store.updateAsset(assetId, {
      status: aborted ? 'cancelled' : 'failed',
      stage: 'done',
      message: aborted ? 'Cancelled' : 'Conversion failed',
      error: aborted ? null : err.message,
      finishedAt: new Date().toISOString(),
    });
    emit(assetId, aborted ? 'cancelled' : 'failed', { error: err.message });
  } finally {
    running.delete(assetId);
    pump();
  }
}

function emit(assetId, type, data = {}) {
  events.emit('update', { assetId, type, ...data, at: new Date().toISOString() });
}
