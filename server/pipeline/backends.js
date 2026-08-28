import path from 'node:path';
import fs from 'node:fs';
import { commandExists, run } from './run.js';
import { BACKEND, ROOT } from '../config.js';

/**
 * Which reconstruction path this machine can actually run, best first:
 *
 *   colmap    an external COLMAP binary plus the CUDA trainer named by
 *             SPLAT_TRAINER_CMD -- the fastest route when a GPU box is set up
 *   gaussian  the bundled Python trainer in trainer/: pycolmap solves the poses
 *             and a differentiable rasterizer optimises the gaussians. Real 3DGS
 *             with no GPU and no system installs, just two pip wheels
 *   preview   the dependency-free proxy, for a machine with neither
 *
 * Only `preview` is a different kind of output. The first two both produce a
 * genuine reconstruction; they differ in how fast they get there.
 */

let cached = null;
let inflight = null;

export const TRAINER_CMD = process.env.SPLAT_TRAINER_CMD || '';
export const COLMAP_CMD = process.env.SPLAT_COLMAP_CMD || 'colmap';
export const FFMPEG_CMD = process.env.SPLAT_FFMPEG_CMD || 'ffmpeg';
export const PYTHON_CMD = process.env.SPLAT_PYTHON_CMD || 'python3';
export const TRAINER_DIR = process.env.SPLAT_TRAINER_DIR || path.join(ROOT, 'trainer');

/** Does `python3` in trainer/ have the two wheels the bundled trainer needs? */
async function detectPythonTrainer() {
  if (!fs.existsSync(path.join(TRAINER_DIR, 'splatworks_train', 'train.py'))) {
    return { available: false, reason: 'the bundled trainer is not present' };
  }
  const missing = [];
  try {
    await run(PYTHON_CMD, ['-c', 'import pycolmap'], { cwd: TRAINER_DIR, timeoutMs: 60_000 });
  } catch {
    missing.push('pycolmap');
  }
  try {
    await run(PYTHON_CMD, ['-c', 'import torch'], { cwd: TRAINER_DIR, timeoutMs: 120_000 });
  } catch {
    missing.push('torch');
  }
  if (missing.length) {
    return { available: false, reason: `install ${missing.join(' and ')} (pip install -r trainer/requirements.txt)` };
  }
  return { available: true, reason: null };
}

/**
 * Capabilities already known, or null while the probe is still running.
 *
 * Detection shells out to `python -c "import torch"`, which costs seconds on a
 * warm cache and far longer on a cold one. Callers that must not block -- the
 * health endpoint above all -- read this instead of awaiting.
 */
export function peekCapabilities() {
  return cached;
}

export async function detectCapabilities({ refresh = false } = {}) {
  if (cached && !refresh) return cached;
  // Concurrent callers share one probe rather than each spawning their own.
  if (inflight && !refresh) return inflight;
  inflight = probeCapabilities();
  try {
    return await inflight;
  } finally {
    inflight = null;
  }
}

async function probeCapabilities() {

  const [colmap, ffmpeg, gpu, python] = await Promise.all([
    commandExists(COLMAP_CMD, ['--help']),
    commandExists(FFMPEG_CMD, ['-version']),
    commandExists('nvidia-smi', ['-L']),
    detectPythonTrainer(),
  ]);
  const trainer = Boolean(TRAINER_CMD);

  const canRunExternal = colmap && trainer;
  let backend;
  if (['colmap', 'preview', 'gaussian'].includes(BACKEND)) backend = BACKEND;
  else if (canRunExternal) backend = 'colmap';
  else if (python.available) backend = 'gaussian';
  else backend = 'preview';

  cached = {
    backend,
    forced: BACKEND !== 'auto',
    colmap,
    ffmpeg,
    gpu,
    trainer,
    trainerCmd: TRAINER_CMD,
    pythonTrainer: python.available,
    /** True when the chosen backend produces a genuine reconstruction. */
    reconstructs: backend !== 'preview',
    reasons: explain({ colmap, trainer, gpu, ffmpeg, python, backend }),
  };
  return cached;
}

function explain({ colmap, trainer, gpu, ffmpeg, python, backend }) {
  const out = [];
  if (backend === 'gaussian') {
    out.push('Using the bundled trainer: pycolmap solves camera poses and gaussians are optimised on the CPU.');
    if (!gpu) out.push('No NVIDIA GPU was detected, so training runs on the CPU and is slow but real.');
  }
  if (backend === 'preview') {
    if (!python.available) out.push(`The bundled trainer is unavailable: ${python.reason}.`);
    if (!colmap) out.push('COLMAP was not found on PATH.');
    if (!trainer) out.push('No external trainer is configured (SPLAT_TRAINER_CMD).');
  }
  if (backend === 'colmap' && !gpu) {
    out.push('No NVIDIA GPU was detected; the configured trainer may be very slow.');
  }
  if (!ffmpeg) out.push('ffmpeg was not found; video frames are extracted in the browser instead.');
  return out;
}

export function resetCapabilityCache() {
  cached = null;
  inflight = null;
}
