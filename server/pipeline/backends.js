import { commandExists } from './run.js';
import { BACKEND } from '../config.js';

/**
 * Which reconstruction path this machine can actually run.
 *
 * A real 3D Gaussian Splatting conversion needs structure-from-motion (COLMAP)
 * to recover camera poses, then a CUDA trainer to optimise the gaussians. Both
 * are heavy external installs, so they are detected rather than bundled, and the
 * preview backend covers the machine that has neither.
 */

let cached = null;

export const TRAINER_CMD = process.env.SPLAT_TRAINER_CMD || '';
export const COLMAP_CMD = process.env.SPLAT_COLMAP_CMD || 'colmap';
export const FFMPEG_CMD = process.env.SPLAT_FFMPEG_CMD || 'ffmpeg';

export async function detectCapabilities({ refresh = false } = {}) {
  if (cached && !refresh) return cached;

  const [colmap, ffmpeg, gpu] = await Promise.all([
    commandExists(COLMAP_CMD, ['--help']),
    commandExists(FFMPEG_CMD, ['-version']),
    commandExists('nvidia-smi', ['-L']),
  ]);
  const trainer = Boolean(TRAINER_CMD);

  const canTrain = colmap && trainer;
  let backend;
  if (BACKEND === 'colmap') backend = 'colmap';
  else if (BACKEND === 'preview') backend = 'preview';
  else backend = canTrain ? 'colmap' : 'preview';

  cached = {
    backend,
    forced: BACKEND !== 'auto',
    colmap,
    ffmpeg,
    gpu,
    trainer,
    trainerCmd: TRAINER_CMD,
    reasons: explain({ colmap, trainer, gpu, ffmpeg }),
  };
  return cached;
}

function explain({ colmap, trainer, gpu, ffmpeg }) {
  const out = [];
  if (!colmap) out.push('COLMAP was not found on PATH, so camera poses cannot be solved.');
  if (!trainer) out.push('No trainer is configured (set SPLAT_TRAINER_CMD) to optimise gaussians.');
  if (colmap && trainer && !gpu) out.push('No NVIDIA GPU was detected; training will be very slow.');
  if (!ffmpeg) out.push('ffmpeg was not found; video frames are extracted in the browser instead.');
  return out;
}

export function resetCapabilityCache() {
  cached = null;
}
