import fsp from 'node:fs/promises';
import path from 'node:path';
import { decodePng, encodePng } from './png.js';
import { fitWithin } from './imageops.js';
import { encodePly, decodePly } from './ply.js';
import { runDepthReconstruction } from './depth.js';
import { encodeSplatBuffer, boundsOf, filterCloud } from './splat.js';
import { reconstructPreview } from './preview.js';
import { detectCapabilities } from './backends.js';
import { runColmap, runTrainer } from './colmap.js';
import { runGaussianTraining } from './gaussian.js';
import { PREVIEW_MAX_FRAMES, PREVIEW_GRID } from '../config.js';

/**
 * Stage plan per backend. Weights are fractions of overall progress, so the
 * UI's bar advances smoothly regardless of which path a conversion takes.
 */
const PLANS = {
  preview: [
    { id: 'ingest', label: 'Reading frames', weight: 0.18 },
    { id: 'reconstruct', label: 'Building gaussians', weight: 0.62 },
    { id: 'export', label: 'Writing splat files', weight: 0.20 },
  ],
  colmap: [
    { id: 'ingest', label: 'Reading frames', weight: 0.06 },
    { id: 'poses', label: 'Solving camera poses (COLMAP)', weight: 0.34 },
    { id: 'train', label: 'Training gaussians', weight: 0.48 },
    { id: 'export', label: 'Writing splat files', weight: 0.12 },
  ],
  // The bundled trainer reports its own phases, so pose solving and gaussian
  // optimisation share one stage whose fraction comes from the trainer itself.
  gaussian: [
    { id: 'ingest', label: 'Reading frames', weight: 0.04 },
    { id: 'train', label: 'Reconstructing', weight: 0.88 },
    { id: 'export', label: 'Writing splat files', weight: 0.08 },
  ],
  // Inferred rather than solved: one panorama, six depth estimates, no poses
  // to recover because there is only ever one viewpoint.
  depth: [
    { id: 'ingest', label: 'Reading frames', weight: 0.04 },
    { id: 'estimate', label: 'Estimating depth', weight: 0.88 },
    { id: 'export', label: 'Writing splat files', weight: 0.08 },
  ],
};

export function stagePlan(backend) {
  return PLANS[backend] || PLANS.preview;
}

/**
 * Convert one asset's frames into a gaussian splat.
 *
 * @param {object} ctx
 * @param {string} ctx.framesDir   directory of numbered PNG frames
 * @param {string} ctx.workDir     scratch space for COLMAP and the trainer
 * @param {string} ctx.outputDir   where the finished splat files land
 * @param {object} ctx.settings    per-conversion settings from the upload form
 * @param {(update: {stage: string, label: string, progress: number, message?: string}) => void} ctx.onProgress
 * @param {(line: string) => void} ctx.onLog
 * @param {AbortSignal} [ctx.signal]
 */
export async function convert(ctx) {
  const caps = await detectCapabilities();
  const framePaths = await listFrames(ctx.framesDir);
  const backend = chooseBackend(ctx.settings, caps, framePaths.length);
  const plan = stagePlan(backend);

  const progress = stageProgress(plan, ctx.onProgress);
  const log = (line) => ctx.onLog(line);

  log(`backend: ${backend}${caps.forced ? ' (forced by SPLAT_BACKEND)' : ''}`);
  for (const reason of caps.reasons) log(reason);

  progress.enter('ingest');
  if (!framePaths.length) throw new Error('no frames were uploaded for this capture');
  log(`${framePaths.length} frame(s) ready`);
  if (backend !== 'depth') refuseSingleViewpoint(ctx.settings.kind, framePaths.length);

  let cloud;
  let stats;

  if (backend === 'gaussian') {
    progress.done('ingest');
    progress.enter('train');
    const { plyPath, report } = await runGaussianTraining({
      imagesDir: ctx.framesDir,
      workDir: ctx.workDir,
      outputDir: path.join(ctx.workDir, 'train'),
      settings: ctx.settings,
      log,
      onProgress: (update) => progress.within('train', update.fraction, update.label),
      signal: ctx.signal,
    });
    cloud = decodePly(await fsp.readFile(plyPath));
    stats = {
      frames: framePaths.length,
      splats: cloud.count,
      iterations: report?.iterations,
      registeredViews: report?.registered_views,
      psnr: report?.psnr,
      trainSeconds: report?.train_seconds,
      sfmSeconds: report?.sfm_seconds,
      trainResolution: report?.resolution,
      // The scale the position learning rate is tied to, and how often the
      // guard rails fired. Both are what distinguishes a marginal run from a
      // healthy one when the output looks plausible either way.
      sceneExtent: report?.scene_extent,
      suppressedGradients: report?.suppressed_gradients,
      device: report?.device,
    };
    if (report?.psnr) log(`reconstruction quality: ${report.psnr.toFixed(2)} dB PSNR`);
    progress.done('train');
  } else if (backend === 'depth') {
    progress.done('ingest');
    progress.enter('estimate');
    const { plyPath, report } = await runDepthReconstruction({
      imagesDir: ctx.framesDir,
      outputDir: path.join(ctx.workDir, 'depth'),
      settings: ctx.settings,
      log: (line) => { log(line); progress.nudge(); },
      onProgress: ({ fraction, label }) => progress.within('estimate', fraction, label),
      signal: ctx.signal,
    });
    cloud = decodePly(await fsp.readFile(plyPath));
    stats = {
      frames: framePaths.length,
      splats: cloud.count,
      // Named so it cannot be mistaken for a measurement further down.
      depth: 'inferred',
      depthModel: report?.model,
      device: report?.device,
    };
    log(`built ${cloud.count.toLocaleString()} gaussians from inferred depth`);
    progress.done('estimate');
  } else if (backend === 'colmap') {
    progress.done('ingest');
    progress.enter('poses');
    const matcher = ctx.settings.matcher || (ctx.settings.kind === 'video' ? 'sequential' : 'exhaustive');
    const { datasetDir, models } = await runColmap({
      imagesDir: ctx.framesDir,
      workDir: ctx.workDir,
      matcher,
      log: (line) => { log(line); progress.nudge(); },
      signal: ctx.signal,
    });
    log(`COLMAP solved ${models} model(s)`);
    progress.done('poses');

    progress.enter('train');
    const iterations = Number(ctx.settings.iterations) || 7000;
    const plyPath = await runTrainer({
      datasetDir,
      outputDir: path.join(ctx.workDir, 'train'),
      iterations,
      log: (line) => { log(line); progress.nudge(); },
      signal: ctx.signal,
    });
    log(`trainer produced ${path.basename(plyPath)}`);
    cloud = decodePly(await fsp.readFile(plyPath));
    stats = { frames: framePaths.length, splats: cloud.count, iterations, matcher };
    progress.done('train');
  } else {
    const maxFrames = Number(ctx.settings.maxFrames) || PREVIEW_MAX_FRAMES;
    const chosen = pickEvenly(framePaths, maxFrames);
    log(`preview backend: using ${chosen.length} of ${framePaths.length} frame(s)`);
    const frames = [];
    for (let i = 0; i < chosen.length; i++) {
      frames.push(decodePng(await fsp.readFile(chosen[i])));
      progress.within('ingest', (i + 1) / chosen.length);
    }
    progress.done('ingest');

    progress.enter('reconstruct');
    const result = reconstructPreview(frames, {
      grid: Number(ctx.settings.detail) || PREVIEW_GRID,
      arcDeg: Number(ctx.settings.arcDeg ?? 360),
      splatScale: Number(ctx.settings.splatScale) || 1,
      subjectThreshold: Number(ctx.settings.subjectThreshold ?? 0.14),
    });
    cloud = result.cloud;
    stats = result.stats;
    log(`built ${cloud.count.toLocaleString()} gaussians from ${frames.length} frame(s)`);
    progress.done('reconstruct');
  }

  cloud = requireRenderableCloud(cloud, log);

  progress.enter('export');
  await fsp.mkdir(ctx.outputDir, { recursive: true });

  const plyBuf = encodePly(cloud);
  const splatBuf = encodeSplatBuffer(cloud);
  const plyName = 'point_cloud.ply';
  const splatName = 'cloud.splat';
  await fsp.writeFile(path.join(ctx.outputDir, plyName), plyBuf);
  progress.within('export', 0.5);
  await fsp.writeFile(path.join(ctx.outputDir, splatName), splatBuf);

  const thumb = await writeThumbnail(framePaths, ctx.outputDir);
  progress.done('export');

  const bounds = boundsOf(cloud);
  return {
    backend,
    splatFile: splatName,
    plyFile: plyName,
    thumbnail: thumb,
    splatCount: cloud.count,
    plyBytes: plyBuf.length,
    splatBytes: splatBuf.length,
    bounds,
    stats,
  };
}

async function listFrames(dir) {
  let entries;
  try {
    entries = await fsp.readdir(dir);
  } catch {
    return [];
  }
  return entries
    .filter((f) => f.toLowerCase().endsWith('.png'))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
    .map((f) => path.join(dir, f));
}

/** Evenly spaced subset, always including the first and last item. */
export function pickEvenly(items, max) {
  if (items.length <= max) return items.slice();
  const out = [];
  for (let i = 0; i < max; i++) {
    out.push(items[Math.round((i * (items.length - 1)) / (max - 1))]);
  }
  return [...new Set(out)];
}

/**
 * Which backend runs this capture.
 *
 * An explicit choice always wins. Otherwise a lone panorama is routed to depth
 * inference, because it is the only thing that can produce anything at all
 * from one viewpoint -- solving it would spend minutes to arrive at the same
 * impossibility every time.
 */
export function chooseBackend(settings = {}, caps = {}, frameCount = 0) {
  if (settings.backend && settings.backend !== 'auto') return settings.backend;
  if (settings.kind === 'pano' && frameCount > 0 && frameCount <= VIEWS_PER_PANORAMA) {
    return 'depth';
  }
  return caps.backend || 'preview';
}

/** Views a single panorama is resampled into, matching the browser's ingest. */
const VIEWS_PER_PANORAMA = 6;

/**
 * Stop a capture that cannot contain depth, before spending minutes proving it.
 *
 * One 360 photo has one optical centre. Every ray it recorded passes through a
 * single point, so no two views of it have a baseline, and depth cannot be
 * triangulated at any resolution or iteration count. Structure-from-motion does
 * fail on it -- but it fails after minutes of feature matching, reporting too
 * little overlap or texture, which is the one explanation that is definitely
 * wrong here and sends people off to reshoot with more frames of the same
 * thing.
 */
export function refuseSingleViewpoint(kind, frameCount) {
  if (kind !== 'pano' || frameCount > VIEWS_PER_PANORAMA) return;
  throw new Error(
    'A single 360 photo cannot be reconstructed. Every ray in it passes through '
    + 'one point, so there is no parallax to measure depth from -- this is a '
    + 'property of the capture, not a setting to raise. Shoot two or more '
    + 'panoramas a step or two apart and convert them together.');
}

/**
 * Refuse to present a model that cannot be seen as a finished conversion.
 *
 * When training diverges it still writes a well-formed PLY, and every signal
 * the UI shows stays correct: the header count, the file size, the status. The
 * viewer loads it, sorts it, reports a frame rate -- and draws nothing, because
 * the values are NaN. That is the worst possible outcome, because it looks like
 * a success and sends the user hunting through the renderer for a fault that is
 * in the numbers.
 *
 * A few stray gaussians are dropped and the rest kept. Beyond a small fraction
 * the reconstruction is not trustworthy even where it is finite, so the
 * conversion fails and says why.
 */
export function requireRenderableCloud(cloud, log = () => {}) {
  if (!cloud.count) {
    throw new Error('Reconstruction produced no gaussians. The frames need more overlap, '
      + 'more texture, or less motion blur.');
  }

  const finite = (i) => {
    for (let k = 0; k < 3; k++) {
      if (!Number.isFinite(cloud.positions[i * 3 + k])) return false;
      if (!Number.isFinite(cloud.scales[i * 3 + k])) return false;
    }
    if (!Number.isFinite(cloud.opacities[i])) return false;
    for (let k = 0; k < 4; k++) if (!Number.isFinite(cloud.rotations[i * 4 + k])) return false;
    return true;
  };

  const cleaned = filterCloud(cloud, finite);
  const dropped = cloud.count - cleaned.count;
  if (!dropped) return cloud;

  const share = dropped / cloud.count;
  if (share > 0.01) {
    throw new Error(
      `Training diverged: ${dropped.toLocaleString()} of ${cloud.count.toLocaleString()} `
      + `gaussians (${(share * 100).toFixed(1)}%) have non-finite values. The result would `
      + 'load and render as an empty scene. This usually means the capture gave the solver '
      + 'too little parallax -- walking straight towards a subject moves the camera along '
      + 'its own view direction, which barely constrains depth. Orbit around the subject '
      + 'instead, or re-run with fewer iterations.');
  }
  log(`dropped ${dropped} non-finite gaussian(s) of ${cloud.count}`);
  return cleaned;
}

async function writeThumbnail(framePaths, outputDir) {
  // The middle frame is usually the best-composed one in an orbit.
  const source = framePaths[Math.floor(framePaths.length / 2)];
  try {
    const img = decodePng(await fsp.readFile(source));
    const small = fitWithin(img, 480);
    await fsp.writeFile(path.join(outputDir, 'thumbnail.png'), encodePng(small));
    return 'thumbnail.png';
  } catch {
    return null;
  }
}

/**
 * Turns per-stage progress into one 0..1 number, and lets long external stages
 * (COLMAP, training) creep forward on each log line so the UI never looks stuck.
 */
function stageProgress(plan, emit) {
  let index = 0;
  let base = 0;
  let localFraction = 0;

  const current = () => plan[index] || plan[plan.length - 1];
  const push = (message) => {
    const stage = current();
    emit({
      stage: stage.id,
      label: stage.label,
      progress: Math.min(0.999, base + stage.weight * localFraction),
      message,
    });
  };

  return {
    enter(id) {
      const i = plan.findIndex((s) => s.id === id);
      if (i !== -1) {
        base = plan.slice(0, i).reduce((n, s) => n + s.weight, 0);
        index = i;
      }
      localFraction = 0;
      push();
    },
    within(id, fraction, message) {
      if (current().id !== id) return;
      localFraction = Math.max(localFraction, Math.min(1, fraction));
      push(message);
    },
    /** Asymptotic creep: each log line closes part of the remaining gap. */
    nudge() {
      localFraction += (1 - localFraction) * 0.03;
      push();
    },
    done(id) {
      if (current().id !== id) return;
      localFraction = 1;
      push();
      base += current().weight;
      index = Math.min(index + 1, plan.length - 1);
      localFraction = 0;
    },
  };
}
