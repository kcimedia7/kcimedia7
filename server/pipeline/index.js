import fsp from 'node:fs/promises';
import path from 'node:path';
import { decodePng, encodePng } from './png.js';
import { fitWithin } from './imageops.js';
import { encodePly, decodePly } from './ply.js';
import { encodeSplatBuffer, boundsOf } from './splat.js';
import { reconstructPreview } from './preview.js';
import { detectCapabilities } from './backends.js';
import { runColmap, runTrainer } from './colmap.js';
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
  const backend = ctx.settings.backend && ctx.settings.backend !== 'auto'
    ? ctx.settings.backend
    : caps.backend;
  const plan = stagePlan(backend);

  const progress = stageProgress(plan, ctx.onProgress);
  const log = (line) => ctx.onLog(line);

  log(`backend: ${backend}${caps.forced ? ' (forced by SPLAT_BACKEND)' : ''}`);
  for (const reason of caps.reasons) log(reason);

  progress.enter('ingest');
  const framePaths = await listFrames(ctx.framesDir);
  if (!framePaths.length) throw new Error('no frames were uploaded for this capture');
  log(`${framePaths.length} frame(s) ready`);

  let cloud;
  let stats;

  if (backend === 'colmap') {
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
    within(id, fraction) {
      if (current().id !== id) return;
      localFraction = Math.max(localFraction, Math.min(1, fraction));
      push();
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
