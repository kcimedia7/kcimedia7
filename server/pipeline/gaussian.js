import fsp from 'node:fs/promises';
import path from 'node:path';
import { run } from './run.js';
import { PYTHON_CMD, TRAINER_DIR } from './backends.js';

/**
 * Real 3D Gaussian Splatting, run by the Python trainer in `trainer/`.
 *
 * Unlike the `colmap` backend -- which drives an external COLMAP binary and
 * whatever CUDA trainer the operator configured -- this one owns the whole
 * reconstruction: pycolmap solves the camera poses, then gaussians are
 * optimised against the posed images with a differentiable rasterizer. It needs
 * no GPU, which is the point: it is slower than CUDA, not different from it.
 */

/** Fractions of the training stage, used to turn trainer output into progress. */
const PHASE_WEIGHTS = { sfm: 0.25, train: 0.7, write: 0.05 };

export async function runGaussianTraining({
  imagesDir, workDir, outputDir, settings = {}, log, onProgress, signal,
}) {
  await fsp.mkdir(outputDir, { recursive: true });
  await fsp.mkdir(workDir, { recursive: true });

  const iterations = clampInt(settings.iterations, 200, 60_000, 3000);
  const resolution = clampInt(settings.trainResolution, 96, 1600, 320);
  const maxGaussians = clampInt(settings.maxGaussians, 1000, 2_000_000, 150_000);

  const args = [
    '-m', 'splatworks_train.train',
    '--images', imagesDir,
    '--output', outputDir,
    '--work', path.join(workDir, 'sfm'),
    '--iterations', String(iterations),
    '--resolution', String(resolution),
    '--max-gaussians', String(maxGaussians),
    '--matcher', settings.matcher === 'sequential' ? 'sequential' : 'exhaustive',
  ];
  if (settings.densifyGradThreshold) {
    args.push('--densify-grad-threshold', String(settings.densifyGradThreshold));
  }

  log(`training: ${PYTHON_CMD} ${args.join(' ')}`);

  await run(PYTHON_CMD, args, {
    cwd: TRAINER_DIR,
    env: { PYTHONUNBUFFERED: '1' },
    signal,
    onLine: (line) => {
      log(line);
      const update = parseProgress(line, iterations);
      if (update) onProgress?.(update);
    },
  });

  const report = await readReport(outputDir);
  const plyPath = path.join(outputDir, 'point_cloud.ply');
  await fsp.access(plyPath);
  return { plyPath, report };
}

/**
 * Map the trainer's stdout onto a 0..1 fraction of the training stage. The
 * trainer prints its own phases, so progress reflects real work rather than a
 * timer.
 */
export function parseProgress(line, totalIterations) {
  if (/^sfm: extracting features/.test(line)) return { fraction: 0.04, label: 'Finding features' };
  if (/^sfm: matching/.test(line)) return { fraction: 0.10, label: 'Matching images' };
  if (/^sfm: incremental mapping/.test(line)) return { fraction: 0.16, label: 'Solving camera poses' };

  const registered = /^sfm: registered (\d+) images, (\d+) points/.exec(line);
  if (registered) {
    return { fraction: PHASE_WEIGHTS.sfm, label: `Solved ${registered[1]} camera poses` };
  }

  const iter = /^iter (\d+)\/(\d+)\s+loss ([\d.]+).*?gaussians (\d+)/.exec(line);
  if (iter) {
    const done = Number(iter[1]) / Math.max(1, Number(iter[2]) || totalIterations);
    return {
      fraction: PHASE_WEIGHTS.sfm + PHASE_WEIGHTS.train * done,
      label: `Optimising gaussians — ${Number(iter[4]).toLocaleString()} splats, loss ${iter[3]}`,
    };
  }

  if (/^final PSNR/.test(line)) {
    // Training is over and the write has started, so this must sit past the
    // last iteration line rather than level with it.
    return {
      fraction: PHASE_WEIGHTS.sfm + PHASE_WEIGHTS.train + PHASE_WEIGHTS.write * 0.5,
      label: 'Writing the splat',
    };
  }
  return null;
}

async function readReport(outputDir) {
  try {
    return JSON.parse(await fsp.readFile(path.join(outputDir, 'report.json'), 'utf8'));
  } catch {
    return null;
  }
}

function clampInt(value, lo, hi, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(hi, Math.max(lo, Math.round(n)));
}
