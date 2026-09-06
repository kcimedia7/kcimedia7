import fsp from 'node:fs/promises';
import path from 'node:path';
import { run } from './run.js';
import { PYTHON_CMD, TRAINER_DIR } from './backends.js';
import { resolveDevice } from '../config.js';

/**
 * Single-panorama reconstruction from predicted depth.
 *
 * The other backends measure. This one guesses: a monocular depth model has an
 * opinion about what is near and what is far, and that opinion is usually
 * convincing enough to walk around in. It exists because the alternative for a
 * single 360 photo is nothing at all -- one optical centre means no parallax,
 * and no amount of solving recovers depth that was never recorded.
 *
 * The result is a different kind of object from a reconstruction, and the
 * pipeline labels it as such rather than letting it sit unmarked beside them.
 */

const PHASE_WEIGHTS = { load: 0.08, estimate: 0.72, fuse: 0.2 };

export async function runDepthReconstruction({
  imagesDir, outputDir, settings = {}, log, onProgress, signal,
}) {
  await fsp.mkdir(outputDir, { recursive: true });

  const device = resolveDevice(settings.device);
  // A 2k panorama unprojects to about 1.85M points. Thinning below the
  // source resolution spreads the gaussians out and costs sharpness, so the
  // default keeps one whole rather than discarding two thirds of it.
  const maxPoints = clampInt(settings.maxGaussians, 10_000, 8_000_000, 2_000_000);
  const fov = clampNumber(settings.panoFovDeg, 40, 179, 100);
  const farRatio = clampNumber(settings.farRatio, 5, 500, 60);

  const args = [
    '-m', 'splatworks_train.from_panorama',
    '--images', imagesDir,
    '--output', outputDir,
    '--fov', String(fov),
    '--max-points', String(maxPoints),
    '--far-ratio', String(farRatio),
    '--device', device,
  ];
  if (settings.depthModel) args.push('--model', String(settings.depthModel));

  log(`depth: ${PYTHON_CMD} ${args.join(' ')}`);

  await run(PYTHON_CMD, args, {
    cwd: TRAINER_DIR,
    env: { PYTHONUNBUFFERED: '1' },
    signal,
    onLine: (line) => {
      log(line);
      const update = parseProgress(line);
      if (update) onProgress?.(update);
    },
  });

  const report = await readReport(outputDir);
  const plyPath = path.join(outputDir, 'point_cloud.ply');
  await fsp.access(plyPath);
  return { plyPath, report };
}

/** Map the estimator's own output onto a fraction of the stage. */
export function parseProgress(line) {
  if (/^depth: reading/.test(line)) return { fraction: 0.04, label: 'Reading the panorama' };
  if (/^depth: loading/.test(line)) {
    // The first run downloads the model, which is most of the wait and reports
    // nothing while it happens.
    return { fraction: 0.08, label: 'Loading the depth model' };
  }
  const estimating = /^depth: estimating (\w+) \((\d+)\/(\d+)\)/.exec(line);
  if (estimating) {
    const done = Number(estimating[2]) / Number(estimating[3]);
    return {
      fraction: PHASE_WEIGHTS.load + PHASE_WEIGHTS.estimate * done,
      label: `Estimating depth (${estimating[1]})`,
    };
  }
  if (/^depth: aligning/.test(line)) {
    return { fraction: PHASE_WEIGHTS.load + PHASE_WEIGHTS.estimate, label: 'Building the cloud' };
  }
  if (/^wrote /.test(line)) return { fraction: 1, label: 'Writing splat files' };
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
  const n = Math.round(Number(value));
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : fallback;
}

function clampNumber(value, lo, hi, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : fallback;
}
