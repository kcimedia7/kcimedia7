import fsp from 'node:fs/promises';
import path from 'node:path';
import { run } from './run.js';
import { COLMAP_CMD, TRAINER_CMD } from './backends.js';

/**
 * The real reconstruction path: COLMAP solves camera poses from the frames, then
 * a configured 3DGS trainer optimises gaussians against those posed images.
 *
 * The trainer is a command template rather than a bundled dependency, because
 * every 3DGS implementation (INRIA's `train.py`, gsplat, nerfstudio, …) has its
 * own CLI. Example:
 *
 *   SPLAT_TRAINER_CMD="python /opt/gaussian-splatting/train.py \
 *     -s {source} -m {output} --iterations {iterations}"
 */

const PLACEHOLDER = /\{([A-Za-z_]+)\}/g;

export function buildTrainerArgv(template, vars) {
  const tokens = tokenize(template);
  if (!tokens.length) throw new Error('SPLAT_TRAINER_CMD is empty');
  // Any {word} is treated as a placeholder, so a typo like {sources} fails here
  // with a clear message rather than reaching the trainer as literal text.
  const filled = tokens.map((t) => t.replace(PLACEHOLDER, (whole, key) => {
    if (!(key in vars)) {
      throw new Error(
        `SPLAT_TRAINER_CMD uses unknown placeholder ${whole}; `
        + `supported placeholders are ${Object.keys(vars).map((k) => `{${k}}`).join(', ')}`,
      );
    }
    return String(vars[key]);
  }));
  return { cmd: filled[0], args: filled.slice(1) };
}

/** Split a command template on whitespace, honouring quoted segments. */
export function tokenize(template) {
  const out = [];
  const re = /"([^"]*)"|'([^']*)'|(\S+)/g;
  let m;
  while ((m = re.exec(template))) out.push(m[1] ?? m[2] ?? m[3]);
  return out;
}

/**
 * Run COLMAP end to end over `imagesDir`, producing the dataset layout that
 * 3DGS trainers expect at `workDir/colmap`.
 */
export async function runColmap({ imagesDir, workDir, log, signal, matcher = 'sequential' }) {
  const base = path.join(workDir, 'colmap');
  const db = path.join(base, 'database.db');
  const sparse = path.join(base, 'sparse');
  await fsp.mkdir(sparse, { recursive: true });

  const onLine = (line) => log(line);
  const exec = (args) => run(COLMAP_CMD, args, { onLine, signal, cwd: workDir });

  log('colmap feature_extractor');
  await exec([
    'feature_extractor',
    '--database_path', db,
    '--image_path', imagesDir,
    '--ImageReader.single_camera', '1',
    '--ImageReader.camera_model', 'OPENCV',
  ]);

  // Video frames arrive in capture order, so sequential matching is both much
  // faster and more reliable than exhaustive matching for that case.
  const matcherCmd = matcher === 'exhaustive' ? 'exhaustive_matcher' : 'sequential_matcher';
  log(`colmap ${matcherCmd}`);
  await exec([matcherCmd, '--database_path', db]);

  log('colmap mapper');
  await exec([
    'mapper',
    '--database_path', db,
    '--image_path', imagesDir,
    '--output_path', sparse,
  ]);

  const models = (await fsp.readdir(sparse)).filter((d) => /^\d+$/.test(d)).sort();
  if (!models.length) {
    throw new Error('COLMAP could not solve camera poses — the frames may lack overlap or texture');
  }

  const dataset = path.join(base, 'dataset');
  log('colmap image_undistorter');
  await exec([
    'image_undistorter',
    '--image_path', imagesDir,
    '--input_path', path.join(sparse, models[0]),
    '--output_path', dataset,
    '--output_type', 'COLMAP',
  ]);

  return { datasetDir: dataset, sparseDir: path.join(sparse, models[0]), models: models.length };
}

/** Invoke the configured trainer and return the PLY it wrote. */
export async function runTrainer({ datasetDir, outputDir, iterations, log, signal }) {
  const { cmd, args } = buildTrainerArgv(TRAINER_CMD, {
    source: datasetDir,
    output: outputDir,
    images: path.join(datasetDir, 'images'),
    iterations,
  });
  log(`trainer: ${cmd} ${args.join(' ')}`);
  await run(cmd, args, { onLine: log, signal });

  const ply = await findPly(outputDir);
  if (!ply) throw new Error(`the trainer finished but wrote no .ply under ${outputDir}`);
  return ply;
}

/** Depth-first search for the newest .ply the trainer produced. */
export async function findPly(dir) {
  let best = null;
  const walk = async (d, depth) => {
    if (depth > 6) return;
    let entries;
    try {
      entries = await fsp.readdir(d, { withFileTypes: true });
    } catch {
      return;
    }
    for (const e of entries) {
      const full = path.join(d, e.name);
      if (e.isDirectory()) await walk(full, depth + 1);
      else if (e.name.endsWith('.ply')) {
        const stat = await fsp.stat(full);
        if (!best || stat.mtimeMs > best.mtimeMs) best = { path: full, mtimeMs: stat.mtimeMs };
      }
    }
  };
  await walk(dir, 0);
  return best?.path || null;
}
