#!/usr/bin/env node
/**
 * Report what is actually inside a gaussian splat PLY.
 *
 * A conversion can finish, write a valid file, load into the viewer and still
 * show nothing. The count in the status bar comes from the header, so it reads
 * correctly no matter what the values are -- which makes "ready but blank"
 * indistinguishable from "ready and fine" without looking at the numbers.
 *
 * The usual causes all show up here: positions that are not finite (training
 * diverged), opacities that are effectively zero (everything is transparent),
 * scales so small they cover no pixels, or an extent so lopsided that the
 * camera frames empty space.
 *
 * With no argument it inspects the most recent conversion in the library, which
 * is almost always the one being asked about:
 *
 *   npm run inspect                    # newest conversion
 *   npm run inspect <asset-id>         # a specific one
 *   npm run inspect path/to/file.ply   # any file, e.g. a download
 *   npm run inspect --list             # what is available
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { decodePly } from '../server/pipeline/ply.js';
import { robustBounds, boundsOf } from '../server/pipeline/splat.js';
import { ASSETS_DIR, DB_FILE } from '../server/config.js';

const c = process.stdout.isTTY
  ? { dim: (s) => `\x1b[2m${s}\x1b[0m`, bold: (s) => `\x1b[1m${s}\x1b[0m`,
      red: (s) => `\x1b[31m${s}\x1b[0m`, green: (s) => `\x1b[32m${s}\x1b[0m`,
      yellow: (s) => `\x1b[33m${s}\x1b[0m` }
  : new Proxy({}, { get: () => (s) => s });

function percentiles(values) {
  const finite = Array.from(values).filter(Number.isFinite).sort((a, b) => a - b);
  if (!finite.length) return null;
  const at = (p) => finite[Math.min(finite.length - 1, Math.floor((finite.length - 1) * p))];
  return { min: finite[0], p05: at(0.05), median: at(0.5), p95: at(0.95), max: finite[finite.length - 1] };
}

const fmt = (n) => (Number.isFinite(n) ? (Math.abs(n) >= 1000 || (Math.abs(n) < 0.001 && n !== 0)
  ? n.toExponential(2) : n.toFixed(4)) : String(n));

function show(label, p) {
  if (!p) return console.log(`  ${label.padEnd(12)} ${c.red('no finite values at all')}`);
  console.log(`  ${label.padEnd(12)} min ${fmt(p.min)}  p05 ${fmt(p.p05)}  median ${fmt(p.median)}`
    + `  p95 ${fmt(p.p95)}  max ${fmt(p.max)}`);
}

export function inspect(cloud) {
  const problems = [];
  const n = cloud.count;

  // Non-finite positions are the failure that hides best: the header count is
  // still right, the renderer still runs, and the camera framing collapses.
  let badPos = 0;
  for (let i = 0; i < n * 3; i++) if (!Number.isFinite(cloud.positions[i])) { badPos += 1; }
  let badScale = 0;
  for (let i = 0; i < n * 3; i++) if (!Number.isFinite(cloud.scales[i])) { badScale += 1; }
  let badOpacity = 0;
  for (let i = 0; i < n; i++) if (!Number.isFinite(cloud.opacities[i])) { badOpacity += 1; }

  const visible = Array.from(cloud.opacities).filter((o) => o > 0.02).length;
  const scaleP = percentiles(cloud.scales);
  const opacityP = percentiles(cloud.opacities);

  console.log(c.bold(`\n${n.toLocaleString()} gaussians\n`));

  console.log(c.bold('finite values'));
  const posLine = `${badPos} of ${n * 3} position components are not finite`;
  console.log(`  ${badPos ? c.red(posLine) : c.green(`all ${n * 3} position components are finite`)}`);
  if (badScale) console.log(`  ${c.red(`${badScale} scale components are not finite`)}`);
  if (badOpacity) console.log(`  ${c.red(`${badOpacity} opacities are not finite`)}`);

  console.log(c.bold('\ndistributions'));
  show('opacity', opacityP);
  show('scale', scaleP);

  console.log(c.bold('\nextent'));
  const rb = robustBounds(cloud);
  const raw = boundsOf(cloud);
  console.log(`  robust centre  [${rb.center.map(fmt).join(', ')}]  radius ${fmt(rb.radius)}`);
  console.log(`  full extent    [${raw.min.map(fmt).join(', ')}] .. [${raw.max.map(fmt).join(', ')}]`);
  const span = [0, 1, 2].map((k) => rb.max[k] - rb.min[k]);
  const aspect = Math.max(...span) / Math.max(1e-9, Math.min(...span));
  console.log(`  axis spans     [${span.map(fmt).join(', ')}]  longest/shortest ${fmt(aspect)}`);

  // --- verdicts -------------------------------------------------------------
  if (badPos) {
    problems.push('Positions contain NaN or Infinity, which means training diverged. '
      + 'The viewer frames the scene from these values, so the camera ends up nowhere '
      + 'and the render is blank even though the gaussian count looks right.');
  }
  if (!rb.center.every(Number.isFinite) || !Number.isFinite(rb.radius)) {
    problems.push('The framing bounds are not finite, so the camera cannot be placed. '
      + 'This is what turns the viewport black.');
  }
  if (visible === 0) {
    problems.push('Every gaussian is effectively transparent (opacity <= 0.02). '
      + 'They are being drawn and contribute nothing.');
  } else if (visible < n * 0.02) {
    problems.push(`Only ${visible} of ${n} gaussians (${(100 * visible / n).toFixed(1)}%) have `
      + 'opacity above 0.02, so almost nothing is visible.');
  }
  if (scaleP && scaleP.p95 < rb.radius * 1e-4) {
    problems.push('Scales are minuscule relative to the scene, so each gaussian covers '
      + 'far less than a pixel and the render is empty at any sane camera distance.');
  }
  if (Number.isFinite(aspect) && aspect > 50) {
    problems.push(`The scene is ${fmt(aspect)}x longer on one axis than another. Forward `
      + 'motion captures (walking straight down a road) do this: there is almost no '
      + 'parallax along the direction of travel, so depth is barely constrained and '
      + 'framing the result is unreliable.');
  }

  console.log('');
  if (problems.length) {
    console.log(c.bold(c.red('problems found')));
    for (const p of problems) console.log(`  ${c.yellow('*')} ${p}\n`);
  } else {
    console.log(c.green('Nothing structurally wrong: finite values, visible opacity, '
      + 'sane scales and extent.'));
    console.log(c.dim('If it still renders blank the problem is in the viewer or the '
      + 'camera, not the model.\n'));
  }
  return { count: n, badPos, badScale, badOpacity, visible, problems };
}

/** Every conversion in the library that produced a model, newest first. */
export function listConversions(assetsDir = ASSETS_DIR, dbFile = DB_FILE) {
  if (!fs.existsSync(assetsDir)) return [];
  let names = {};
  try {
    const db = JSON.parse(fs.readFileSync(dbFile, 'utf8'));
    for (const a of (Array.isArray(db) ? db : db.assets || [])) {
      if (a?.id) names[a.id] = a.name || '';
    }
  } catch { /* the library index is a convenience here, not a requirement */ }

  const out = [];
  for (const id of fs.readdirSync(assetsDir)) {
    const ply = path.join(assetsDir, id, 'output', 'point_cloud.ply');
    let stat;
    try { stat = fs.statSync(ply); } catch { continue; }
    out.push({ id, ply, mtime: stat.mtimeMs, bytes: stat.size, name: names[id] || '' });
  }
  return out.sort((a, b) => b.mtime - a.mtime);
}

function main() {
  const arg = process.argv[2];

  if (arg === '--list' || arg === '-l') {
    const all = listConversions();
    if (!all.length) {
      console.error(`no conversions found under ${ASSETS_DIR}`);
      process.exit(2);
    }
    for (const a of all) {
      console.log(`${a.id}  ${(a.bytes / 1024).toFixed(0).padStart(7)} KB  `
        + `${new Date(a.mtime).toLocaleString()}  ${a.name}`);
    }
    return;
  }

  let target = arg;

  // No argument: take the newest conversion. Asking someone to assemble a path
  // into a directory they have never opened is where this goes wrong.
  if (!target) {
    const [newest] = listConversions();
    if (!newest) {
      console.error(`No conversions found under ${ASSETS_DIR}.`);
      console.error('Run this from the project directory, or pass a .ply file directly.');
      process.exit(2);
    }
    target = newest.ply;
    console.log(c.dim(`newest conversion: ${newest.id}${newest.name ? `  "${newest.name}"` : ''}`));
  } else if (!fs.existsSync(target)) {
    // An asset id rather than a path.
    const match = listConversions().find((a) => a.id === target);
    if (match) {
      target = match.ply;
    } else {
      console.error(`No such file, and no conversion with id "${arg}".`);
      console.error('Run "npm run inspect --list" to see what is available.');
      process.exit(2);
    }
  }

  const buffer = fs.readFileSync(target);
  console.log(c.dim(`${path.resolve(target)}  (${(buffer.length / 1024).toFixed(0)} KB)`));
  const cloud = decodePly(buffer);
  const { problems } = inspect(cloud);
  process.exit(problems.length ? 1 : 0);
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  main();
}
