#!/usr/bin/env node
/**
 * Collect everything needed to diagnose a conversion into one paste-able block.
 *
 * Debugging a bad reconstruction otherwise takes several rounds of "run this,
 * paste that", and each round can fail on its own terms -- wrong directory,
 * placeholder typed literally, output truncated. This asks once.
 *
 *   npm run report          # newest conversion
 *   npm run report <id>     # a specific one
 *
 * Only named environment variables are printed, never the whole environment,
 * which is where credentials live.
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { decodePly } from '../server/pipeline/ply.js';
import { DB_FILE, ASSETS_DIR } from '../server/config.js';
import { inspect, listConversions } from './inspect-ply.mjs';

const line = (s = '') => console.log(s);
const rule = (title) => { line(); line(`--- ${title} ${'-'.repeat(Math.max(0, 58 - title.length))}`); };

function tryRun(cmd, args) {
  try {
    return execFileSync(cmd, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
  } catch {
    return null;
  }
}

function environment() {
  rule('environment');
  line(`os        ${os.type()} ${os.release()} ${os.arch()}`);
  line(`cpus      ${os.cpus().length}`);
  line(`memory    ${(os.totalmem() / 1024 ** 3).toFixed(1)} GB`);
  line(`node      ${process.version}`);

  const python = tryRun('python', ['--version']) || tryRun('python3', ['--version']);
  line(`python    ${python || 'not found'}`);

  // Whether torch can actually reach the GPU is the single most useful fact
  // when a conversion was slow or produced a poor model.
  const torch = tryRun('python', ['-c',
    'import torch;print(torch.__version__, torch.cuda.is_available(), '
    + 'torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")'])
    || tryRun('python3', ['-c',
      'import torch;print(torch.__version__, torch.cuda.is_available(), '
      + 'torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")']);
  line(`torch     ${torch || 'not installed (the gaussian backend cannot run)'}`);

  const pycolmap = tryRun('python', ['-c', 'import pycolmap;print(pycolmap.__version__)'])
    || tryRun('python3', ['-c', 'import pycolmap;print(pycolmap.__version__)']);
  line(`pycolmap  ${pycolmap || 'not installed'}`);

  const smi = tryRun('nvidia-smi',
    ['--query-gpu=name,memory.total,driver_version', '--format=csv,noheader']);
  line(`gpu       ${smi || 'nvidia-smi not found'}`);

  for (const key of ['SPLAT_BACKEND', 'SPLAT_TRAIN_DEVICE', 'SPLAT_DATA_DIR',
    'SPLAT_TRAINER_CMD', 'SPLAT_CONCURRENCY']) {
    if (process.env[key]) line(`${key.padEnd(20)} ${process.env[key]}`);
  }

  const commit = tryRun('git', ['rev-parse', '--short', 'HEAD']);
  const branch = tryRun('git', ['rev-parse', '--abbrev-ref', 'HEAD']);
  line(`commit    ${commit || 'unknown'} on ${branch || 'unknown'}`);
}

function loadAssets() {
  try {
    const db = JSON.parse(fs.readFileSync(DB_FILE, 'utf8'));
    return Array.isArray(db) ? db : db.assets || [];
  } catch {
    return [];
  }
}

function main() {
  const wanted = process.argv[2];
  environment();

  const assets = loadAssets();
  rule('library');
  if (!assets.length) {
    line(`no library at ${DB_FILE}`);
  } else {
    for (const a of assets) {
      line(`${a.id}  ${String(a.status).padEnd(9)} ${String(a.kind).padEnd(7)} `
        + `${String(a.backend || '-').padEnd(9)} ${String(a.result?.splatCount ?? '-').padStart(9)} `
        + `${a.name || ''}`);
      if (a.error) line(`    error: ${a.error}`);
    }
  }

  const asset = wanted
    ? assets.find((a) => a.id === wanted)
    : assets.slice().sort((a, b) =>
      String(b.createdAt).localeCompare(String(a.createdAt)))[0];

  if (!asset) {
    rule('conversion');
    line(wanted ? `no conversion with id ${wanted}` : 'no conversions to report on');
    return;
  }

  rule(`conversion ${asset.id}`);
  line(`name       ${asset.name}`);
  line(`status     ${asset.status}  (${asset.stage})`);
  line(`kind       ${asset.kind}`);
  line(`backend    ${asset.backend || '-'}`);
  line(`frames     ${asset.source?.frameCount ?? '-'}`);
  line(`created    ${asset.createdAt}`);
  line(`finished   ${asset.finishedAt || '-'}`);
  if (asset.error) line(`error      ${asset.error}`);
  line(`settings   ${JSON.stringify(asset.settings)}`);
  line(`result     ${JSON.stringify(asset.result?.stats ?? null)}`);

  rule('log (last 80 lines)');
  for (const entry of (asset.log || []).slice(-80)) {
    line(typeof entry === 'string' ? entry : entry.line);
  }

  rule('model');
  const ply = path.join(ASSETS_DIR, asset.id, 'output', 'point_cloud.ply');
  if (!fs.existsSync(ply)) {
    line(`no model written at ${ply}`);
    const others = listConversions();
    if (others.length) line(`conversions that did write one: ${others.map((o) => o.id).join(', ')}`);
    return;
  }
  const buffer = fs.readFileSync(ply);
  line(`${ply}  (${(buffer.length / 1024).toFixed(0)} KB)`);
  try {
    inspect(decodePly(buffer));
  } catch (err) {
    line(`the model could not be decoded: ${err.message}`);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
  main();
}
