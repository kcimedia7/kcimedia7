import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { parseProgress } from '../server/pipeline/gaussian.js';
import { stagePlan } from '../server/pipeline/index.js';

/**
 * The `gaussian` backend hands reconstruction to the Python trainer and reads
 * its stdout back as progress. These check the seam between the two: the stage
 * plan, and the parsing of the trainer's own output.
 */

test('the gaussian stage plan covers the whole progress bar', () => {
  const plan = stagePlan('gaussian');
  assert.equal(plan.reduce((n, s) => n + s.weight, 0).toFixed(6), '1.000000');
  assert.deepEqual(plan.map((s) => s.id), ['ingest', 'train', 'export']);
});

test('trainer output maps onto monotonically rising progress', () => {
  const lines = [
    'sfm: extracting features',
    'sfm: matching (exhaustive)',
    'sfm: incremental mapping',
    'sfm: registered 24 images, 1625 points',
    'iter 1/600  loss 0.1974  l1 0.1108  psnr 16.83dB  gaussians 1627',
    'iter 300/600  loss 0.1083  l1 0.0572  psnr 21.38dB  gaussians 7625',
    'iter 600/600  loss 0.0838  l1 0.0345  psnr 23.86dB  gaussians 12000',
    'final PSNR over 24 training views: 23.10 dB',
  ];
  let previous = -1;
  for (const line of lines) {
    const update = parseProgress(line, 600);
    assert.ok(update, `no progress parsed from: ${line}`);
    assert.ok(update.fraction > previous, `progress went backwards at: ${line}`);
    assert.ok(update.fraction <= 1);
    previous = update.fraction;
  }
});

test('progress lines carry a human-readable status', () => {
  assert.match(parseProgress('sfm: registered 24 images, 1625 points', 600).label,
    /Solved 24 camera poses/);
  const training = parseProgress('iter 300/600  loss 0.1083  l1 0.0572  psnr 21.38dB  gaussians 7625', 600);
  assert.match(training.label, /7,625 splats/);
  assert.match(training.label, /loss 0\.1083/);
});

test('unrelated trainer chatter is ignored rather than misparsed', () => {
  for (const line of ['', 'some warning from a library', 'Traceback (most recent call last):']) {
    assert.equal(parseProgress(line, 600), null);
  }
});

test('the trainer CLI exposes the flags the backend invokes', () => {
  // The Node side builds this argv; a rename on the Python side must fail here
  // rather than at conversion time.
  let help;
  try {
    help = execFileSync('python3', ['-m', 'splatworks_train.train', '--help'], {
      cwd: new URL('../trainer/', import.meta.url).pathname,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    });
  } catch {
    return; // trainer dependencies absent; detection covers that path
  }
  for (const flag of ['--images', '--output', '--work', '--iterations',
                      '--resolution', '--max-gaussians', '--matcher']) {
    assert.ok(help.includes(flag), `trainer CLI is missing ${flag}`);
  }
});
