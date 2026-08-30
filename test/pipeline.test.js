import test from 'node:test';
import assert from 'node:assert/strict';
import { reconstructPreview } from '../server/pipeline/preview.js';
import { pickEvenly, stagePlan, requireRenderableCloud } from '../server/pipeline/index.js';
import { buildTrainerArgv, tokenize } from '../server/pipeline/colmap.js';
import { sanitiseSettings } from '../server/api.js';
import { boundsOf, createCloud, robustBounds } from '../server/pipeline/splat.js';

/** A frame with a bright subject on a flat backdrop. */
function frame(width = 96, height = 72, hue = 0) {
  const data = Buffer.alloc(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      data[i] = 40; data[i + 1] = 42; data[i + 2] = 48; data[i + 3] = 255;
      const d = Math.hypot(x - width / 2, y - height / 2);
      if (d < width / 4) {
        data[i] = 220 - hue; data[i + 1] = 90 + hue; data[i + 2] = 180;
      }
    }
  }
  return { width, height, data };
}

test('preview reconstruction builds a normalised, non-empty cloud', () => {
  const frames = [0, 1, 2, 3, 4, 5].map((i) => frame(96, 72, i * 20));
  const { cloud, stats } = reconstructPreview(frames, { grid: 64 });

  assert.ok(cloud.count > 100, `expected a substantial cloud, got ${cloud.count}`);
  assert.equal(stats.frames, 6);
  assert.equal(stats.splats, cloud.count);

  // normaliseCloud should have centred it in a unit-ish volume.
  const { center, radius } = boundsOf(cloud);
  for (const c of center) assert.ok(Math.abs(c) < 1.5, 'roughly centred');
  assert.ok(radius > 0.2 && radius < 4, `radius ${radius} should be order 1`);
});

test('every reconstructed gaussian is finite and in range', () => {
  const { cloud } = reconstructPreview([frame(), frame(96, 72, 40)], { grid: 48 });
  for (let i = 0; i < cloud.count; i++) {
    for (let k = 0; k < 3; k++) {
      assert.ok(Number.isFinite(cloud.positions[i * 3 + k]), 'position is finite');
      assert.ok(cloud.scales[i * 3 + k] > 0, 'scale is positive');
      const c = cloud.colors[i * 3 + k];
      assert.ok(c >= 0 && c <= 1, `colour ${c} in range`);
    }
    assert.ok(cloud.opacities[i] > 0 && cloud.opacities[i] <= 1, 'opacity in range');
    const q = [0, 1, 2, 3].map((k) => cloud.rotations[i * 4 + k]);
    assert.ok(Math.abs(Math.hypot(...q) - 1) < 1e-3, 'rotation is a unit quaternion');
  }
});

test('reconstruction is deterministic for the same input', () => {
  const frames = [frame(), frame(96, 72, 30)];
  const a = reconstructPreview(frames, { grid: 48 }).cloud;
  const b = reconstructPreview(frames, { grid: 48 }).cloud;
  assert.equal(a.count, b.count);
  assert.deepEqual(Array.from(a.positions.slice(0, 60)), Array.from(b.positions.slice(0, 60)));
});

test('a single frame still produces a viewable relief', () => {
  const { cloud } = reconstructPreview([frame()], { grid: 48 });
  assert.ok(cloud.count > 50);
});

test('reconstruction needs at least one frame', () => {
  assert.throws(() => reconstructPreview([], {}), /at least one frame/);
});

test('a 360 degree orbit does not stack the last frame onto the first', () => {
  // Closed arcs must step by n, not n-1, or two cameras coincide.
  const frames = [0, 1, 2, 3].map((i) => frame(64, 48, i * 30));
  const { cloud } = reconstructPreview(frames, { grid: 40, arcDeg: 360 });
  assert.ok(cloud.count > 0);
  const { min, max } = boundsOf(cloud);
  // Cameras all round the subject means real extent on both horizontal axes.
  assert.ok(max[0] - min[0] > 0.5, 'x extent');
  assert.ok(max[2] - min[2] > 0.5, 'z extent');
});

test('pickEvenly keeps the ends and thins the middle', () => {
  const items = Array.from({ length: 100 }, (_, i) => i);
  const picked = pickEvenly(items, 10);
  assert.equal(picked.length, 10);
  assert.equal(picked[0], 0);
  assert.equal(picked.at(-1), 99);
  assert.deepEqual(picked, [...picked].sort((a, b) => a - b));

  assert.deepEqual(pickEvenly([1, 2, 3], 10), [1, 2, 3]);
});

test('stage plans cover the whole progress bar', () => {
  for (const backend of ['preview', 'colmap']) {
    const total = stagePlan(backend).reduce((n, s) => n + s.weight, 0);
    assert.ok(Math.abs(total - 1) < 1e-9, `${backend} weights sum to 1`);
  }
});

test('trainer command templates expand their placeholders', () => {
  const { cmd, args } = buildTrainerArgv(
    'python train.py -s {source} -m {output} --iterations {iterations}',
    { source: '/data/ds', output: '/data/out', images: '/data/ds/images', iterations: 7000 },
  );
  assert.equal(cmd, 'python');
  assert.deepEqual(args, ['train.py', '-s', '/data/ds', '-m', '/data/out', '--iterations', '7000']);
});

test('quoted paths in a trainer template stay one argument', () => {
  assert.deepEqual(
    tokenize('python "/opt/gaussian splatting/train.py" -s {source}'),
    ['python', '/opt/gaussian splatting/train.py', '-s', '{source}'],
  );
});

test('an unknown placeholder in a trainer template is reported', () => {
  assert.throws(
    () => buildTrainerArgv('train --thing {nonsense}', { source: 'a', output: 'b', images: 'c', iterations: 1 }),
    /unknown placeholder/,
  );
});

test('settings from the client are clamped and filtered', () => {
  const s = sanitiseSettings({
    backend: 'rm -rf /',
    iterations: 1e9,
    detail: -50,
    arcDeg: 720,
    splatScale: 'big',
    injected: 'should not survive',
  });
  assert.equal(s.backend, undefined);
  assert.equal(s.iterations, 60_000);
  assert.equal(s.detail, 24);
  assert.equal(s.arcDeg, 360);
  assert.equal(s.splatScale, undefined);
  assert.equal(s.injected, undefined);

  assert.equal(sanitiseSettings({ backend: 'preview' }).backend, 'preview');
});

test('a diverged reconstruction fails instead of reporting success', () => {
  // Training that diverges still writes a well-formed PLY: the count, the file
  // size and the status all look right, and the viewer draws nothing because
  // the values are NaN. Presenting that as a finished conversion sends the user
  // looking for a renderer fault that does not exist.
  const cloud = createCloud(1000);
  for (let i = 0; i < 1000; i++) {
    cloud.positions[i * 3] = i * 0.01;
    for (let k = 0; k < 3; k++) cloud.scales[i * 3 + k] = 0.02;
    cloud.opacities[i] = 0.8;
    cloud.rotations[i * 4] = 1;
  }
  // Well past the tolerance: the surviving gaussians are not trustworthy either.
  for (let i = 0; i < 50; i++) cloud.positions[i * 3 + 1] = NaN;
  assert.throws(() => requireRenderableCloud(cloud), /diverged/);
});

test('a handful of stray gaussians is dropped rather than failing the run', () => {
  // One bad splat in thousands is not worth discarding half an hour of compute
  // for, as long as it cannot poison the framing.
  const cloud = createCloud(1000);
  for (let i = 0; i < 1000; i++) {
    cloud.positions[i * 3] = i * 0.01;
    for (let k = 0; k < 3; k++) cloud.scales[i * 3 + k] = 0.02;
    cloud.opacities[i] = 0.8;
    cloud.rotations[i * 4] = 1;
  }
  cloud.positions[7 * 3 + 2] = Infinity;
  cloud.opacities[11] = NaN;
  const cleaned = requireRenderableCloud(cloud);
  assert.equal(cleaned.count, 998);
  for (let i = 0; i < cleaned.count; i++) {
    assert.ok(Number.isFinite(cleaned.positions[i * 3 + 2]));
    assert.ok(Number.isFinite(cleaned.opacities[i]));
  }
  // And the framing it feeds must come back placeable.
  const b = robustBounds(cleaned);
  assert.ok(Number.isFinite(b.radius) && b.center.every(Number.isFinite));
});

test('an empty reconstruction is reported as such', () => {
  assert.throws(() => requireRenderableCloud(createCloud(0)), /no gaussians/);
});

test('bounds stay placeable when positions are not finite', () => {
  // This is the specific failure that turns the viewport black while the
  // gaussian count still reads correctly, so it is checked directly.
  const cloud = createCloud(200);
  for (let i = 0; i < 200; i++) {
    for (let k = 0; k < 3; k++) cloud.positions[i * 3 + k] = i;
  }
  for (let i = 0; i < 60; i++) cloud.positions[i * 3] = NaN; // 30%, past the percentile
  for (const b of [robustBounds(cloud), boundsOf(cloud)]) {
    assert.ok(Number.isFinite(b.radius), 'radius must stay finite');
    assert.ok(b.center.every(Number.isFinite), 'centre must stay finite');
  }
  // Even with nothing finite at all, the camera must still be placeable.
  const allBad = createCloud(10);
  allBad.positions.fill(NaN);
  for (const b of [robustBounds(allBad), boundsOf(allBad)]) {
    assert.ok(Number.isFinite(b.radius) && b.center.every(Number.isFinite));
  }
});
