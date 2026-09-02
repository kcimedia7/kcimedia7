import test from 'node:test';
import assert from 'node:assert/strict';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
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

test('the inspector finds conversions without being handed a path', async () => {
  // Assembling a path into a directory you have never opened is where
  // diagnosing a bad conversion actually fails, so the tool resolves the
  // library itself and reports the newest conversion first.
  const { listConversions } = await import('../ops/inspect-ply.mjs');
  const dir = await fsp.mkdtemp(path.join(os.tmpdir(), 'splat-inspect-'));
  const assets = path.join(dir, 'assets');
  for (const [id, when] of [['aaa', 1000], ['bbb', 3000], ['ccc', 2000]]) {
    const out = path.join(assets, id, 'output');
    await fsp.mkdir(out, { recursive: true });
    const ply = path.join(out, 'point_cloud.ply');
    await fsp.writeFile(ply, 'ply\n');
    await fsp.utimes(ply, new Date(when), new Date(when));
  }
  // An asset directory with no model must not appear as a conversion.
  await fsp.mkdir(path.join(assets, 'ddd', 'frames'), { recursive: true });
  await fsp.writeFile(path.join(dir, 'library.json'),
    JSON.stringify([{ id: 'bbb', name: 'the street' }]));

  const found = listConversions(assets, path.join(dir, 'library.json'));
  assert.deepEqual(found.map((f) => f.id), ['bbb', 'ccc', 'aaa'], 'newest must come first');
  assert.equal(found[0].name, 'the street', 'names come from the library index');
  assert.equal(found[1].name, '', 'a missing name is not an error');

  // A missing library index must not stop the listing.
  const noDb = listConversions(assets, path.join(dir, 'gone.json'));
  assert.equal(noDb.length, 3);
  assert.equal(listConversions(path.join(dir, 'nothing-here')).length, 0);

  await fsp.rm(dir, { recursive: true, force: true });
});

test('a single panorama is refused before the solver wastes minutes on it', async () => {
  // One 360 photo has one optical centre, so no two of its views have a
  // baseline and depth cannot be triangulated. COLMAP does fail on it -- but
  // only after minutes of feature matching, and it reports too little overlap
  // or texture, which is the one explanation that is definitely wrong here.
  const { refuseSingleViewpoint } = await import('../server/pipeline/index.js');

  assert.throws(() => refuseSingleViewpoint('pano', 6), /single 360 photo/);
  assert.throws(() => refuseSingleViewpoint('pano', 6), /no parallax/);
  assert.throws(() => refuseSingleViewpoint('pano', 3), /single 360 photo/);

  // Two panoramas have a baseline between them, so they are allowed through.
  assert.doesNotThrow(() => refuseSingleViewpoint('pano', 12));
  // And ordinary captures are never subject to this, however few frames.
  assert.doesNotThrow(() => refuseSingleViewpoint('photos', 4));
  assert.doesNotThrow(() => refuseSingleViewpoint('video', 6));
});

test('a lone panorama is routed to depth inference rather than to the solver', async () => {
  // Solving it would spend minutes arriving at the same impossibility every
  // time, and inference is the only thing that produces anything from one
  // viewpoint.
  const { chooseBackend } = await import('../server/pipeline/index.js');
  const caps = { backend: 'gaussian' };

  assert.equal(chooseBackend({ kind: 'pano' }, caps, 6), 'depth');
  assert.equal(chooseBackend({ kind: 'pano' }, caps, 3), 'depth');

  // Two panoramas have a baseline, so they go to the solver like anything else.
  assert.equal(chooseBackend({ kind: 'pano' }, caps, 12), 'gaussian');
  assert.equal(chooseBackend({ kind: 'photos' }, caps, 4), 'gaussian');
  assert.equal(chooseBackend({ kind: 'video' }, caps, 6), 'gaussian');

  // An explicit choice always wins, in both directions.
  assert.equal(chooseBackend({ kind: 'pano', backend: 'preview' }, caps, 6), 'preview');
  assert.equal(chooseBackend({ kind: 'photos', backend: 'depth' }, caps, 40), 'depth');
  assert.equal(chooseBackend({ kind: 'pano', backend: 'auto' }, caps, 6), 'depth');

  // With nothing detected at all there is still a sane answer.
  assert.equal(chooseBackend({}, {}, 0), 'preview');
});

test('the depth stage plan covers the whole progress bar', async () => {
  const plan = stagePlan('depth');
  const total = plan.reduce((sum, s) => sum + s.weight, 0);
  assert.ok(Math.abs(total - 1) < 1e-9, `weights sum to ${total}`);
  assert.deepEqual(plan.map((s) => s.id), ['ingest', 'estimate', 'export']);
});

test('depth progress follows the estimator through its six views', async () => {
  const { parseProgress } = await import('../server/pipeline/depth.js');

  assert.match(parseProgress('depth: reading 6 views from /x').label, /Reading/);
  // The first run downloads the model and says nothing for a long time, so the
  // bar has to show that it is loading rather than appear stuck at zero.
  assert.match(parseProgress('depth: loading depth-anything/x on cuda').label, /Loading/);

  const third = parseProgress('depth: estimating back (3/6)');
  const fifth = parseProgress('depth: estimating up (5/6)');
  assert.ok(fifth.fraction > third.fraction, 'progress must advance with the views');
  assert.match(third.label, /back/);
  assert.ok(parseProgress('wrote /x/point_cloud.ply (500 gaussians, 1 bytes)').fraction === 1);
  assert.equal(parseProgress('something else entirely'), null);
});
