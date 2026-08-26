import test from 'node:test';
import assert from 'node:assert/strict';
import { encodePng, decodePng } from '../server/pipeline/png.js';
import { encodePly, decodePly, parsePlyHeader } from '../server/pipeline/ply.js';
import {
  createCloud, encodeSplatBuffer, decodeSplatBuffer, concatClouds, filterCloud,
  boundsOf, robustBounds, normaliseCloud, quatFromBasis, quatToMatrix, sigmoid, logit,
} from '../server/pipeline/splat.js';

function sampleCloud(count = 50) {
  const cloud = createCloud(count);
  for (let i = 0; i < count; i++) {
    cloud.positions[i * 3 + 0] = Math.sin(i) * 2;
    cloud.positions[i * 3 + 1] = Math.cos(i * 1.7);
    cloud.positions[i * 3 + 2] = (i / count) * 3 - 1.5;
    cloud.scales[i * 3 + 0] = 0.01 + (i % 7) * 0.003;
    cloud.scales[i * 3 + 1] = 0.02;
    cloud.scales[i * 3 + 2] = 0.005;
    const n = Math.hypot(1, i % 3, i % 5, i % 2) || 1;
    cloud.rotations[i * 4 + 0] = 1 / n;
    cloud.rotations[i * 4 + 1] = (i % 3) / n;
    cloud.rotations[i * 4 + 2] = (i % 5) / n;
    cloud.rotations[i * 4 + 3] = (i % 2) / n;
    cloud.colors[i * 3 + 0] = (i % 10) / 10;
    cloud.colors[i * 3 + 1] = 0.5;
    cloud.colors[i * 3 + 2] = 1 - (i % 10) / 10;
    cloud.opacities[i] = 0.1 + ((i % 9) / 10);
  }
  return cloud;
}

test('PNG survives an encode/decode round trip', () => {
  const width = 23;
  const height = 17;
  const data = Buffer.alloc(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4 + 0] = i % 256;
    data[i * 4 + 1] = (i * 7) % 256;
    data[i * 4 + 2] = (i * 13) % 256;
    data[i * 4 + 3] = 255;
  }
  const decoded = decodePng(encodePng({ width, height, data }));
  assert.equal(decoded.width, width);
  assert.equal(decoded.height, height);
  assert.deepEqual(Buffer.from(decoded.data), data);
});

test('decodePng rejects data that is not a PNG', () => {
  assert.throws(() => decodePng(Buffer.from('definitely not a png')), /not a PNG/);
});

test('PLY round trip preserves positions exactly and other fields closely', () => {
  const cloud = sampleCloud();
  const back = decodePly(encodePly(cloud));

  assert.equal(back.count, cloud.count);
  for (let i = 0; i < cloud.count * 3; i++) {
    assert.equal(back.positions[i], cloud.positions[i], `position ${i}`);
  }
  for (let i = 0; i < cloud.count; i++) {
    // Opacity and scale go through logit/log and back, so allow float slop.
    assert.ok(Math.abs(back.opacities[i] - cloud.opacities[i]) < 1e-5, 'opacity');
    for (let k = 0; k < 3; k++) {
      assert.ok(Math.abs(back.scales[i * 3 + k] - cloud.scales[i * 3 + k]) < 1e-6, 'scale');
      assert.ok(Math.abs(back.colors[i * 3 + k] - cloud.colors[i * 3 + k]) < 1e-5, 'colour');
    }
  }
});

test('PLY header advertises the properties 3DGS viewers look for', () => {
  const header = parsePlyHeader(encodePly(sampleCloud(3)));
  assert.equal(header.format, 'binary_little_endian');
  const names = header.elements.find((e) => e.name === 'vertex').properties.map((p) => p.name);
  for (const required of ['x', 'y', 'z', 'f_dc_0', 'opacity', 'scale_0', 'rot_0', 'rot_3']) {
    assert.ok(names.includes(required), `missing ${required}`);
  }
});

test('a plain point cloud PLY decodes into renderable gaussians', () => {
  // COLMAP's sparse output has no gaussian fields; every point should still
  // come back with a usable scale, rotation and opacity.
  const ascii = [
    'ply', 'format ascii 1.0', 'element vertex 2',
    'property float x', 'property float y', 'property float z',
    'property uchar red', 'property uchar green', 'property uchar blue',
    'end_header', '1 2 3 255 0 0', '4 5 6 0 255 0', '',
  ].join('\n');
  const cloud = decodePly(Buffer.from(ascii, 'latin1'));
  assert.equal(cloud.count, 2);
  assert.deepEqual([...cloud.positions.slice(0, 3)], [1, 2, 3]);
  assert.ok(cloud.scales[0] > 0, 'points need a non-zero scale to render');
  assert.equal(cloud.opacities[0], 1);
  assert.ok(Math.abs(cloud.colors[0] - 1) < 1e-6);
});

test('.splat round trip keeps positions and scales exact', () => {
  const cloud = sampleCloud(30);
  const back = decodeSplatBuffer(encodeSplatBuffer(cloud));
  assert.equal(back.count, cloud.count);
  for (let i = 0; i < cloud.count * 3; i++) {
    assert.equal(back.positions[i], cloud.positions[i]);
    assert.equal(back.scales[i], cloud.scales[i]);
  }
  // Colour, opacity and rotation are quantised to bytes.
  for (let i = 0; i < cloud.count; i++) {
    assert.ok(Math.abs(back.opacities[i] - cloud.opacities[i]) <= 1 / 255);
  }
});

test('.splat encoding is exactly 32 bytes per gaussian', () => {
  assert.equal(encodeSplatBuffer(sampleCloud(11)).length, 11 * 32);
});

test('sigmoid and logit invert each other', () => {
  for (const p of [0.01, 0.25, 0.5, 0.9, 0.999]) {
    assert.ok(Math.abs(sigmoid(logit(p)) - p) < 1e-9);
  }
});

test('concat and filter keep clouds consistent', () => {
  const a = sampleCloud(5);
  const b = sampleCloud(7);
  const joined = concatClouds([a, b]);
  assert.equal(joined.count, 12);
  assert.equal(joined.positions[5 * 3], b.positions[0]);

  const evens = filterCloud(joined, (i) => i % 2 === 0);
  assert.equal(evens.count, 6);
  assert.equal(evens.positions[0], joined.positions[0]);
  assert.equal(evens.positions[3], joined.positions[6]);
});

test('bounds and normalisation centre a cloud on the origin', () => {
  const cloud = sampleCloud(200);
  normaliseCloud(cloud, 1);
  const { center, radius } = robustBounds(cloud);
  for (const c of center) assert.ok(Math.abs(c) < 1e-5, 'centred');
  assert.ok(Math.abs(radius - 1) < 1e-5, 'unit radius');
});

test('robustBounds ignores a handful of far outliers', () => {
  const cloud = createCloud(100);
  for (let i = 0; i < 100; i++) cloud.positions[i * 3] = (i / 99) * 2 - 1;
  cloud.positions[0] = -500;   // strays that would wreck a min/max fit
  cloud.positions[99 * 3] = 500;
  assert.ok(boundsOf(cloud).radius > 100, 'plain bounds follow the outliers');
  assert.ok(robustBounds(cloud).radius < 2, 'robust bounds do not');
});

test('quatFromBasis and quatToMatrix are inverses', () => {
  const right = [0, 0, -1];
  const up = [0, 1, 0];
  const forward = [1, 0, 0];
  const m = quatToMatrix(quatFromBasis(right, up, forward));
  // Columns of the rotation matrix are the basis vectors we started from.
  const col = (k) => [m[k], m[3 + k], m[6 + k]];
  assert.deepEqual(col(0).map((v) => Math.round(v)), right);
  assert.deepEqual(col(1).map((v) => Math.round(v)), up);
  assert.deepEqual(col(2).map((v) => Math.round(v)), forward);
});
