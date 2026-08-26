import test from 'node:test';
import assert from 'node:assert/strict';
import { applyEdits, normaliseEdits, isIdentity } from '../server/edits.js';
import { createCloud } from '../server/pipeline/splat.js';

function grid(n = 5) {
  // A cube of points from -1 to 1 on each axis, opacity ramping with index.
  const cloud = createCloud(n * n * n);
  let i = 0;
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) {
      for (let z = 0; z < n; z++) {
        cloud.positions[i * 3 + 0] = (x / (n - 1)) * 2 - 1;
        cloud.positions[i * 3 + 1] = (y / (n - 1)) * 2 - 1;
        cloud.positions[i * 3 + 2] = (z / (n - 1)) * 2 - 1;
        cloud.scales[i * 3 + 0] = 0.05;
        cloud.scales[i * 3 + 1] = 0.05;
        cloud.scales[i * 3 + 2] = 0.05;
        cloud.rotations[i * 4] = 1;
        cloud.colors[i * 3 + 0] = 0.8;
        cloud.colors[i * 3 + 1] = 0.4;
        cloud.colors[i * 3 + 2] = 0.2;
        cloud.opacities[i] = (i % 10) / 10;
        i++;
      }
    }
  }
  return cloud;
}

test('default edits are the identity', () => {
  assert.ok(isIdentity(normaliseEdits({})));
  const cloud = grid(3);
  assert.equal(applyEdits(cloud, normaliseEdits({})).count, cloud.count);
});

test('translation and scale move every gaussian', () => {
  const cloud = grid(3);
  const out = applyEdits(cloud, { translate: [1, 2, 3], scale: 2 });
  assert.equal(out.count, cloud.count);
  for (let i = 0; i < cloud.count; i++) {
    assert.ok(Math.abs(out.positions[i * 3 + 0] - (cloud.positions[i * 3 + 0] * 2 + 1)) < 1e-5);
    assert.ok(Math.abs(out.positions[i * 3 + 1] - (cloud.positions[i * 3 + 1] * 2 + 2)) < 1e-5);
    // Scaling the model scales the gaussians with it, or they would not match.
    assert.ok(Math.abs(out.scales[i * 3] - cloud.scales[i * 3] * 2) < 1e-6);
  }
});

test('a quarter turn about Y maps +X onto -Z', () => {
  const cloud = createCloud(1);
  cloud.positions.set([1, 0, 0]);
  cloud.scales.set([0.1, 0.1, 0.1]);
  cloud.rotations.set([1, 0, 0, 0]);
  cloud.opacities[0] = 1;
  const out = applyEdits(cloud, { rotate: [0, Math.PI / 2, 0] });
  assert.ok(Math.abs(out.positions[0] - 0) < 1e-6, 'x');
  assert.ok(Math.abs(out.positions[1] - 0) < 1e-6, 'y');
  assert.ok(Math.abs(out.positions[2] + 1) < 1e-6, 'z should be -1');
});

test('splatScale resizes gaussians without moving them', () => {
  const cloud = grid(3);
  const out = applyEdits(cloud, { splatScale: 3 });
  for (let i = 0; i < cloud.count; i++) {
    assert.equal(out.positions[i * 3], cloud.positions[i * 3]);
    assert.ok(Math.abs(out.scales[i * 3] - cloud.scales[i * 3] * 3) < 1e-6);
  }
});

test('crop keeps only what is inside the box', () => {
  const cloud = grid(5);
  const out = applyEdits(cloud, { crop: { min: [-0.1, -0.1, -0.1], max: [1.1, 1.1, 1.1] } });
  assert.ok(out.count < cloud.count);
  assert.ok(out.count > 0);
  for (let i = 0; i < out.count; i++) {
    for (let k = 0; k < 3; k++) {
      assert.ok(out.positions[i * 3 + k] >= -0.1 - 1e-6);
    }
  }
});

test('inverted crop keeps exactly the complement', () => {
  const cloud = grid(5);
  const box = { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] };
  const inside = applyEdits(cloud, { crop: { ...box, invert: false } });
  const outside = applyEdits(cloud, { crop: { ...box, invert: true } });
  assert.equal(inside.count + outside.count, cloud.count);
});

test('crop is evaluated before the transform, so the box does not drift', () => {
  const cloud = grid(5);
  const box = { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] };
  const plain = applyEdits(cloud, { crop: box });
  const moved = applyEdits(cloud, { crop: box, translate: [10, 0, 0], rotate: [0, 1, 0] });
  assert.equal(plain.count, moved.count);
});

test('pruning drops gaussians below the opacity threshold', () => {
  const cloud = grid(5);
  const out = applyEdits(cloud, { pruneBelowOpacity: 0.5 });
  assert.ok(out.count < cloud.count);
  for (let i = 0; i < out.count; i++) assert.ok(out.opacities[i] >= 0.5 - 1e-6);
});

test('opacity multiplier is applied before the prune test', () => {
  const cloud = createCloud(1);
  cloud.opacities[0] = 0.3;
  cloud.rotations[0] = 1;
  // 0.3 alone would be pruned at 0.5; doubled it survives.
  assert.equal(applyEdits(cloud, { pruneBelowOpacity: 0.5 }).count, 0);
  assert.equal(applyEdits(cloud, { pruneBelowOpacity: 0.5, opacity: 2 }).count, 1);
});

test('exposure and saturation grade colour within range', () => {
  const cloud = grid(2);
  const brighter = applyEdits(cloud, { exposure: 1 });
  assert.ok(brighter.colors[0] > cloud.colors[0]);

  const grey = applyEdits(cloud, { saturation: 0 });
  assert.ok(Math.abs(grey.colors[0] - grey.colors[1]) < 1e-6, 'channels converge when desaturated');
  for (const c of grey.colors) assert.ok(c >= 0 && c <= 1);
});

test('normaliseEdits clamps hostile input instead of trusting it', () => {
  const e = normaliseEdits({
    scale: 1e9,
    opacity: -5,
    saturation: 'not a number',
    translate: [1, 2],           // wrong arity
    rotate: [Infinity, 0, 0],
    background: 'javascript:alert(1)',
    pruneBelowOpacity: 99,
  });
  assert.ok(e.scale <= 100);
  assert.equal(e.opacity, 0);
  assert.equal(e.saturation, 1);
  assert.deepEqual(e.translate, [0, 0, 0]);
  assert.equal(e.rotate[0], 0);
  assert.equal(e.background, '#0b0d12');
  assert.equal(e.pruneBelowOpacity, 1);
});

test('normaliseEdits orders a reversed crop box', () => {
  const e = normaliseEdits({ crop: { min: [1, 1, 1], max: [-1, -1, -1] } });
  for (let k = 0; k < 3; k++) assert.ok(e.crop.min[k] <= e.crop.max[k]);
});
