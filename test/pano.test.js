import test from 'node:test';
import assert from 'node:assert/strict';
import {
  CUBE_FACES, DEFAULT_FACE_FOV_DEG, decodeRadianceHdr, rgbeToLinear, toneMap,
  linearToSrgb, equirectToPerspective, looksEquirectangular, faceIntrinsics,
  isHdrFile, isExrFile, MAX_PANO_EDGE,
} from '../web/js/pano.js';

/**
 * A 360 photo is useless to structure-from-motion until it has been resampled
 * into pinhole views, and a resampling that is subtly wrong does not fail --
 * it reconstructs the wrong scene. So the geometry is checked against
 * independently computed directions rather than against itself.
 */

/**
 * Build an equirect whose pixels encode their own longitude and latitude.
 *
 * Longitude is stored as sin and cos rather than as a ramp. A ramp would jump
 * from 255 back to 0 at the seam, and since the reprojection blends across the
 * seam -- correctly, those columns are neighbours on the sphere -- a ramp would
 * report the correct behaviour as an error. Latitude has no seam, so it ramps.
 */
function angleChart(width = 256, height = 128) {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const at = (y * width + x) * 4;
      const lon = ((x + 0.5) / width - 0.5) * 2 * Math.PI;
      data[at] = Math.round((Math.sin(lon) * 0.5 + 0.5) * 255);
      data[at + 1] = Math.round((y / (height - 1)) * 255);
      data[at + 2] = Math.round((Math.cos(lon) * 0.5 + 0.5) * 255);
      data[at + 3] = 255;
    }
  }
  return { width, height, data };
}

/** What the chart above stores at a given direction, computed from scratch. */
function expectedAt(dx, dy, dz, width = 256, height = 128) {
  const len = Math.hypot(dx, dy, dz);
  const lon = Math.atan2(dx, dz);
  const lat = Math.asin(dy / len);
  const v = (0.5 - lat / Math.PI) * height - 0.5;
  return {
    r: (Math.sin(lon) * 0.5 + 0.5) * 255,
    g: (Math.min(height - 1, Math.max(0, v)) / (height - 1)) * 255,
    b: (Math.cos(lon) * 0.5 + 0.5) * 255,
  };
}

function centrePixel(face) {
  const mid = Math.floor(face.width / 2);
  const at = (mid * face.width + mid) * 4;
  return [face.data[at], face.data[at + 1], face.data[at + 2]];
}

test('each extracted view looks along its own optical axis', () => {
  // The centre pixel of a perspective view must sample the panorama exactly
  // where the view's forward vector points. A transposed or negated basis
  // still produces a plausible-looking image, so this compares against the
  // direction rather than against a reference render.
  const src = angleChart();
  for (const face of CUBE_FACES) {
    const view = equirectToPerspective(src, face, { size: 64 });
    const [fx, fy, fz] = face.forward;
    const want = expectedAt(fx, fy, fz);
    const [r, g, b] = centrePixel(view);
    // Latitude is well defined everywhere, including at the poles.
    // The centre of an even-sized image sits half a pixel off the axis.
    assert.ok(Math.abs(g - want.g) < 6, `${face.name}: latitude ${g} vs ${want.g}`);
    // Longitude is not: every meridian meets at the pole, so for the views
    // looking straight up and down the centre pixel's longitude is whatever
    // the half-pixel offset happens to land on. Asserting it would be
    // asserting an arbitrary number.
    if (Math.abs(fy) === 1) continue;
    assert.ok(Math.abs(r - want.r) < 6, `${face.name}: sin(longitude) ${r} vs ${want.r}`);
    assert.ok(Math.abs(b - want.b) < 6, `${face.name}: cos(longitude) ${b} vs ${want.b}`);
  }
});

test('the pole views sample the top and bottom of the panorama', () => {
  // Latitude is the only meaningful probe straight up or down, and it is the
  // one that catches an inverted vertical axis -- a mistake that leaves every
  // extracted view upside down while still looking like a valid image.
  const src = angleChart();
  const view = (name) => equirectToPerspective(src, CUBE_FACES.find((f) => f.name === name), { size: 64 });
  const [, upLat] = centrePixel(view('up'));
  const [, downLat] = centrePixel(view('down'));
  assert.ok(upLat < 8, `looking up should read the first rows, got ${upLat}`);
  assert.ok(downLat > 247, `looking down should read the last rows, got ${downLat}`);
});

test('every face basis is orthonormal and right-handed', () => {
  // right x up === forward is what makes the extracted image un-mirrored. A
  // left-handed basis yields a mirror image, which feature matching will
  // happily match to nothing.
  for (const { name, forward, right, up } of CUBE_FACES) {
    for (const [label, v] of [['forward', forward], ['right', right], ['up', up]]) {
      assert.ok(Math.abs(Math.hypot(...v) - 1) < 1e-12, `${name} ${label} is not unit length`);
    }
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    assert.ok(Math.abs(dot(forward, right)) < 1e-12, `${name}: forward and right are not perpendicular`);
    assert.ok(Math.abs(dot(forward, up)) < 1e-12, `${name}: forward and up are not perpendicular`);
    assert.ok(Math.abs(dot(right, up)) < 1e-12, `${name}: right and up are not perpendicular`);
    const cross = [
      right[1] * up[2] - right[2] * up[1],
      right[2] * up[0] - right[0] * up[2],
      right[0] * up[1] - right[1] * up[0],
    ];
    for (let i = 0; i < 3; i++) {
      assert.ok(Math.abs(cross[i] - forward[i]) < 1e-12, `${name} is left-handed (mirrored)`);
    }
  }
});

test('the six views at 90 degrees cover the whole sphere', () => {
  // Any direction must fall inside at least one view, or reconstruction would
  // have a blind spot that no amount of capture fixes.
  const t = Math.tan((90 * Math.PI) / 360);
  let uncovered = 0;
  for (let a = 0; a < 40; a++) {
    for (let b = 0; b < 40; b++) {
      const lon = (a / 40) * 2 * Math.PI - Math.PI;
      const lat = Math.asin((b / 39) * 2 - 1); // equal-area in latitude
      const d = [
        Math.cos(lat) * Math.sin(lon),
        Math.sin(lat),
        Math.cos(lat) * Math.cos(lon),
      ];
      const covered = CUBE_FACES.some(({ forward, right, up }) => {
        const z = d[0] * forward[0] + d[1] * forward[1] + d[2] * forward[2];
        if (z <= 1e-9) return false;
        const x = (d[0] * right[0] + d[1] * right[1] + d[2] * right[2]) / z;
        const y = (d[0] * up[0] + d[1] * up[1] + d[2] * up[2]) / z;
        return Math.abs(x) <= t + 1e-9 && Math.abs(y) <= t + 1e-9;
      });
      if (!covered) uncovered += 1;
    }
  }
  assert.equal(uncovered, 0, `${uncovered} directions fall outside every view`);
});

test('the default field of view leaves neighbouring views overlapping', () => {
  // Exactly 90 degrees tiles the sphere but shares no area, so no feature can
  // appear in two views and nothing ties them together during matching.
  assert.ok(DEFAULT_FACE_FOV_DEG > 90,
    'views that merely touch give feature matching nothing to work with');
  const t = Math.tan((DEFAULT_FACE_FOV_DEG * Math.PI) / 360);
  // A direction 45 degrees off the front axis is the corner between front and
  // right; with overlap it must be inside both.
  const d = [Math.sin(Math.PI / 4), 0, Math.cos(Math.PI / 4)];
  const inside = CUBE_FACES.filter(({ forward, right, up }) => {
    const z = d[0] * forward[0] + d[1] * forward[1] + d[2] * forward[2];
    if (z <= 1e-9) return false;
    const x = (d[0] * right[0] + d[1] * right[1] + d[2] * right[2]) / z;
    const y = (d[0] * up[0] + d[1] * up[1] + d[2] * up[2]) / z;
    return Math.abs(x) <= t && Math.abs(y) <= t;
  });
  assert.ok(inside.length >= 2, 'the seam direction should appear in two views');
});

test('a view straddling the panorama seam reads across it', () => {
  // The back view is centred on the left and right edges of the source. If
  // longitude clamped instead of wrapping, half of it would be a smear of the
  // edge column.
  const width = 256;
  const src = angleChart(width, 128);
  // The chart's own longitude encoding is continuous across the seam, so the
  // back view -- centred exactly on it -- must read cos(pi) at its middle.
  // Clamping instead of wrapping would smear one edge column across half the
  // view and land nowhere near it.
  const back = equirectToPerspective(src, CUBE_FACES.find((f) => f.name === 'back'), { size: 64 });
  const [, , blue] = centrePixel(back);
  const want = (Math.cos(Math.PI) * 0.5 + 0.5) * 255; // 0
  assert.ok(Math.abs(blue - want) < 6, `seam read ${blue}, expected ${want}`);
});

test('an impossible field of view is rejected', () => {
  const front = CUBE_FACES[0];
  const src = angleChart(64, 32);
  for (const fovDeg of [0, -10, 180, 360]) {
    assert.throws(() => equirectToPerspective(src, front, { size: 16, fovDeg }),
      /field of view/, `${fovDeg} degrees should not be accepted`);
  }
});

test('face intrinsics match the field of view they were built from', () => {
  const { fx, cx, width } = faceIntrinsics(512, 90);
  // At 90 degrees the focal length in pixels equals half the sensor width.
  assert.ok(Math.abs(fx - 256) < 1e-9, `expected fx 256, got ${fx}`);
  assert.equal(cx, 256);
  assert.equal(width, 512);
  // A wider view has a shorter focal length.
  assert.ok(faceIntrinsics(512, 120).fx < fx);
});

test('only a 2:1 image is treated as a panorama', () => {
  assert.ok(looksEquirectangular(4096, 2048));
  assert.ok(looksEquirectangular(4000, 2000));
  assert.ok(looksEquirectangular(4040, 2000), 'a small crop should still pass');
  assert.ok(!looksEquirectangular(4096, 4096), 'a square photo is not a panorama');
  assert.ok(!looksEquirectangular(1920, 1080), 'ordinary 16:9 is not a panorama');
  assert.ok(!looksEquirectangular(0, 0));
});

test('panorama file types are told apart by extension', () => {
  assert.ok(isHdrFile({ name: 'kitchen.hdr' }));
  assert.ok(isHdrFile({ name: 'KITCHEN.HDR' }));
  assert.ok(!isHdrFile({ name: 'kitchen.jpg' }));
  assert.ok(isExrFile({ name: 'kitchen.exr' }));
  assert.ok(!isExrFile({ name: 'kitchen.hdr' }));
});

/* ------------------------------------------------------------------ *
 * Radiance HDR
 * ------------------------------------------------------------------ */

/** Encode RGBE scanlines the way Radiance does, to test the decoder against. */
function encodeHdr(width, height, pixelAt, { rle = true } = {}) {
  const header = Buffer.from(
    `#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y ${height} +X ${width}\n`, 'latin1');
  const rows = [];
  for (let y = 0; y < height; y++) {
    const row = [];
    for (let x = 0; x < width; x++) row.push(pixelAt(x, y));
    // Radiance only uses the run-length scanline format for widths of 8 and
    // up, and a decoder is required to read the marker as an ordinary pixel
    // below that. Emitting RLE anyway would produce a file no real encoder
    // writes and no real decoder reads.
    if (!rle || width < 8 || width >= 32768) {
      rows.push(Buffer.from(row.flat()));
      continue;
    }
    const body = [2, 2, (width >> 8) & 0xff, width & 0xff];
    for (let c = 0; c < 4; c++) {
      // Literal runs only; the decoder must handle both, and the repeat path
      // is exercised by the dedicated test below.
      let x = 0;
      while (x < width) {
        const n = Math.min(128, width - x);
        body.push(n);
        for (let k = 0; k < n; k++) body.push(row[x + k][c]);
        x += n;
      }
    }
    rows.push(Buffer.from(body));
  }
  return new Uint8Array(Buffer.concat([header, ...rows]));
}

test('a run-length encoded HDR decodes to the pixels it was built from', () => {
  const width = 64;
  const height = 8;
  const pixel = (x, y) => [x * 3 % 256, y * 31 % 256, (x + y) % 256, 128];
  const decoded = decodeRadianceHdr(encodeHdr(width, height, pixel));
  assert.equal(decoded.width, width);
  assert.equal(decoded.height, height);
  const expect = rgbeToLinear(new Uint8Array(
    Array.from({ length: width * height }, (_, i) => pixel(i % width, Math.floor(i / width))).flat()));
  for (let i = 0; i < expect.length; i++) {
    assert.ok(Math.abs(decoded.data[i] - expect[i]) < 1e-6, `sample ${i} differs`);
  }
});

test('a flat HDR with no run-length encoding decodes identically', () => {
  // Small images skip RLE entirely, and a decoder that only handles the RLE
  // path returns garbage for them rather than failing.
  const pixel = (x, y) => [x % 256, y % 256, 7, 130];
  const rleFree = decodeRadianceHdr(encodeHdr(4, 4, pixel, { rle: false }));
  assert.equal(rleFree.width, 4);
  const expect = rgbeToLinear(new Uint8Array(
    Array.from({ length: 16 }, (_, i) => pixel(i % 4, Math.floor(i / 4))).flat()));
  for (let i = 0; i < expect.length; i++) {
    assert.ok(Math.abs(rleFree.data[i] - expect[i]) < 1e-6, `sample ${i} differs`);
  }
});

test('a repeated run in an HDR expands to the pixel it repeats', () => {
  const width = 16;
  const header = Buffer.from(`#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X ${width}\n`, 'latin1');
  const body = [10, 20, 30, 140];              // one real pixel
  body.push(1, 1, 1, width - 1);               // then repeat it to fill the row
  const decoded = decodeRadianceHdr(new Uint8Array(Buffer.concat([header, Buffer.from(body)])));
  assert.equal(decoded.width, width);
  for (let x = 1; x < width; x++) {
    for (let c = 0; c < 3; c++) {
      assert.equal(decoded.data[x * 3 + c], decoded.data[c], `pixel ${x} channel ${c}`);
    }
  }
});

test('a file that is not a Radiance HDR is refused by name', () => {
  assert.throws(() => decodeRadianceHdr(new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0a])),
    /Radiance/);
  // Random bytes with no newline must not be spread into a call.
  assert.throws(() => decodeRadianceHdr(new Uint8Array(5000).fill(0x41)), /Radiance/);
  // Truncated data must say so rather than return a half-black image.
  const truncated = encodeHdr(64, 8, () => [1, 2, 3, 128]).subarray(0, 60);
  assert.throws(() => decodeRadianceHdr(truncated));
});

test('an XYZE image is refused rather than read as RGB', () => {
  const header = Buffer.from('#?RADIANCE\nFORMAT=32-bit_rle_xyze\n\n-Y 1 +X 8\n', 'latin1');
  assert.throws(() => decodeRadianceHdr(new Uint8Array(header)), /unsupported HDR format/);
});

test('the RGBE exponent expands to real radiance', () => {
  // Exponent 0 is exactly black by definition, not a very small number.
  const black = rgbeToLinear(new Uint8Array([255, 255, 255, 0]));
  assert.deepEqual(Array.from(black), [0, 0, 0]);
  // Radiance stores the mantissa at bucket centre, so 128 with exponent 136
  // reconstructs just above 128/256.
  const mid = rgbeToLinear(new Uint8Array([128, 128, 128, 136]));
  assert.ok(Math.abs(mid[0] - 128.5) < 1e-3, `got ${mid[0]}`);
  // Each exponent step doubles the value.
  const up = rgbeToLinear(new Uint8Array([128, 128, 128, 137]));
  assert.ok(Math.abs(up[0] - 2 * mid[0]) < 1e-3);
});

/* ------------------------------------------------------------------ *
 * Tone mapping
 * ------------------------------------------------------------------ */

test('tone mapping brings any exposure to a usable mid grey', () => {
  // The same scene shot four stops apart must land in the same place, or the
  // frames from one panorama would not match those from the next.
  const n = 32 * 32;
  const results = [];
  for (const gain of [0.01, 1, 100, 10_000]) {
    const rgb = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const v = (0.2 + (i % 16) / 16) * gain;
      rgb[i * 3] = v; rgb[i * 3 + 1] = v; rgb[i * 3 + 2] = v;
    }
    const out = toneMap(rgb, 32, 32);
    let sum = 0;
    for (let i = 0; i < n; i++) sum += out[i * 4];
    results.push(sum / n);
  }
  for (const mean of results) {
    assert.ok(mean > 60 && mean < 210, `exposure landed at ${mean}, outside a usable range`);
  }
  const spread = Math.max(...results) - Math.min(...results);
  assert.ok(spread < 8, `six orders of magnitude moved the result by ${spread} levels`);
});

test('tone mapping keeps highlights apart instead of clipping them', () => {
  // A window blown to flat white contributes no features. Values well above
  // the clip point of a naive conversion must stay distinguishable.
  const n = 4;
  const rgb = new Float32Array(n * 3);
  const values = [1, 10, 100, 1000];
  for (let i = 0; i < n; i++) {
    rgb[i * 3] = values[i]; rgb[i * 3 + 1] = values[i]; rgb[i * 3 + 2] = values[i];
  }
  const out = toneMap(rgb, n, 1);
  const levels = [0, 1, 2, 3].map((i) => out[i * 4]);
  for (let i = 1; i < levels.length; i++) {
    assert.ok(levels[i] > levels[i - 1],
      `${values[i]} and ${values[i - 1]} both became ${levels[i]}`);
  }
});

test('tone mapping preserves hue rather than desaturating', () => {
  // Curving each channel on its own pulls saturated colours toward grey, which
  // changes what a descriptor sees. A pure red must stay pure red.
  const rgb = new Float32Array([8, 0, 0, 0.5, 0, 0]);
  const out = toneMap(rgb, 2, 1);
  for (const i of [0, 1]) {
    assert.ok(out[i * 4] > out[i * 4 + 1], 'red channel should dominate');
    assert.equal(out[i * 4 + 1], 0, 'green should stay zero');
    assert.equal(out[i * 4 + 2], 0, 'blue should stay zero');
  }
});

test('tone mapping never emits a non-finite or out-of-range sample', () => {
  const rgb = new Float32Array([0, 0, 0, 1e12, 1e12, 1e12, 1e-12, 0, 5]);
  const out = toneMap(rgb, 3, 1);
  for (const v of out) {
    assert.ok(Number.isFinite(v) && v >= 0 && v <= 255, `bad sample ${v}`);
  }
  // A fully black image has no exposure information and must not divide by zero.
  const black = toneMap(new Float32Array(12), 2, 2);
  for (const v of black) assert.ok(Number.isFinite(v));
});

test('the sRGB transfer function matches its definition at the joins', () => {
  assert.equal(linearToSrgb(0), 0);
  assert.equal(linearToSrgb(1), 1);
  assert.equal(linearToSrgb(-5), 0, 'negative radiance clamps to black');
  assert.equal(linearToSrgb(5), 1);
  // The linear segment meets the power segment at 0.0031308.
  const below = linearToSrgb(0.0031308 - 1e-9);
  const above = linearToSrgb(0.0031308 + 1e-9);
  assert.ok(Math.abs(below - above) < 1e-6, 'the two segments do not meet');
  // Mid grey: 0.18 linear is about 0.46 encoded.
  assert.ok(Math.abs(linearToSrgb(0.18) - 0.4620) < 1e-3);
});

test('a panorama is box-averaged down while it decodes', () => {
  // Holding an 8k panorama as floating-point radiance costs hundreds of
  // megabytes, so rows are folded into the output grid as they are read. The
  // averaging must preserve the radiance, not just the size.
  const width = 64;
  const height = 32;
  // A uniform value survives any correct box filter unchanged.
  const flat = decodeRadianceHdr(encodeHdr(width, height, () => [128, 128, 128, 136]),
    { maxEdge: 16 });
  assert.equal(flat.width, 16);
  assert.equal(flat.height, 8);
  for (let i = 0; i < flat.data.length; i++) {
    assert.ok(Math.abs(flat.data[i] - 128.5) < 1e-3, `sample ${i} is ${flat.data[i]}`);
  }

  // A gradient must come back as the mean of each block, not a dropped sample.
  const ramp = decodeRadianceHdr(
    encodeHdr(4, 1, (x) => [x * 8, 0, 0, 136]), { maxEdge: 2 });
  assert.equal(ramp.width, 2);
  // Columns 0 and 1 hold mantissas 0 and 8, whose bucket centres average to 4.5.
  assert.ok(Math.abs(ramp.data[0] - 4.5) < 1e-3, `got ${ramp.data[0]}`);
  // Columns 2 and 3 hold 16 and 24, averaging to 20.5.
  assert.ok(Math.abs(ramp.data[3] - 20.5) < 1e-3, `got ${ramp.data[3]}`);
});

test('decoding stays under the default edge without being asked', () => {
  // The cap has to apply by default, or the first 8k panorama a user drops in
  // takes the tab down before any explicit option could have helped.
  assert.ok(MAX_PANO_EDGE <= 4096, 'the cap must be small enough to matter');
  const wide = decodeRadianceHdr(encodeHdr(MAX_PANO_EDGE * 2, 2, () => [64, 64, 64, 136]));
  assert.equal(wide.width, MAX_PANO_EDGE);
  // A panorama already within the cap is left at its own resolution.
  const small = decodeRadianceHdr(encodeHdr(32, 16, () => [64, 64, 64, 136]));
  assert.equal(small.width, 32);
  assert.equal(small.height, 16);
});
