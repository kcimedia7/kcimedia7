/**
 * In-memory Gaussian splat cloud, structure-of-arrays.
 *
 *   positions  xyz, world units
 *   scales     xyz, linear world units (PLY stores these as log)
 *   rotations  quaternion wxyz, normalised (PLY stores rot_0..3 as wxyz)
 *   colors     linear RGB 0..1 (PLY stores these as SH degree-0 coefficients)
 *   opacities  0..1 (PLY stores the logit)
 */

export const SH_C0 = 0.28209479177387814;

export function createCloud(count) {
  return {
    count,
    positions: new Float32Array(count * 3),
    scales: new Float32Array(count * 3),
    rotations: new Float32Array(count * 4),
    colors: new Float32Array(count * 3),
    opacities: new Float32Array(count),
  };
}

export function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

export function logit(p) {
  const c = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
  return Math.log(c / (1 - c));
}

export function shToColor(dc) {
  return 0.5 + SH_C0 * dc;
}

export function colorToSh(c) {
  return (c - 0.5) / SH_C0;
}

export function concatClouds(clouds) {
  const list = clouds.filter((c) => c && c.count > 0);
  const total = list.reduce((n, c) => n + c.count, 0);
  const out = createCloud(total);
  let o = 0;
  for (const c of list) {
    out.positions.set(c.positions.subarray(0, c.count * 3), o * 3);
    out.scales.set(c.scales.subarray(0, c.count * 3), o * 3);
    out.rotations.set(c.rotations.subarray(0, c.count * 4), o * 4);
    out.colors.set(c.colors.subarray(0, c.count * 3), o * 3);
    out.opacities.set(c.opacities.subarray(0, c.count), o);
    o += c.count;
  }
  return out;
}

/** Keep only the splats whose index passes `predicate`. */
export function filterCloud(cloud, predicate) {
  const keep = [];
  for (let i = 0; i < cloud.count; i++) if (predicate(i)) keep.push(i);
  const out = createCloud(keep.length);
  keep.forEach((src, dst) => {
    for (let k = 0; k < 3; k++) {
      out.positions[dst * 3 + k] = cloud.positions[src * 3 + k];
      out.scales[dst * 3 + k] = cloud.scales[src * 3 + k];
      out.colors[dst * 3 + k] = cloud.colors[src * 3 + k];
    }
    for (let k = 0; k < 4; k++) out.rotations[dst * 4 + k] = cloud.rotations[src * 4 + k];
    out.opacities[dst] = cloud.opacities[src];
  });
  return out;
}

export function boundsOf(cloud) {
  if (!cloud.count) {
    return { min: [0, 0, 0], max: [0, 0, 0], center: [0, 0, 0], radius: 1 };
  }
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < cloud.count; i++) {
    for (let k = 0; k < 3; k++) {
      const v = cloud.positions[i * 3 + k];
      // A diverged training run writes NaN into positions. Comparisons against
      // NaN are all false so it would not widen the bounds here, but it does
      // poison every percentile in robustBounds -- and bounds that are not
      // finite leave the camera unplaceable and the viewport black, while the
      // gaussian count still reads correctly. Skip them in both places.
      if (!Number.isFinite(v)) continue;
      if (v < min[k]) min[k] = v;
      if (v > max[k]) max[k] = v;
    }
  }
  // Nothing finite anywhere: fall back to a unit box rather than returning
  // Infinity, which fails the same way NaN does.
  for (let k = 0; k < 3; k++) {
    if (!Number.isFinite(min[k]) || !Number.isFinite(max[k])) { min[k] = 0; max[k] = 0; }
  }
  const center = [0, 1, 2].map((k) => (min[k] + max[k]) / 2);
  const radius = Math.max(
    1e-3,
    Math.hypot(max[0] - center[0], max[1] - center[1], max[2] - center[2]),
  );
  return { min, max, center, radius };
}

/**
 * Robust centre/extent used to normalise a reconstruction into a predictable
 * viewing volume. Percentiles rather than min/max so a handful of stray
 * gaussians behind the camera cannot shrink the subject to a dot.
 */
export function robustBounds(cloud, lowPct = 0.02, highPct = 0.98) {
  if (!cloud.count) return boundsOf(cloud);
  const axes = [0, 1, 2].map((k) => {
    // Typed-array sort is numeric, but it orders NaN last -- so a diverged run
    // with more than (1 - highPct) of its positions non-finite would put NaN at
    // the high percentile and make the radius NaN. Collect only finite values.
    const vals = new Float32Array(cloud.count);
    let n = 0;
    for (let i = 0; i < cloud.count; i++) {
      const v = cloud.positions[i * 3 + k];
      if (Number.isFinite(v)) vals[n++] = v;
    }
    if (!n) return [0, 0];
    const finite = vals.subarray(0, n);
    finite.sort();
    const lo = finite[Math.floor((n - 1) * lowPct)];
    const hi = finite[Math.floor((n - 1) * highPct)];
    return [lo, hi];
  });
  const center = axes.map(([lo, hi]) => (lo + hi) / 2);
  const radius = Math.max(1e-3, Math.max(...axes.map(([lo, hi]) => (hi - lo) / 2)));
  return { min: axes.map((a) => a[0]), max: axes.map((a) => a[1]), center, radius };
}

/** Scale and translate in place so the cloud sits at the origin within `targetRadius`. */
export function normaliseCloud(cloud, targetRadius = 1) {
  const { center, radius } = robustBounds(cloud);
  const k = targetRadius / radius;
  for (let i = 0; i < cloud.count; i++) {
    for (let a = 0; a < 3; a++) {
      cloud.positions[i * 3 + a] = (cloud.positions[i * 3 + a] - center[a]) * k;
      cloud.scales[i * 3 + a] *= k;
    }
  }
  return cloud;
}

export function quatFromAxisAngle(axis, angle) {
  const len = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  const h = angle / 2;
  const s = Math.sin(h) / len;
  return [Math.cos(h), axis[0] * s, axis[1] * s, axis[2] * s];
}

export function quatMultiply(a, b) {
  const [aw, ax, ay, az] = a;
  const [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

export function quatNormalize(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

/** Rotation matrix (row-major 3x3) from a wxyz quaternion. */
export function quatToMatrix(q) {
  const [w, x, y, z] = quatNormalize(q);
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
}

export function quatFromEuler(rx, ry, rz) {
  const qx = quatFromAxisAngle([1, 0, 0], rx);
  const qy = quatFromAxisAngle([0, 1, 0], ry);
  const qz = quatFromAxisAngle([0, 0, 1], rz);
  return quatNormalize(quatMultiply(quatMultiply(qz, qy), qx));
}

/** Encode to the compact 32-byte-per-splat `.splat` format the viewer streams. */
export function encodeSplatBuffer(cloud) {
  const out = new ArrayBuffer(cloud.count * 32);
  const f32 = new Float32Array(out);
  const u8 = new Uint8Array(out);
  for (let i = 0; i < cloud.count; i++) {
    const f = i * 8;
    f32[f + 0] = cloud.positions[i * 3 + 0];
    f32[f + 1] = cloud.positions[i * 3 + 1];
    f32[f + 2] = cloud.positions[i * 3 + 2];
    f32[f + 3] = cloud.scales[i * 3 + 0];
    f32[f + 4] = cloud.scales[i * 3 + 1];
    f32[f + 5] = cloud.scales[i * 3 + 2];
    const b = i * 32 + 24;
    for (let k = 0; k < 3; k++) {
      u8[b + k] = clamp255(cloud.colors[i * 3 + k] * 255);
    }
    u8[b + 3] = clamp255(cloud.opacities[i] * 255);
    const q = quatNormalize([
      cloud.rotations[i * 4 + 0], cloud.rotations[i * 4 + 1],
      cloud.rotations[i * 4 + 2], cloud.rotations[i * 4 + 3],
    ]);
    for (let k = 0; k < 4; k++) u8[b + 4 + k] = clamp255(q[k] * 128 + 128);
  }
  return Buffer.from(out);
}

export function decodeSplatBuffer(buffer) {
  const count = Math.floor(buffer.byteLength / 32);
  const cloud = createCloud(count);
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  for (let i = 0; i < count; i++) {
    const o = i * 32;
    for (let k = 0; k < 3; k++) {
      cloud.positions[i * 3 + k] = view.getFloat32(o + k * 4, true);
      cloud.scales[i * 3 + k] = view.getFloat32(o + 12 + k * 4, true);
      cloud.colors[i * 3 + k] = view.getUint8(o + 24 + k) / 255;
    }
    cloud.opacities[i] = view.getUint8(o + 27) / 255;
    for (let k = 0; k < 4; k++) {
      cloud.rotations[i * 4 + k] = (view.getUint8(o + 28 + k) - 128) / 128;
    }
  }
  return cloud;
}

function clamp255(v) {
  return Math.max(0, Math.min(255, Math.round(v)));
}

/** Quaternion (wxyz) from an orthonormal basis given as column vectors. */
export function quatFromBasis(right, up, forward) {
  // Column-major basis -> row-major rotation matrix m[row*3+col].
  const m = [
    right[0], up[0], forward[0],
    right[1], up[1], forward[1],
    right[2], up[2], forward[2],
  ];
  const trace = m[0] + m[4] + m[8];
  let q;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    q = [0.25 * s, (m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s];
  } else if (m[0] > m[4] && m[0] > m[8]) {
    const s = Math.sqrt(1 + m[0] - m[4] - m[8]) * 2;
    q = [(m[7] - m[5]) / s, 0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s];
  } else if (m[4] > m[8]) {
    const s = Math.sqrt(1 + m[4] - m[0] - m[8]) * 2;
    q = [(m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s];
  } else {
    const s = Math.sqrt(1 + m[8] - m[0] - m[4]) * 2;
    q = [(m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s];
  }
  return quatNormalize(q);
}
