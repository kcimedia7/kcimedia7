import {
  createCloud, filterCloud, quatFromEuler, quatMultiply, quatNormalize, quatToMatrix,
} from './pipeline/splat.js';
import { defaultEdits } from './store.js';

/**
 * Edits are stored as parameters, never baked into the saved cloud, so a
 * conversion can be re-adjusted forever. The viewer applies the same parameters
 * on the GPU for live feedback; this module is the authoritative version used
 * when exporting a .ply, so what you download matches what you saw.
 */

export function normaliseEdits(input = {}) {
  const d = defaultEdits();
  const num = (v, fallback, lo, hi) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return fallback;
    return Math.min(hi, Math.max(lo, n));
  };
  const vec3 = (v, fallback, lo, hi) => (Array.isArray(v) && v.length === 3
    ? v.map((x, i) => num(x, fallback[i], lo, hi))
    : fallback);

  let crop = null;
  if (input.crop && Array.isArray(input.crop.min) && Array.isArray(input.crop.max)) {
    const min = vec3(input.crop.min, [-2, -2, -2], -50, 50);
    const max = vec3(input.crop.max, [2, 2, 2], -50, 50);
    crop = {
      min: min.map((v, i) => Math.min(v, max[i])),
      max: max.map((v, i) => Math.max(v, min[i])),
      invert: Boolean(input.crop.invert),
    };
  }

  return {
    translate: vec3(input.translate, d.translate, -100, 100),
    rotate: vec3(input.rotate, d.rotate, -Math.PI * 4, Math.PI * 4),
    scale: num(input.scale, d.scale, 0.01, 100),
    splatScale: num(input.splatScale, d.splatScale, 0.05, 8),
    opacity: num(input.opacity, d.opacity, 0, 4),
    exposure: num(input.exposure, d.exposure, -4, 4),
    saturation: num(input.saturation, d.saturation, 0, 4),
    crop,
    pruneBelowOpacity: num(input.pruneBelowOpacity, d.pruneBelowOpacity, 0, 1),
    background: typeof input.background === 'string' && /^#[0-9a-f]{6}$/i.test(input.background)
      ? input.background
      : d.background,
  };
}

export function isIdentity(edits) {
  const e = normaliseEdits(edits);
  return e.translate.every((v) => v === 0)
    && e.rotate.every((v) => v === 0)
    && e.scale === 1 && e.splatScale === 1 && e.opacity === 1
    && e.exposure === 0 && e.saturation === 1
    && !e.crop && e.pruneBelowOpacity === 0;
}

/**
 * Apply edits to a cloud, returning a new cloud.
 *
 * Order matters and mirrors the shader: crop and prune are evaluated against
 * the ORIGINAL coordinates (so a crop box the user drew does not slide around
 * when they later rotate the model), then the survivors are transformed and
 * their colours graded.
 */
export function applyEdits(cloud, rawEdits) {
  const e = normaliseEdits(rawEdits);

  const kept = filterCloud(cloud, (i) => {
    if (cloud.opacities[i] * e.opacity < e.pruneBelowOpacity) return false;
    if (e.crop) {
      const inside = [0, 1, 2].every((k) => {
        const v = cloud.positions[i * 3 + k];
        return v >= e.crop.min[k] && v <= e.crop.max[k];
      });
      if (inside === Boolean(e.crop.invert)) return false;
    }
    return true;
  });

  const out = createCloud(kept.count);
  const q = quatNormalize(quatFromEuler(e.rotate[0], e.rotate[1], e.rotate[2]));
  const m = quatToMatrix(q);
  const gain = Math.pow(2, e.exposure);

  for (let i = 0; i < kept.count; i++) {
    const x = kept.positions[i * 3 + 0];
    const y = kept.positions[i * 3 + 1];
    const z = kept.positions[i * 3 + 2];
    out.positions[i * 3 + 0] = (m[0] * x + m[1] * y + m[2] * z) * e.scale + e.translate[0];
    out.positions[i * 3 + 1] = (m[3] * x + m[4] * y + m[5] * z) * e.scale + e.translate[1];
    out.positions[i * 3 + 2] = (m[6] * x + m[7] * y + m[8] * z) * e.scale + e.translate[2];

    for (let k = 0; k < 3; k++) {
      out.scales[i * 3 + k] = kept.scales[i * 3 + k] * e.scale * e.splatScale;
    }

    const rotated = quatMultiply(q, [
      kept.rotations[i * 4 + 0], kept.rotations[i * 4 + 1],
      kept.rotations[i * 4 + 2], kept.rotations[i * 4 + 3],
    ]);
    const rn = quatNormalize(rotated);
    for (let k = 0; k < 4; k++) out.rotations[i * 4 + k] = rn[k];

    const r = kept.colors[i * 3 + 0] * gain;
    const g = kept.colors[i * 3 + 1] * gain;
    const b = kept.colors[i * 3 + 2] * gain;
    const grey = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    out.colors[i * 3 + 0] = clamp01(grey + (r - grey) * e.saturation);
    out.colors[i * 3 + 1] = clamp01(grey + (g - grey) * e.saturation);
    out.colors[i * 3 + 2] = clamp01(grey + (b - grey) * e.saturation);

    out.opacities[i] = clamp01(kept.opacities[i] * e.opacity);
  }
  return out;
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}
