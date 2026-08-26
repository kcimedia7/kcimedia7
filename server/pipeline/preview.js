import { createCloud, concatClouds, quatFromBasis, normaliseCloud } from './splat.js';
import {
  fitWithin, luminance, boxBlur, gradientMagnitude, borderColor, normaliseField,
} from './imageops.js';

/**
 * Preview reconstruction — the backend used when no COLMAP + CUDA trainer is
 * installed.
 *
 * This is NOT structure-from-motion. It cannot recover true camera poses, so it
 * assumes the capture pattern the app asks users to shoot: a turntable orbit
 * around a subject. Each frame is turned into a view-aligned relief of oriented
 * gaussian discs, back-projected through its assumed camera, and the reliefs are
 * merged. The result is a real, editable, exportable gaussian cloud built from
 * the user's own pixels — a fast proxy that looks like the subject from the
 * angles it was shot from, not a metrically accurate reconstruction.
 *
 * Depth comes from three cheap monocular cues, blended and smoothed:
 *   subject   how far a pixel's colour sits from the estimated backdrop colour
 *   detail    local gradient energy — in-focus regions read as nearer
 *   centre    a mild radial bias, because handheld captures frame the subject
 */

const DEFAULTS = {
  fovDeg: 55,
  orbitRadius: 2.6,
  arcDeg: 360,
  elevationDeg: 8,
  depthSpread: 0.55,
  subjectThreshold: 0.14,
  // Every camera's backdrop lands BEHIND its own subject but in FRONT of the
  // subject as seen from the opposite side of the orbit, so retained backdrop
  // samples veil the thing you actually captured. Off by default; raise it when
  // converting a whole scene rather than an isolated subject.
  backdropKeep: 0,
  discFlatten: 0.18,
  splatScale: 1.0,
};

/**
 * @param {Array<{width:number,height:number,data:Buffer}>} frames  decoded RGBA frames, capture order
 * @param {object} [options]
 * @returns {{cloud: object, stats: object}}
 */
export function reconstructPreview(frames, options = {}) {
  const opt = { ...DEFAULTS, ...options };
  if (!frames.length) throw new Error('preview reconstruction needs at least one frame');

  const grid = Math.max(24, Math.min(512, options.grid || 160));
  const perFrameBudget = Math.max(400, Math.floor((grid * grid) / frames.length) * 4);
  const clouds = [];
  let discarded = 0;

  frames.forEach((frame, index) => {
    const angle = frameAngle(index, frames.length, opt.arcDeg);
    const camera = orbitCamera(angle, opt);
    const { cloud, dropped } = reliefFromFrame(frame, camera, opt, perFrameBudget);
    discarded += dropped;
    clouds.push(cloud);
  });

  const cloud = concatClouds(clouds);
  normaliseCloud(cloud, 1);

  return {
    cloud,
    stats: {
      frames: frames.length,
      splats: cloud.count,
      discardedBackdropSamples: discarded,
      arcDeg: opt.arcDeg,
    },
  };
}

function frameAngle(index, total, arcDeg) {
  if (total === 1) return 0;
  // A full 360 orbit must not place the first and last frame on top of each
  // other, so the closed case steps by n rather than n-1.
  const closed = Math.abs(arcDeg % 360) < 1e-6 && arcDeg !== 0;
  const t = closed ? index / total : index / (total - 1);
  return (t * arcDeg * Math.PI) / 180;
}

function orbitCamera(angle, opt) {
  const elev = (opt.elevationDeg * Math.PI) / 180;
  const r = opt.orbitRadius;
  const position = [
    Math.sin(angle) * r * Math.cos(elev),
    Math.sin(elev) * r,
    Math.cos(angle) * r * Math.cos(elev),
  ];
  const forward = normalize([-position[0], -position[1], -position[2]]);
  const worldUp = [0, 1, 0];
  let right = cross(forward, worldUp);
  if (Math.hypot(...right) < 1e-5) right = [1, 0, 0];
  right = normalize(right);
  const up = normalize(cross(right, forward));
  return { position, forward, right, up, quat: quatFromBasis(right, up, forward) };
}

function reliefFromFrame(frame, camera, opt, budget) {
  const image = fitWithin(frame, 384);
  const { width, height, data } = image;

  const depthField = depthCues(image, opt);

  // Sample on a regular grid sized to the per-frame splat budget.
  const step = Math.max(1, Math.round(Math.sqrt((width * height) / budget)));
  const tanHalf = Math.tan((opt.fovDeg * Math.PI) / 360);
  const aspect = width / height;

  const nearD = opt.orbitRadius - opt.depthSpread;
  const farD = opt.orbitRadius + opt.depthSpread;

  const positions = [];
  const colors = [];
  const opacities = [];
  const scales = [];
  let dropped = 0;

  for (let y = Math.floor(step / 2); y < height; y += step) {
    for (let x = Math.floor(step / 2); x < width; x += step) {
      const i = y * width + x;
      const subject = depthField.subject[i];
      if (subject < opt.subjectThreshold) {
        // Backdrop: drop it, or keep a deterministic sparse sprinkle for context.
        dropped++;
        if (!opt.backdropKeep || hash2(x, y) > opt.backdropKeep) continue;
      }

      const depth01 = depthField.depth[i];
      const dist = farD - depth01 * (farD - nearD);

      const ndcX = ((x + 0.5) / width) * 2 - 1;
      const ndcY = 1 - ((y + 0.5) / height) * 2;

      const dir = normalize([
        camera.forward[0] + ndcX * tanHalf * aspect * camera.right[0] + ndcY * tanHalf * camera.up[0],
        camera.forward[1] + ndcX * tanHalf * aspect * camera.right[1] + ndcY * tanHalf * camera.up[1],
        camera.forward[2] + ndcX * tanHalf * aspect * camera.right[2] + ndcY * tanHalf * camera.up[2],
      ]);

      positions.push(
        camera.position[0] + dir[0] * dist,
        camera.position[1] + dir[1] * dist,
        camera.position[2] + dir[2] * dist,
      );

      const s = i * 4;
      // Stored as-is: 3DGS colours are displayed directly by viewers rather
      // than gamma-decoded, so keeping the frame's values matches what a real
      // trainer's degree-0 SH term would produce.
      colors.push(data[s] / 255, data[s + 1] / 255, data[s + 2] / 255);

      // Confidence-weighted alpha: backdrop samples stay faint, subject is solid.
      const alpha = 0.35 + 0.6 * Math.min(1, subject / 0.5);
      opacities.push(alpha * (data[s + 3] / 255));

      // A pixel's footprint at this distance, as a disc facing its source camera.
      const footprint = ((2 * tanHalf * dist) / height) * step * 1.0 * opt.splatScale;
      scales.push(footprint, footprint, footprint * opt.discFlatten);
    }
  }

  const count = opacities.length;
  const cloud = createCloud(count);
  cloud.positions.set(positions);
  cloud.colors.set(colors);
  cloud.opacities.set(opacities);
  cloud.scales.set(scales);
  for (let i = 0; i < count; i++) {
    cloud.rotations[i * 4 + 0] = camera.quat[0];
    cloud.rotations[i * 4 + 1] = camera.quat[1];
    cloud.rotations[i * 4 + 2] = camera.quat[2];
    cloud.rotations[i * 4 + 3] = camera.quat[3];
  }
  return { cloud, dropped };
}

function depthCues(image, opt) {
  const { width, height, data } = image;
  const n = width * height;
  const bg = borderColor(image);

  const subjectRaw = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const s = i * 4;
    const dr = data[s] / 255 - bg[0];
    const dg = data[s + 1] / 255 - bg[1];
    const db = data[s + 2] / 255 - bg[2];
    subjectRaw[i] = Math.min(1, Math.hypot(dr, dg, db) / 0.6);
  }
  const blurRadius = Math.max(1, Math.round(Math.min(width, height) * 0.02));
  const subject = boxBlur(subjectRaw, width, height, blurRadius, 2);

  const gray = luminance(image);
  const detail = boxBlur(gradientMagnitude(gray, width, height), width, height, blurRadius * 2, 2);
  const detailN = normaliseField(detail);

  const depth = new Float32Array(n);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const nx = ((x + 0.5) / width) * 2 - 1;
      const ny = ((y + 0.5) / height) * 2 - 1;
      const centre = 1 - Math.min(1, Math.hypot(nx, ny) / 1.4);
      depth[i] = 0.55 * subject[i] + 0.25 * detailN[i] + 0.20 * centre;
    }
  }
  const smoothed = boxBlur(normaliseField(depth), width, height, blurRadius, 2);
  return { depth: normaliseField(smoothed), subject };
}

/** Deterministic per-pixel jitter, so re-running a conversion is reproducible. */
function hash2(x, y) {
  let h = (x * 374761393 + y * 668265263) >>> 0;
  h = (h ^ (h >>> 13)) >>> 0;
  h = Math.imul(h, 1274126177) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function normalize(v) {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
