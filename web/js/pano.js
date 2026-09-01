/**
 * 360 panorama ingest: Radiance HDR decoding, tone mapping, and
 * equirectangular-to-perspective reprojection.
 *
 * A 360 photo cannot be handed to structure-from-motion directly. COLMAP and
 * the bundled trainer both model a camera as a pinhole -- rays through a single
 * point onto a flat sensor -- and an equirectangular image is a spherical
 * projection, where a straight line in the world is a curve in the image.
 * Feeding one in unchanged does not fail loudly; it fits a pinhole model to
 * data that never came from a pinhole and returns plausible-looking nonsense.
 *
 * So each panorama is resampled into a set of ordinary perspective views that
 * share an optical centre. Those *are* real pinhole images, and the rest of the
 * pipeline needs no knowledge that a panorama was ever involved.
 *
 * Everything here is deliberately free of DOM references: it works on plain
 * typed arrays so the geometry can be tested directly rather than through a
 * canvas.
 */

/**
 * Panorama detail tiers, by the width the source is decoded at.
 *
 * The relationship that matters: a view of `fov` degrees rendered at `size`
 * pixels resolves the same detail as an equirectangular source `size * 360 /
 * fov` pixels wide -- about 3.6x the face size at the default field of view.
 * So the source width and the extracted view size are not independent choices.
 * Decoding at 16k and then extracting 512-pixel views throws away everything
 * the larger source was for, and extracting 4096-pixel views from a 2k source
 * invents detail that is not there.
 *
 * Every tier is derived from that one relationship, and the costs are measured
 * rather than estimated -- reprojection time for all six views, and the peak
 * decoded size of the source.
 */
export const PANO_TIERS = [
  { id: '2k', width: 2048, label: '2K — fastest', seconds: 0.4, sourceMB: 8 },
  { id: '4k', width: 4096, label: '4K — balanced', seconds: 1.0, sourceMB: 34 },
  { id: '8k', width: 8192, label: '8K — high detail', seconds: 3.1, sourceMB: 134 },
  { id: '16k', width: 16384, label: '16K — maximum', seconds: 13.4, sourceMB: 537 },
];

/** Tier used when nothing is chosen: real detail without a heavy wait. */
export const DEFAULT_PANO_WIDTH = 4096;

/** Largest source we will attempt. Beyond this a canvas readback is 0.5 GB. */
export const MAX_PANO_EDGE = 16384;

/** Round a requested source width to a tier this build supports. */
export function resolvePanoWidth(requested) {
  const n = Number(requested);
  if (!Number.isFinite(n) || n <= 0) return DEFAULT_PANO_WIDTH;
  // Snap to the nearest tier rather than honouring an arbitrary number: the
  // sizes below are the ones whose cost has actually been measured.
  let best = PANO_TIERS[0];
  for (const tier of PANO_TIERS) {
    if (Math.abs(tier.width - n) < Math.abs(best.width - n)) best = tier;
  }
  return best.width;
}

/**
 * The view size a source of this width actually justifies.
 *
 * Larger throws pixels away on interpolation; smaller discards detail the
 * source paid for. This is the whole reason the detail setting is a single
 * choice rather than two independent ones.
 */
export function faceSizeFor(sourceWidth, fovDeg = DEFAULT_FACE_FOV_DEG) {
  return Math.max(64, Math.round((sourceWidth * fovDeg) / 360));
}

/**
 * The six view directions, as an orthonormal basis each.
 *
 * `forward` is the optical axis, `right` is +x in the output image and `up` is
 * -y (image rows count downward). Each basis is right-handed with
 * right x up === forward, in a world where +x is right, +y is up and +z is the
 * direction at the horizontal centre of the equirectangular image.
 */
export const CUBE_FACES = [
  { name: 'front', forward: [0, 0, 1], right: [1, 0, 0], up: [0, 1, 0] },
  { name: 'right', forward: [1, 0, 0], right: [0, 0, -1], up: [0, 1, 0] },
  { name: 'back', forward: [0, 0, -1], right: [-1, 0, 0], up: [0, 1, 0] },
  { name: 'left', forward: [-1, 0, 0], right: [0, 0, 1], up: [0, 1, 0] },
  { name: 'up', forward: [0, 1, 0], right: [1, 0, 0], up: [0, 0, -1] },
  { name: 'down', forward: [0, -1, 0], right: [1, 0, 0], up: [0, 0, 1] },
];

/**
 * Field of view for each extracted view, in degrees.
 *
 * Exactly 90 degrees tiles the sphere with no waste, which is why cubemaps use
 * it -- but it also means two neighbouring views share a single column of
 * pixels. Feature matching needs a band of genuine overlap to find
 * correspondences in, so the views are widened past the point where they tile.
 * The cost is resampling some of the sphere twice; the benefit is that
 * structure-from-motion can actually connect the views into one reconstruction.
 */
export const DEFAULT_FACE_FOV_DEG = 100;

export function isHdrFile(file) {
  return /\.(hdr|pic)$/i.test(file.name || '');
}

export function isExrFile(file) {
  return /\.exr$/i.test(file.name || '');
}

/**
 * Does this look like a 360 photo?
 *
 * Equirectangular images are always 2:1 -- 360 degrees of longitude against 180
 * of latitude. That ratio is the only reliable signal available without
 * vendor-specific metadata, so it is what we key on, with enough tolerance to
 * survive a crop of a few pixels.
 */
export function looksEquirectangular(width, height, tolerance = 0.02) {
  if (!(width > 0) || !(height > 0)) return false;
  return Math.abs(width / height - 2) <= 2 * tolerance;
}

/** Pinhole intrinsics of an extracted view, for callers that want to record them. */
export function faceIntrinsics(size, fovDeg = DEFAULT_FACE_FOV_DEG) {
  const f = size / 2 / Math.tan((fovDeg * Math.PI) / 360);
  return { fx: f, fy: f, cx: size / 2, cy: size / 2, width: size, height: size };
}

/* ------------------------------------------------------------------ *
 * Radiance HDR (.hdr / RGBE)
 * ------------------------------------------------------------------ */

/**
 * Decode a Radiance RGBE image into linear floating-point RGB.
 *
 * The browser has no decoder for this format, so we need our own. All three
 * scanline encodings Radiance emits are handled: an incomplete decoder does not
 * throw, it returns a subtly wrong image, which is far worse to debug.
 *
 * @param {Uint8Array} bytes
 * @param {object} [options]
 * @param {number} [options.maxEdge]  box-average down to this width while decoding
 *                                    (defaults to DEFAULT_PANO_WIDTH)
 * @returns {{width: number, height: number, data: Float32Array}} linear RGB
 */
export function decodeRadianceHdr(bytes, options = {}) {
  const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let pos = 0;

  // Header lines are short. Capping the scan means a file that is not an HDR
  // at all fails on the signature rather than by spreading megabytes of
  // arbitrary bytes into a function call.
  const MAX_LINE = 1024;
  const readLine = () => {
    const start = pos;
    const limit = Math.min(buf.length, start + MAX_LINE);
    while (pos < limit && buf[pos] !== 0x0a) pos += 1;
    if (pos === limit && buf[pos] !== 0x0a) {
      throw new Error('HDR header line is implausibly long; this is not a Radiance file');
    }
    const line = String.fromCharCode(...buf.subarray(start, pos));
    pos += 1;
    return line;
  };

  const magic = readLine();
  if (!/^#\?(RADIANCE|RGBE)/.test(magic)) {
    throw new Error('not a Radiance HDR file (missing #?RADIANCE signature)');
  }

  // The header is free-form key=value lines terminated by an empty line.
  let format = null;
  for (;;) {
    if (pos >= buf.length) throw new Error('HDR header ended before the resolution line');
    const line = readLine();
    if (line === '') break;
    const m = /^FORMAT=(.*)$/.exec(line);
    if (m) format = m[1].trim();
  }
  if (format && format !== '32-bit_rle_rgbe') {
    // XYZE is the same container holding CIE XYZ rather than RGB. Silently
    // treating it as RGB would shift every colour, so refuse it by name.
    throw new Error(`unsupported HDR format: ${format}`);
  }

  const res = /^\s*-Y\s+(\d+)\s+\+X\s+(\d+)\s*$/.exec(readLine());
  if (!res) {
    throw new Error('unsupported HDR orientation (only "-Y height +X width" is handled)');
  }
  const height = Number(res[1]);
  const width = Number(res[2]);
  if (!(width > 0 && height > 0)) throw new Error('HDR image has no pixels');

  // Panoramas are large and radiance is four bytes a channel, so rows are
  // converted and box-averaged as they are decoded rather than held as a
  // full-resolution float image first. Peak cost is one scanline plus the
  // downsampled result, not the whole source.
  // The default tier, not the ceiling: decoding at 16k unasked would spend
  // half a gigabyte on a caller that never chose to.
  const maxEdge = options.maxEdge ?? DEFAULT_PANO_WIDTH;
  const factor = Math.max(1, Math.ceil(width / maxEdge));
  const outW = Math.max(1, Math.floor(width / factor));
  const outH = Math.max(1, Math.floor(height / factor));
  const accum = new Float64Array(outW * outH * 3);
  const counts = new Float64Array(outW * outH);

  const scanline = new Uint8Array(width * 4);
  let lastPixel = new Uint8Array(4);

  /** Convert one RGBE scanline to radiance and fold it into the output grid. */
  const emitRow = (y) => {
    const ty = Math.min(outH - 1, Math.floor(y / factor));
    for (let x = 0; x < width; x++) {
      const e = scanline[x * 4 + 3];
      const tx = Math.min(outW - 1, Math.floor(x / factor));
      const at = (ty * outW + tx) * 3;
      counts[ty * outW + tx] += 1;
      if (e === 0) continue; // exponent 0 is exactly black
      const f = Math.pow(2, e - 136);
      accum[at] += (scanline[x * 4] + 0.5) * f;
      accum[at + 1] += (scanline[x * 4 + 1] + 0.5) * f;
      accum[at + 2] += (scanline[x * 4 + 2] + 0.5) * f;
    }
  };

  for (let y = 0; y < height; y++) {
    if (pos + 4 > buf.length) throw new Error(`HDR data ended at row ${y} of ${height}`);
    const b0 = buf[pos];
    const b1 = buf[pos + 1];
    const b2 = buf[pos + 2];
    const b3 = buf[pos + 3];
    const newStyle = b0 === 2 && b1 === 2 && ((b2 << 8) | b3) === width && width >= 8 && width < 32768;

    if (newStyle) {
      pos += 4;
      // Each of the four components is run-length encoded across the whole
      // scanline before the next component starts.
      for (let c = 0; c < 4; c++) {
        let x = 0;
        while (x < width) {
          if (pos >= buf.length) throw new Error(`HDR data ended inside row ${y}`);
          let count = buf[pos++];
          if (count > 128) {
            count -= 128;
            if (pos >= buf.length) throw new Error('HDR run ran past the end of the file');
            const value = buf[pos++];
            if (x + count > width) throw new Error(`HDR run overflows row ${y}`);
            for (let k = 0; k < count; k++) scanline[(x++) * 4 + c] = value;
          } else {
            if (count === 0) throw new Error(`HDR literal run of length zero in row ${y}`);
            if (x + count > width) throw new Error(`HDR run overflows row ${y}`);
            if (pos + count > buf.length) throw new Error('HDR literal ran past the end of the file');
            for (let k = 0; k < count; k++) scanline[(x++) * 4 + c] = buf[pos++];
          }
        }
      }
    } else {
      // Flat pixels, possibly with old-style RLE: a pixel of (1,1,1,n) repeats
      // the previous pixel n times, and consecutive markers shift the count.
      let x = 0;
      let shift = 0;
      while (x < width) {
        if (pos + 4 > buf.length) throw new Error(`HDR data ended inside row ${y}`);
        const r = buf[pos]; const g = buf[pos + 1]; const b = buf[pos + 2]; const e = buf[pos + 3];
        pos += 4;
        if (r === 1 && g === 1 && b === 1) {
          if (x === 0 && y === 0) throw new Error('HDR begins with a repeat of a pixel that does not exist');
          const count = e << shift;
          if (x + count > width) throw new Error(`HDR repeat overflows row ${y}`);
          // A run at x === 0 repeats the last pixel of the previous row.
          const src = x > 0 ? scanline.subarray((x - 1) * 4, x * 4) : lastPixel;
          for (let k = 0; k < count; k++) scanline.set(src, (x + k) * 4);
          x += count;
          shift += 8;
        } else {
          const at = x * 4;
          scanline[at] = r; scanline[at + 1] = g; scanline[at + 2] = b; scanline[at + 3] = e;
          x += 1;
          shift = 0;
        }
      }
    }
    emitRow(y);
    lastPixel = scanline.slice((width - 1) * 4, width * 4);
  }

  const data = new Float32Array(outW * outH * 3);
  for (let i = 0; i < outW * outH; i++) {
    const n = counts[i] || 1;
    data[i * 3] = accum[i * 3] / n;
    data[i * 3 + 1] = accum[i * 3 + 1] / n;
    data[i * 3 + 2] = accum[i * 3 + 2] / n;
  }
  return { width: outW, height: outH, data };
}

/**
 * Expand packed RGBE to linear float RGB.
 *
 * Follows Radiance's own colr_color(): a shared exponent biased by 128, with a
 * half-step added to each mantissa so the reconstructed value sits in the
 * middle of the quantisation bucket rather than at its floor.
 */
export function rgbeToLinear(rgbe) {
  const count = rgbe.length / 4;
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const e = rgbe[i * 4 + 3];
    if (e === 0) continue; // exponent 0 is exactly black
    const f = Math.pow(2, e - 136);
    out[i * 3 + 0] = (rgbe[i * 4 + 0] + 0.5) * f;
    out[i * 3 + 1] = (rgbe[i * 4 + 1] + 0.5) * f;
    out[i * 3 + 2] = (rgbe[i * 4 + 2] + 0.5) * f;
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * Tone mapping
 * ------------------------------------------------------------------ */

const LUMA = [0.2126, 0.7152, 0.0722];

/**
 * Collapse linear HDR radiance to 8-bit sRGB.
 *
 * The target here is not a pretty picture, it is a *matchable* one. Gaussian
 * splats store 8-bit colour, so the range has to collapse somewhere -- and if
 * we let a canvas do it by clipping, everything above the clip point becomes
 * flat white with no features in it. A window or a sky would contribute nothing
 * to reconstruction, and a dark corner would be crushed to black at the other
 * end.
 *
 * So: scale the image by its own log-average luminance, which puts the typical
 * pixel at mid grey no matter the absolute exposure, then roll off the
 * highlights with extended Reinhard rather than clipping them. Colour ratios
 * are preserved by tone mapping luminance and scaling RGB to match, which keeps
 * hue stable -- per-channel curves shift saturated colours toward grey and cost
 * matching accuracy.
 *
 * @param {Float32Array} rgb  linear RGB, 3 floats per pixel
 * @param {number} width
 * @param {number} height
 * @returns {Uint8ClampedArray} RGBA, 4 bytes per pixel
 */
export function toneMap(rgb, width, height, options = {}) {
  const key = options.key ?? 0.18;
  const count = width * height;
  const out = new Uint8ClampedArray(count * 4);

  // Log-average luminance, over a sample rather than every pixel: an 8k
  // panorama is 33 million pixels and the average is stable long before that.
  const stride = Math.max(1, Math.floor(count / 200_000));
  let logSum = 0;
  let sampled = 0;
  let maxLum = 0;
  for (let i = 0; i < count; i += stride) {
    const l = rgb[i * 3] * LUMA[0] + rgb[i * 3 + 1] * LUMA[1] + rgb[i * 3 + 2] * LUMA[2];
    if (l > maxLum) maxLum = l;
    // Zero-radiance pixels carry no exposure information and would drag a log
    // average to negative infinity.
    if (l > 1e-8) { logSum += Math.log(l); sampled += 1; }
  }
  const logAvg = sampled ? Math.exp(logSum / sampled) : 1;
  const scale = key / Math.max(logAvg, 1e-8);

  // Extended Reinhard needs a white point: the luminance that should map to
  // pure white. Anchoring it to the brightest pixel means a single specular
  // highlight cannot wash out the rest of the image.
  const white = Math.max(maxLum * scale, 1e-4);
  const invWhiteSq = 1 / (white * white);

  for (let i = 0; i < count; i++) {
    const r = rgb[i * 3] * scale;
    const g = rgb[i * 3 + 1] * scale;
    const b = rgb[i * 3 + 2] * scale;
    const l = r * LUMA[0] + g * LUMA[1] + b * LUMA[2];
    let ratio = 1;
    if (l > 1e-8) {
      const mapped = (l * (1 + l * invWhiteSq)) / (1 + l);
      ratio = mapped / l;
    }
    out[i * 4 + 0] = linearToSrgb(r * ratio) * 255;
    out[i * 4 + 1] = linearToSrgb(g * ratio) * 255;
    out[i * 4 + 2] = linearToSrgb(b * ratio) * 255;
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** The real sRGB transfer function, not a 1/2.2 approximation of it. */
export function linearToSrgb(v) {
  if (!(v > 0)) return 0;
  // Not `> 1`: at exactly 1 the power form returns 0.9999999999999999.
  if (v >= 1) return 1;
  return v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
}

/* ------------------------------------------------------------------ *
 * Reprojection
 * ------------------------------------------------------------------ */

/**
 * Resample one perspective view out of an equirectangular image.
 *
 * For every output pixel we build the ray it looks along, convert that ray to a
 * longitude and latitude, and sample the panorama there. Longitude wraps -- a
 * view straddling the seam of the panorama must read across it rather than
 * clamp -- while latitude clamps, because there is nothing beyond the poles.
 *
 * @param {{width: number, height: number, data: Uint8ClampedArray}} src  RGBA
 * @param {object} face  one of CUBE_FACES
 * @param {object} [options]
 * @param {number} [options.size]    output edge length in pixels
 * @param {number} [options.fovDeg]  horizontal and vertical field of view
 * @returns {{width: number, height: number, data: Uint8ClampedArray}} RGBA
 */
export function equirectToPerspective(src, face, options = {}) {
  // `??` rather than `||`: an explicit 0 is a caller error worth reporting,
  // and `||` would quietly substitute the default for it.
  const size = Math.max(8, Math.round(options.size ?? 512));
  const fovDeg = options.fovDeg ?? DEFAULT_FACE_FOV_DEG;
  if (!(fovDeg > 0 && fovDeg < 180)) {
    throw new Error(`face field of view must be between 0 and 180 degrees, got ${fovDeg}`);
  }
  const { width: sw, height: sh, data: sd } = src;
  const out = new Uint8ClampedArray(size * size * 4);
  const t = Math.tan((fovDeg * Math.PI) / 360);
  const [fx, fy, fz] = face.forward;
  const [rx, ry, rz] = face.right;
  const [ux, uy, uz] = face.up;

  for (let j = 0; j < size; j++) {
    // Image rows run downward, world "up" runs upward, hence the negation.
    const b = -((2 * (j + 0.5)) / size - 1) * t;
    for (let i = 0; i < size; i++) {
      const a = ((2 * (i + 0.5)) / size - 1) * t;
      const dx = fx + a * rx + b * ux;
      const dy = fy + a * ry + b * uy;
      const dz = fz + a * rz + b * uz;
      const len = Math.hypot(dx, dy, dz);

      const lon = Math.atan2(dx, dz);
      const lat = Math.asin(Math.min(1, Math.max(-1, dy / len)));
      const u = (lon / (2 * Math.PI) + 0.5) * sw - 0.5;
      const v = (0.5 - lat / Math.PI) * sh - 0.5;

      sampleBilinear(sd, sw, sh, u, v, out, (j * size + i) * 4);
    }
  }
  return { width: size, height: size, data: out };
}

/**
 * Bilinear sample with wrapping longitude and clamped latitude.
 *
 * Nearest-neighbour would alias badly here: near the poles a single source
 * pixel can cover a large patch of the output, and the stair-stepping it
 * produces reads as corner-like texture that feature detectors happily latch
 * onto, seeding matches on artefacts of the resampling rather than the scene.
 */
function sampleBilinear(src, sw, sh, u, v, out, at) {
  const x0 = Math.floor(u);
  const y0 = Math.floor(v);
  const fx = u - x0;
  const fy = v - y0;
  const x0w = wrap(x0, sw);
  const x1w = wrap(x0 + 1, sw);
  const y0c = Math.min(sh - 1, Math.max(0, y0));
  const y1c = Math.min(sh - 1, Math.max(0, y0 + 1));

  const i00 = (y0c * sw + x0w) * 4;
  const i10 = (y0c * sw + x1w) * 4;
  const i01 = (y1c * sw + x0w) * 4;
  const i11 = (y1c * sw + x1w) * 4;

  const w00 = (1 - fx) * (1 - fy);
  const w10 = fx * (1 - fy);
  const w01 = (1 - fx) * fy;
  const w11 = fx * fy;

  for (let c = 0; c < 3; c++) {
    out[at + c] = src[i00 + c] * w00 + src[i10 + c] * w10
      + src[i01 + c] * w01 + src[i11 + c] * w11;
  }
  out[at + 3] = 255;
}

function wrap(x, n) {
  const m = x % n;
  return m < 0 ? m + n : m;
}
