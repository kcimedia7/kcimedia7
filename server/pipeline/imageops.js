/** Small float-image helpers shared by the preview reconstruction. */

/** Area-average downscale of an RGBA image to fit within `maxDim`. */
export function fitWithin({ width, height, data }, maxDim) {
  const k = Math.max(width, height) / maxDim;
  if (k <= 1) return { width, height, data };
  const w = Math.max(1, Math.round(width / k));
  const h = Math.max(1, Math.round(height / k));
  const out = Buffer.alloc(w * h * 4);
  for (let y = 0; y < h; y++) {
    const y0 = Math.floor((y * height) / h);
    const y1 = Math.max(y0 + 1, Math.floor(((y + 1) * height) / h));
    for (let x = 0; x < w; x++) {
      const x0 = Math.floor((x * width) / w);
      const x1 = Math.max(x0 + 1, Math.floor(((x + 1) * width) / w));
      let r = 0, g = 0, b = 0, a = 0, n = 0;
      for (let sy = y0; sy < y1; sy++) {
        for (let sx = x0; sx < x1; sx++) {
          const s = (sy * width + sx) * 4;
          r += data[s]; g += data[s + 1]; b += data[s + 2]; a += data[s + 3];
          n++;
        }
      }
      const d = (y * w + x) * 4;
      out[d] = r / n; out[d + 1] = g / n; out[d + 2] = b / n; out[d + 3] = a / n;
    }
  }
  return { width: w, height: h, data: out };
}

export function luminance(image) {
  const { width, height, data } = image;
  const out = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    out[i] = (0.2126 * data[i * 4] + 0.7152 * data[i * 4 + 1] + 0.0722 * data[i * 4 + 2]) / 255;
  }
  return out;
}

/** Separable box blur, repeated to approximate a gaussian. */
export function boxBlur(src, width, height, radius, passes = 2) {
  if (radius < 1) return Float32Array.from(src);
  let a = Float32Array.from(src);
  let b = new Float32Array(src.length);
  for (let p = 0; p < passes; p++) {
    blurAxis(a, b, width, height, radius, true);
    blurAxis(b, a, width, height, radius, false);
  }
  return a;
}

function blurAxis(src, dst, width, height, radius, horizontal) {
  const outer = horizontal ? height : width;
  const inner = horizontal ? width : height;
  const step = horizontal ? 1 : width;
  for (let o = 0; o < outer; o++) {
    const base = horizontal ? o * width : o;
    let sum = 0;
    let count = 0;
    for (let i = -radius; i <= radius; i++) {
      const c = clampIndex(i, inner);
      sum += src[base + c * step];
      count++;
    }
    for (let i = 0; i < inner; i++) {
      dst[base + i * step] = sum / count;
      const outIdx = clampIndex(i - radius, inner);
      const inIdx = clampIndex(i + radius + 1, inner);
      sum += src[base + inIdx * step] - src[base + outIdx * step];
    }
  }
}

function clampIndex(i, n) {
  return i < 0 ? 0 : i >= n ? n - 1 : i;
}

/** Sobel gradient magnitude, normalised to roughly 0..1. */
export function gradientMagnitude(gray, width, height) {
  const out = new Float32Array(width * height);
  const at = (x, y) => gray[clampIndex(y, height) * width + clampIndex(x, width)];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const gx = at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)
        - at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1);
      const gy = at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)
        - at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1);
      out[y * width + x] = Math.min(1, Math.hypot(gx, gy) / 4);
    }
  }
  return out;
}

/** Median RGB of the outer frame of the image — a cheap backdrop estimate. */
export function borderColor({ width, height, data }, band = 0.06) {
  const bx = Math.max(1, Math.round(width * band));
  const by = Math.max(1, Math.round(height * band));
  const rs = [], gs = [], bs = [];
  const take = (x, y) => {
    const s = (y * width + x) * 4;
    rs.push(data[s]); gs.push(data[s + 1]); bs.push(data[s + 2]);
  };
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (x < bx || x >= width - bx || y < by || y >= height - by) take(x, y);
    }
  }
  return [median(rs) / 255, median(gs) / 255, median(bs) / 255];
}

function median(arr) {
  if (!arr.length) return 0;
  arr.sort((a, b) => a - b);
  return arr[arr.length >> 1];
}

export function normaliseField(field) {
  let lo = Infinity, hi = -Infinity;
  for (const v of field) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo;
  const out = new Float32Array(field.length);
  if (span < 1e-6) return out;
  for (let i = 0; i < field.length; i++) out[i] = (field[i] - lo) / span;
  return out;
}
