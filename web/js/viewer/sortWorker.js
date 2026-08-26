/**
 * Depth sorting worker.
 *
 * Alpha blending needs the splats drawn back to front, and that order changes
 * whenever the camera or the edit transform moves. Sorting hundreds of
 * thousands of gaussians every frame on the main thread would stall the UI, so
 * it happens here with a 16-bit counting sort — O(n), no comparisons.
 *
 * The main thread sends the depth as an affine function of the ORIGINAL
 * position (`A · p + B`), which folds the view matrix and the edit transform
 * into three floats, so positions are uploaded once and never again.
 */

let positions = null;
let count = 0;
let order = null;
let scratch = null;
let pending = null;
let running = false;

const BUCKETS = 65536;

self.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === 'load') {
    positions = new Float32Array(msg.positions);
    count = positions.length / 3;
    order = new Uint32Array(count);
    scratch = new Float32Array(count);
    self.postMessage({ type: 'ready', count });
    return;
  }
  if (msg.type === 'sort') {
    // Only the newest request matters; drop any that piled up behind it.
    pending = msg;
    if (!running) drain();
  }
};

function drain() {
  running = true;
  while (pending) {
    const job = pending;
    pending = null;
    try {
      const result = sort(job.axis, job.bias);
      // The index buffer is transferred out, so allocate a fresh one next time.
      self.postMessage({ type: 'sorted', order: result.buffer, count, generation: job.generation },
        [result.buffer]);
      order = new Uint32Array(count);
    } catch (err) {
      self.postMessage({ type: 'error', message: String(err && err.message || err) });
    }
  }
  running = false;
}

function sort(axis, bias) {
  const [ax, ay, az] = axis;
  let min = Infinity;
  let max = -Infinity;

  for (let i = 0; i < count; i++) {
    const d = ax * positions[i * 3] + ay * positions[i * 3 + 1] + az * positions[i * 3 + 2] + bias;
    scratch[i] = d;
    if (d < min) min = d;
    if (d > max) max = d;
  }

  const out = order;
  if (!(max > min)) {
    for (let i = 0; i < count; i++) out[i] = i;
    return out;
  }

  // Camera-space z is negative in front of the eye, so ascending depth is
  // farthest-first — exactly the back-to-front order blending needs.
  const scale = (BUCKETS - 1) / (max - min);
  const counts = new Uint32Array(BUCKETS);
  const bucket = new Uint16Array(count);

  for (let i = 0; i < count; i++) {
    const b = (scratch[i] - min) * scale | 0;
    bucket[i] = b;
    counts[b]++;
  }
  // Prefix-sum the histogram into starting offsets.
  let offset = 0;
  for (let b = 0; b < BUCKETS; b++) {
    const c = counts[b];
    counts[b] = offset;
    offset += c;
  }
  for (let i = 0; i < count; i++) out[counts[bucket[i]]++] = i;
  return out;
}
