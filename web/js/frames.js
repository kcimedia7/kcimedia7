/**
 * Frame extraction in the browser.
 *
 * The browser already has hardware decoders for every format a phone produces,
 * so frames are extracted here rather than server-side. That removes the ffmpeg
 * dependency entirely, keeps HEIC/HEVC handling on the platform that supports
 * it, and means the server only ever receives normalised PNG frames.
 */

const DEFAULT_MAX_DIM = 640;

export function isVideo(file) {
  return file.type.startsWith('video/') || /\.(mp4|mov|m4v|webm|avi|mkv)$/i.test(file.name);
}

export function isImage(file) {
  return file.type.startsWith('image/') || /\.(jpe?g|png|webp|gif|avif|heic|heif|bmp)$/i.test(file.name);
}

/**
 * Turn a selection of files into an ordered list of PNG frames.
 *
 * @param {File[]} files
 * @param {object} options
 * @param {number} [options.targetFrames]  how many frames to pull from a video
 * @param {number} [options.maxDim]        longest edge of each extracted frame
 * @param {(p: {done: number, total: number, label: string}) => void} [options.onProgress]
 * @returns {Promise<{frames: Blob[], kind: 'photos'|'video', previews: string[]}>}
 */
export async function extractFrames(files, options = {}) {
  const maxDim = options.maxDim || DEFAULT_MAX_DIM;
  const targetFrames = options.targetFrames || 32;
  const onProgress = options.onProgress || (() => {});

  const videos = files.filter(isVideo);
  const images = files.filter((f) => !isVideo(f) && isImage(f));
  if (!videos.length && !images.length) {
    throw new Error('Select photos or a video — nothing else can be converted.');
  }

  const frames = [];
  const previews = [];
  const total = images.length + videos.length * targetFrames;
  let done = 0;

  for (const file of images) {
    const blob = await frameFromImage(file, maxDim);
    frames.push(blob);
    if (previews.length < 8) previews.push(URL.createObjectURL(blob));
    done += 1;
    onProgress({ done, total, label: `Reading ${file.name}` });
  }

  for (const file of videos) {
    const perVideo = Math.max(2, Math.round(targetFrames / videos.length));
    const extracted = await framesFromVideo(file, {
      count: perVideo,
      maxDim,
      onProgress: (i) => onProgress({ done: done + i, total, label: `Extracting frames from ${file.name}` }),
    });
    for (const blob of extracted) {
      frames.push(blob);
      if (previews.length < 8) previews.push(URL.createObjectURL(blob));
    }
    done += extracted.length;
  }

  if (!frames.length) throw new Error('No frames could be read from that selection.');
  return { frames, kind: videos.length ? 'video' : 'photos', previews };
}

async function frameFromImage(file, maxDim) {
  let bitmap;
  try {
    // `from-image` applies the EXIF rotation phones write, so portrait shots
    // are not silently reconstructed on their side.
    bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch {
    throw new Error(`${file.name} could not be decoded by this browser.`);
  }
  try {
    return await drawToPng(bitmap, bitmap.width, bitmap.height, maxDim);
  } finally {
    bitmap.close?.();
  }
}

async function framesFromVideo(file, { count, maxDim, onProgress }) {
  const url = URL.createObjectURL(file);
  const video = document.createElement('video');
  video.preload = 'auto';
  video.muted = true;
  video.playsInline = true;
  video.crossOrigin = 'anonymous';
  video.src = url;

  try {
    await once(video, 'loadedmetadata', 30_000, `${file.name} could not be opened as a video.`);
    const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
    if (!duration) throw new Error(`${file.name} reports no duration, so frames cannot be sampled.`);

    // Skip the very start and end: the first frames are often blurred by the
    // hand reaching for the shutter, the last by putting the phone down.
    const start = Math.min(0.08 * duration, 0.4);
    const end = duration - Math.min(0.08 * duration, 0.4);
    const span = Math.max(end - start, duration * 0.5);

    const out = [];
    for (let i = 0; i < count; i++) {
      const t = start + (span * i) / Math.max(1, count - 1);
      await seek(video, Math.min(t, duration - 0.02));
      out.push(await drawToPng(video, video.videoWidth, video.videoHeight, maxDim));
      onProgress?.(i + 1);
    }
    return out;
  } finally {
    video.removeAttribute('src');
    video.load();
    URL.revokeObjectURL(url);
  }
}

function seek(video, time) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error('Timed out seeking through the video.'));
    }, 20_000);
    const onSeeked = () => { cleanup(); resolve(); };
    const onError = () => { cleanup(); reject(new Error('The video failed while seeking.')); };
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener('seeked', onSeeked);
      video.removeEventListener('error', onError);
    };
    video.addEventListener('seeked', onSeeked, { once: true });
    video.addEventListener('error', onError, { once: true });
    video.currentTime = time;
  });
}

async function drawToPng(source, width, height, maxDim) {
  if (!width || !height) throw new Error('A frame arrived with no dimensions.');
  const k = Math.min(1, maxDim / Math.max(width, height));
  const w = Math.max(1, Math.round(width * k));
  const h = Math.max(1, Math.round(height * k));

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { alpha: false, willReadFrequently: false });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, w, h);

  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
  if (!blob) throw new Error('This browser refused to encode a frame.');
  return blob;
}

function once(target, event, timeoutMs, message) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { cleanup(); reject(new Error(message)); }, timeoutMs);
    const onEvent = () => { cleanup(); resolve(); };
    const onError = () => { cleanup(); reject(new Error(message)); };
    const cleanup = () => {
      clearTimeout(timer);
      target.removeEventListener(event, onEvent);
      target.removeEventListener('error', onError);
    };
    target.addEventListener(event, onEvent, { once: true });
    target.addEventListener('error', onError, { once: true });
  });
}
