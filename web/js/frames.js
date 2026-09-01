/**
 * Frame extraction in the browser.
 *
 * The browser already has hardware decoders for every format a phone produces,
 * so frames are extracted here rather than server-side. That removes the ffmpeg
 * dependency entirely, keeps HEIC/HEVC handling on the platform that supports
 * it, and means the server only ever receives normalised PNG frames.
 */

import {
  CUBE_FACES, DEFAULT_FACE_FOV_DEG, decodeRadianceHdr, toneMap, faceSizeFor,
  equirectToPerspective, looksEquirectangular, isHdrFile, isExrFile,
  resolvePanoWidth,
} from './pano.js';

const DEFAULT_MAX_DIM = 640;

export function isVideo(file) {
  return file.type.startsWith('video/') || /\.(mp4|mov|m4v|webm|avi|mkv)$/i.test(file.name);
}

export function isImage(file) {
  return file.type.startsWith('image/')
    || /\.(jpe?g|png|webp|gif|avif|heic|heif|bmp)$/i.test(file.name)
    || isHdrFile(file);
}

/**
 * Turn a selection of files into an ordered list of PNG frames.
 *
 * @param {File[]} files
 * @param {object} options
 * @param {number} [options.targetFrames]  how many frames to pull from a video
 * @param {number} [options.maxDim]        longest edge of each extracted frame
 * @param {number} [options.faceFovDeg]  field of view of each panorama view
 * @param {(p: {done: number, total: number, label: string}) => void} [options.onProgress]
 * @returns {Promise<{frames: Blob[], kind: 'photos'|'video'|'pano', previews: string[]}>}
 */
export async function extractFrames(files, options = {}) {
  const maxDim = options.maxDim || DEFAULT_MAX_DIM;
  const targetFrames = options.targetFrames || 32;
  const onProgress = options.onProgress || (() => {});

  // Checked against the raw selection, before anything is filtered out: an
  // .exr is not an image this build can read, so the format filter would drop
  // it and the user would be told only that nothing was usable.
  const exr = files.find(isExrFile);
  if (exr) {
    throw new Error(`${exr.name} is an OpenEXR file, which this build cannot read. `
      + 'Save it as Radiance .hdr, or as a JPEG if it is already tone mapped.');
  }

  const videos = files.filter(isVideo);
  const images = files.filter((f) => !isVideo(f) && isImage(f));
  if (!videos.length && !images.length) {
    throw new Error('Select photos or a video — nothing else can be converted.');
  }

  // A panorama has to be identified before it is treated as a photo, and for
  // ordinary image formats that means looking at its dimensions rather than
  // its name: a 360 shot from a Theta or an Insta360 arrives as a plain JPEG.
  const classified = [];
  for (const file of images) {
    classified.push({ file, pano: await isPanorama(file) });
  }
  const panos = classified.filter((c) => c.pano);
  const photos = classified.filter((c) => !c.pano);

  // Panoramas are resampled into square views, and structure-from-motion here
  // fits one shared camera to the whole capture -- which it enforces by
  // skipping every image of a different shape. Mixing would therefore drop one
  // group or the other on the server, so it is refused where the reason can
  // still be explained.
  if (panos.length && (photos.length || videos.length)) {
    const other = photos.length && videos.length ? 'photos and video'
      : (photos.length ? 'ordinary photos' : 'video');
    throw new Error(`360 panoramas cannot be combined with ${other} in one conversion — `
      + 'they become square views that no single camera model fits. '
      + 'Convert the panoramas on their own, then the rest separately.');
  }

  const frames = [];
  const previews = [];
  const total = photos.length + panos.length * CUBE_FACES.length + videos.length * targetFrames;
  let done = 0;

  // The source width, the extracted view size and the frames that get uploaded
  // are one decision, not three: a 16k source reprojected into 512-pixel views
  // discards everything it was decoded for.
  const panoWidth = resolvePanoWidth(options.panoWidth);
  const fovDeg = options.faceFovDeg || DEFAULT_FACE_FOV_DEG;
  let faceSize = 0;

  for (const { file } of panos) {
    const views = await framesFromPano(file, {
      sourceWidth: panoWidth,
      fovDeg,
      onSize: (n) => { faceSize = Math.max(faceSize, n); },
      onProgress: (i, name) => onProgress({
        done: done + i, total, label: `Reprojecting ${file.name} (${name})`,
      }),
    });
    for (const blob of views) {
      frames.push(blob);
      if (previews.length < 8) previews.push(URL.createObjectURL(blob));
    }
    done += views.length;
  }

  for (const { file } of photos) {
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
  const kind = videos.length ? 'video' : (panos.length && !photos.length ? 'pano' : 'photos');
  // faceSize travels with the result so the caller can tell the trainer what
  // resolution the frames actually carry.
  return {
    frames, kind, previews, panoCount: panos.length,
    frameSize: panos.length && !photos.length ? (faceSize || maxDim) : maxDim,
  };
}

/**
 * Is this file a 360 photo?
 *
 * Radiance files are panoramas by convention -- the format exists for
 * environment capture -- but an ordinary JPEG has to be measured. The 2:1
 * aspect ratio of an equirectangular projection is the only signal available
 * without reading vendor-specific metadata, and `createImageBitmap` is the
 * cheapest way to get dimensions without decoding into a canvas.
 */
export async function isPanorama(file) {
  if (isHdrFile(file)) return true;
  let bitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    return false; // undecodable files are reported by the photo path instead
  }
  try {
    return looksEquirectangular(bitmap.width, bitmap.height);
  } finally {
    bitmap.close?.();
  }
}

/**
 * Resample one panorama into the six perspective views the pipeline can use.
 *
 * The panorama itself never reaches the server: what goes up are ordinary
 * pinhole frames that structure-from-motion can pose, which is the whole point
 * of doing this here rather than shipping a 100 MB HDR over the network.
 */
async function framesFromPano(file, { sourceWidth, fovDeg, onProgress, onSize }) {
  const source = isHdrFile(file)
    ? await readHdr(file, sourceWidth)
    : await readLdrPano(file, sourceWidth);
  if (!looksEquirectangular(source.width, source.height, 0.06)) {
    throw new Error(`${file.name} is ${source.width}x${source.height}, which is not the 2:1 `
      + 'shape of an equirectangular 360 photo.');
  }
  // Derived from what was actually decoded, not from the tier that was asked
  // for. A 4k panorama processed at the 16k tier decodes at its own 4k -- so
  // sizing the views from the request would upscale, paying the larger tier's
  // time, memory and training cost for detail the file never had.
  const size = faceSizeFor(source.width, fovDeg);
  onSize?.(size);
  const out = [];
  for (const face of CUBE_FACES) {
    const view = equirectToPerspective(source, face, { size, fovDeg });
    out.push(await imageDataToPng(view));
    onProgress?.(out.length, face.name);
  }
  return out;
}

/** Decode a Radiance HDR and tone map it to the 8-bit colour splats store. */
async function readHdr(file, sourceWidth) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let decoded;
  try {
    decoded = decodeRadianceHdr(bytes, { maxEdge: sourceWidth });
  } catch (err) {
    throw new Error(`${file.name} could not be read as a Radiance HDR: ${err.message}`);
  }
  return {
    width: decoded.width,
    height: decoded.height,
    data: toneMap(decoded.data, decoded.width, decoded.height),
  };
}

/** Decode an already-tone-mapped equirectangular image through the canvas. */
async function readLdrPano(file, sourceWidth) {
  let bitmap;
  try {
    // Resizing during decode keeps a 16k panorama from ever existing as a
    // full-size bitmap; the aspect ratio is preserved so the 2:1 check still
    // means what it says.
    // Only ever downscale. Enlarging a 4k panorama to 16k would cost the time
    // and memory of the larger tier while adding no detail at all, so a source
    // smaller than the requested tier is decoded at its own size.
    const decodeAt = Math.min(sourceWidth, await naturalWidth(file));
    bitmap = await createImageBitmap(file, {
      resizeWidth: decodeAt,
      resizeQuality: 'high',
      imageOrientation: 'from-image',
    });
  } catch {
    throw new Error(`${file.name} could not be decoded by this browser.`);
  }
  try {
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext('2d', { alpha: false, willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    const image = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    return { width: image.width, height: image.height, data: image.data };
  } finally {
    bitmap.close?.();
  }
}

/** Wrap raw RGBA in a canvas and encode it as the PNG the server expects. */
async function imageDataToPng({ width, height, data }) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { alpha: false });
  ctx.putImageData(new ImageData(new Uint8ClampedArray(data.buffer, data.byteOffset, data.length),
    width, height), 0, 0);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
  if (!blob) throw new Error('This browser refused to encode a panorama view.');
  return blob;
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

/** The panorama's own width, so a tier can never upscale it. */
async function naturalWidth(file) {
  const probe = await createImageBitmap(file);
  try {
    return probe.width;
  } finally {
    probe.close?.();
  }
}
