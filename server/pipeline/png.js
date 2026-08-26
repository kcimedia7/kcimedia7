import zlib from 'node:zlib';

/**
 * Minimal PNG codec built on node's zlib.
 *
 * The browser does all format decoding at capture time and uploads frames as
 * 8-bit non-interlaced PNG, so the server only has to handle that subset —
 * which keeps the pipeline dependency-free.
 */

const SIG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const CHANNELS = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };

export function decodePng(buffer) {
  if (buffer.length < 8 || !buffer.subarray(0, 8).equals(SIG)) throw new Error('not a PNG file');

  let ihdr = null;
  let palette = null;
  let transparency = null;
  const idat = [];

  let off = 8;
  while (off + 8 <= buffer.length) {
    const length = buffer.readUInt32BE(off);
    const type = buffer.toString('latin1', off + 4, off + 8);
    const start = off + 8;
    const end = start + length;
    if (end > buffer.length) throw new Error('PNG chunk runs past the end of the file');

    if (type === 'IHDR') {
      ihdr = {
        width: buffer.readUInt32BE(start),
        height: buffer.readUInt32BE(start + 4),
        bitDepth: buffer[start + 8],
        colorType: buffer[start + 9],
        compression: buffer[start + 10],
        filter: buffer[start + 11],
        interlace: buffer[start + 12],
      };
    } else if (type === 'PLTE') {
      palette = buffer.subarray(start, end);
    } else if (type === 'tRNS') {
      transparency = buffer.subarray(start, end);
    } else if (type === 'IDAT') {
      idat.push(buffer.subarray(start, end));
    } else if (type === 'IEND') {
      break;
    }
    off = end + 4; // skip the CRC
  }

  if (!ihdr) throw new Error('PNG has no IHDR chunk');
  if (ihdr.interlace !== 0) throw new Error('interlaced PNGs are not supported');
  if (ihdr.bitDepth !== 8) throw new Error(`unsupported PNG bit depth: ${ihdr.bitDepth}`);
  const channels = CHANNELS[ihdr.colorType];
  if (!channels) throw new Error(`unsupported PNG colour type: ${ihdr.colorType}`);

  const raw = zlib.inflateSync(Buffer.concat(idat));
  const { width, height } = ihdr;
  const bpp = channels; // bitDepth is always 8 here
  const stride = width * bpp;
  const out = Buffer.allocUnsafe(height * stride);

  let pos = 0;
  for (let y = 0; y < height; y++) {
    const filter = raw[pos++];
    const line = raw.subarray(pos, pos + stride);
    pos += stride;
    const cur = out.subarray(y * stride, (y + 1) * stride);
    const prev = y > 0 ? out.subarray((y - 1) * stride, y * stride) : null;
    unfilter(filter, line, cur, prev, bpp, stride);
  }

  return toRgba({ ...ihdr, channels }, out, palette, transparency);
}

function unfilter(filter, line, cur, prev, bpp, stride) {
  switch (filter) {
    case 0:
      line.copy(cur);
      break;
    case 1:
      for (let i = 0; i < stride; i++) {
        cur[i] = (line[i] + (i >= bpp ? cur[i - bpp] : 0)) & 0xff;
      }
      break;
    case 2:
      for (let i = 0; i < stride; i++) {
        cur[i] = (line[i] + (prev ? prev[i] : 0)) & 0xff;
      }
      break;
    case 3:
      for (let i = 0; i < stride; i++) {
        const a = i >= bpp ? cur[i - bpp] : 0;
        const b = prev ? prev[i] : 0;
        cur[i] = (line[i] + ((a + b) >> 1)) & 0xff;
      }
      break;
    case 4:
      for (let i = 0; i < stride; i++) {
        const a = i >= bpp ? cur[i - bpp] : 0;
        const b = prev ? prev[i] : 0;
        const c = prev && i >= bpp ? prev[i - bpp] : 0;
        cur[i] = (line[i] + paeth(a, b, c)) & 0xff;
      }
      break;
    default:
      throw new Error(`unsupported PNG filter type: ${filter}`);
  }
}

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  return pb <= pc ? b : c;
}

function toRgba(ihdr, bytes, palette, transparency) {
  const { width, height, colorType, channels } = ihdr;
  const px = width * height;
  const data = Buffer.allocUnsafe(px * 4);
  for (let i = 0; i < px; i++) {
    const s = i * channels;
    const d = i * 4;
    if (colorType === 6) {
      data[d] = bytes[s]; data[d + 1] = bytes[s + 1]; data[d + 2] = bytes[s + 2]; data[d + 3] = bytes[s + 3];
    } else if (colorType === 2) {
      data[d] = bytes[s]; data[d + 1] = bytes[s + 1]; data[d + 2] = bytes[s + 2]; data[d + 3] = 255;
    } else if (colorType === 0) {
      data[d] = data[d + 1] = data[d + 2] = bytes[s]; data[d + 3] = 255;
    } else if (colorType === 4) {
      data[d] = data[d + 1] = data[d + 2] = bytes[s]; data[d + 3] = bytes[s + 1];
    } else if (colorType === 3) {
      if (!palette) throw new Error('indexed PNG has no palette');
      const idx = bytes[s] * 3;
      data[d] = palette[idx]; data[d + 1] = palette[idx + 1]; data[d + 2] = palette[idx + 2];
      data[d + 3] = transparency && bytes[s] < transparency.length ? transparency[bytes[s]] : 255;
    }
  }
  return { width, height, data };
}

export function encodePng({ width, height, data }) {
  const stride = width * 4;
  const raw = Buffer.allocUnsafe(height * (stride + 1));
  for (let y = 0; y < height; y++) {
    raw[y * (stride + 1)] = 0; // no per-line filter; thumbnails are small
    data.copy
      ? data.copy(raw, y * (stride + 1) + 1, y * stride, (y + 1) * stride)
      : Buffer.from(data.buffer, data.byteOffset + y * stride, stride).copy(raw, y * (stride + 1) + 1);
  }
  const idat = zlib.deflateSync(raw, { level: 6 });

  const ihdr = Buffer.allocUnsafe(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;   // bit depth
  ihdr[9] = 6;   // RGBA
  ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;

  return Buffer.concat([SIG, chunk('IHDR', ihdr), chunk('IDAT', idat), chunk('IEND', Buffer.alloc(0))]);
}

function chunk(type, body) {
  const out = Buffer.allocUnsafe(body.length + 12);
  out.writeUInt32BE(body.length, 0);
  out.write(type, 4, 'latin1');
  body.copy(out, 8);
  out.writeUInt32BE(crc32(out.subarray(4, 8 + body.length)) >>> 0, 8 + body.length);
  return out;
}

const CRC_TABLE = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();

function crc32(buf) {
  let c = -1;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return c ^ -1;
}
