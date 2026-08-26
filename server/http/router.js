import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { createReadStream } from 'node:fs';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.ply': 'application/octet-stream',
  '.splat': 'application/octet-stream',
  '.txt': 'text/plain; charset=utf-8',
};

export function mimeFor(file) {
  return MIME[path.extname(file).toLowerCase()] || 'application/octet-stream';
}

export class Router {
  constructor() {
    this.routes = [];
  }

  add(method, pattern, handler) {
    const keys = [];
    // Escape regex metacharacters first (so `export.ply` matches literally),
    // then expand `:param` and a trailing `*`.
    const source = pattern
      .replace(/[.+?^${}()|[\]\\]/g, '\\$&')
      .replace(/\/:([A-Za-z0-9_]+)/g, (_, key) => { keys.push(key); return '/([^/]+)'; })
      .replace(/\*$/, '(.*)');
    this.routes.push({ method, regex: new RegExp(`^${source}$`), keys, handler });
    return this;
  }

  get(p, h) { return this.add('GET', p, h); }
  post(p, h) { return this.add('POST', p, h); }
  patch(p, h) { return this.add('PATCH', p, h); }
  delete(p, h) { return this.add('DELETE', p, h); }

  match(method, pathname) {
    let pathExists = false;
    for (const route of this.routes) {
      const m = route.regex.exec(pathname);
      if (!m) continue;
      pathExists = true;
      if (route.method !== method) continue;
      const params = {};
      route.keys.forEach((k, i) => { params[k] = decodeURIComponent(m[i + 1]); });
      return { route, params };
    }
    return pathExists ? { methodMismatch: true } : null;
  }
}

export function sendJson(res, status, body) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(payload),
    'cache-control': 'no-store',
  });
  res.end(payload);
}

export function sendError(res, status, message, extra = {}) {
  sendJson(res, status, { error: message, ...extra });
}

export async function readJsonBody(req, limit = 4 * 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > limit) throw Object.assign(new Error('request body is too large'), { status: 413 });
    chunks.push(chunk);
  }
  if (!chunks.length) return {};
  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch {
    throw Object.assign(new Error('request body is not valid JSON'), { status: 400 });
  }
}

/**
 * Serve a file with byte-range support. Splat clouds are large binaries, so
 * range requests let the viewer resume an interrupted load.
 */
export async function sendFile(req, res, filePath, { download, cache = 'no-cache' } = {}) {
  let stat;
  try {
    stat = await fsp.stat(filePath);
  } catch {
    sendError(res, 404, 'file not found');
    return;
  }
  if (!stat.isFile()) {
    sendError(res, 404, 'file not found');
    return;
  }

  const etag = `W/"${stat.size.toString(16)}-${Math.floor(stat.mtimeMs).toString(16)}"`;
  const headers = {
    'content-type': mimeFor(filePath),
    'accept-ranges': 'bytes',
    'cache-control': cache,
    etag,
    'last-modified': stat.mtime.toUTCString(),
  };
  if (download) {
    headers['content-disposition'] = `attachment; filename="${sanitiseFilename(download)}"`;
  }

  if (req.headers['if-none-match'] === etag) {
    res.writeHead(304, { etag, 'cache-control': cache });
    res.end();
    return;
  }

  const range = parseRange(req.headers.range, stat.size);
  if (range === 'invalid') {
    res.writeHead(416, { 'content-range': `bytes */${stat.size}` });
    res.end();
    return;
  }

  if (range) {
    headers['content-range'] = `bytes ${range.start}-${range.end}/${stat.size}`;
    headers['content-length'] = range.end - range.start + 1;
    res.writeHead(206, headers);
    if (req.method === 'HEAD') return res.end();
    createReadStream(filePath, range).pipe(res);
    return;
  }

  headers['content-length'] = stat.size;
  res.writeHead(200, headers);
  if (req.method === 'HEAD') return res.end();
  createReadStream(filePath).pipe(res);
}

export function parseRange(header, size) {
  if (!header) return null;
  const m = /^bytes=(\d*)-(\d*)$/.exec(header.trim());
  if (!m) return null;
  const [, rawStart, rawEnd] = m;
  if (rawStart === '' && rawEnd === '') return 'invalid';
  let start;
  let end;
  if (rawStart === '') {
    const len = Number(rawEnd);
    if (!len) return 'invalid';
    start = Math.max(0, size - len);
    end = size - 1;
  } else {
    start = Number(rawStart);
    end = rawEnd === '' ? size - 1 : Math.min(Number(rawEnd), size - 1);
  }
  if (!Number.isFinite(start) || !Number.isFinite(end) || start > end || start >= size) return 'invalid';
  return { start, end };
}

export function sanitiseFilename(name) {
  const cleaned = String(name)
    .replace(/[^\w.\- ]+/g, '_')   // drop path separators, quotes and control bytes
    .replace(/^[.\s_]+/, '')        // never produce a dotfile or leading padding
    .slice(0, 120)
    .trim();
  // Whatever is left has to be nameable; a run of punctuation is not.
  return /[a-zA-Z0-9]/.test(cleaned) ? cleaned : 'download';
}

/** Resolve a URL path inside `root`, refusing anything that escapes it. */
export function resolveStatic(root, urlPath) {
  const decoded = decodeURIComponent(urlPath.split('?')[0]);
  const rel = decoded === '/' ? '/index.html' : decoded;
  const full = path.join(root, path.normalize(rel));
  if (full !== root && !full.startsWith(root + path.sep)) return null;
  if (!fs.existsSync(full)) return null;
  if (fs.statSync(full).isDirectory()) {
    const index = path.join(full, 'index.html');
    return fs.existsSync(index) ? index : null;
  }
  return full;
}
