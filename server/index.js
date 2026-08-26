import http from 'node:http';
import { createApiRouter } from './api.js';
import { sendError, sendFile, resolveStatic } from './http/router.js';
import * as store from './store.js';
import * as jobs from './jobs.js';
import { detectCapabilities } from './pipeline/backends.js';
import { PORT, HOST, WEB_DIR, DATA_DIR } from './config.js';

const api = createApiRouter();

export function createServer() {
  return http.createServer(async (req, res) => {
    const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);

    // Same-origin app; CORS is only opened for reads so an external viewer can
    // pull an exported splat if someone wants that.
    if (req.method === 'OPTIONS') {
      res.writeHead(204, {
        'access-control-allow-origin': '*',
        'access-control-allow-methods': 'GET, HEAD, OPTIONS',
        'access-control-allow-headers': 'content-type',
      });
      return res.end();
    }

    try {
      const matched = api.match(req.method === 'HEAD' ? 'GET' : req.method, url.pathname);
      if (matched?.route) {
        return await matched.route.handler(req, res, { params: matched.params, query: url.searchParams });
      }
      if (matched?.methodMismatch) return sendError(res, 405, 'method not allowed');

      if (url.pathname.startsWith('/api/')) return sendError(res, 404, 'no such endpoint');

      const file = resolveStatic(WEB_DIR, url.pathname);
      if (file) {
        return await sendFile(req, res, file, {
          cache: file.endsWith('.html') ? 'no-cache' : 'public, max-age=300',
        });
      }

      // Client-side routes (/library, /asset/<id>) render from index.html.
      const index = resolveStatic(WEB_DIR, '/index.html');
      if (index) return await sendFile(req, res, index, { cache: 'no-cache' });

      sendError(res, 404, 'not found');
    } catch (err) {
      console.error(`${req.method} ${url.pathname} failed:`, err);
      if (!res.headersSent) sendError(res, err.status || 500, err.message || 'internal error');
      else res.end();
    }
  });
}

export async function start() {
  await store.init();
  const caps = await detectCapabilities();
  await jobs.resumeInterrupted();

  const server = createServer();
  await new Promise((resolve) => server.listen(PORT, HOST, resolve));

  console.log(`SplatWorks listening on http://${HOST}:${PORT}`);
  console.log(`  data directory : ${DATA_DIR}`);
  console.log(`  backend        : ${caps.backend}${caps.forced ? ' (forced)' : ''}`);
  for (const reason of caps.reasons) console.log(`  · ${reason}`);
  if (caps.backend === 'preview') {
    console.log('  Preview mode builds a fast proxy splat from your frames rather than a');
    console.log('  true photogrammetric reconstruction. See README.md to enable COLMAP.');
  }
  return server;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  start().catch((err) => {
    console.error('failed to start:', err);
    process.exit(1);
  });
}
