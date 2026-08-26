import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { Router, parseRange, sanitiseFilename, resolveStatic, mimeFor } from '../server/http/router.js';
import { WEB_DIR } from '../server/config.js';

test('routes match methods and extract parameters', () => {
  const r = new Router();
  r.get('/api/assets', () => 'list');
  r.get('/api/assets/:id', () => 'one');
  r.post('/api/assets/:id/reconvert', () => 'again');

  assert.equal(r.match('GET', '/api/assets').route.handler(), 'list');
  const one = r.match('GET', '/api/assets/abc123');
  assert.equal(one.params.id, 'abc123');
  assert.equal(r.match('POST', '/api/assets/x/reconvert').params.id, 'x');
  assert.equal(r.match('GET', '/api/nope'), null);
});

test('a known path with the wrong method reports a method mismatch', () => {
  const r = new Router();
  r.get('/api/assets', () => {});
  assert.equal(r.match('DELETE', '/api/assets').methodMismatch, true);
});

test('dots in a route are literal, not wildcards', () => {
  const r = new Router();
  r.get('/api/assets/:id/export.ply', () => 'ply');
  assert.ok(r.match('GET', '/api/assets/x/export.ply'));
  assert.equal(r.match('GET', '/api/assets/x/exportXply'), null);
});

test('a parameter cannot swallow extra path segments', () => {
  const r = new Router();
  r.get('/api/assets/:id', () => {});
  assert.equal(r.match('GET', '/api/assets/a/b'), null);
});

test('byte ranges parse the forms clients actually send', () => {
  assert.deepEqual(parseRange('bytes=0-99', 1000), { start: 0, end: 99 });
  assert.deepEqual(parseRange('bytes=500-', 1000), { start: 500, end: 999 });
  assert.deepEqual(parseRange('bytes=-100', 1000), { start: 900, end: 999 });
  assert.deepEqual(parseRange('bytes=0-99999', 1000), { start: 0, end: 999 });
  assert.equal(parseRange(undefined, 1000), null);
  assert.equal(parseRange('bytes=900-100', 1000), 'invalid');
  assert.equal(parseRange('bytes=5000-6000', 1000), 'invalid');
  assert.equal(parseRange('bytes=-', 1000), 'invalid');
});

test('download filenames are stripped of path and quote characters', () => {
  assert.equal(sanitiseFilename('../../etc/passwd'), 'etc_passwd');
  assert.equal(sanitiseFilename('.hidden.ply'), 'hidden.ply');
  assert.equal(sanitiseFilename('a "quoted" name.ply'), 'a _quoted_ name.ply');
  assert.equal(sanitiseFilename('!!!'), 'download');
  assert.ok(sanitiseFilename('x'.repeat(500)).length <= 120);
});

test('static resolution refuses to escape the web root', () => {
  assert.equal(resolveStatic(WEB_DIR, '/../server/index.js'), null);
  assert.equal(resolveStatic(WEB_DIR, '/../../etc/passwd'), null);
  assert.equal(resolveStatic(WEB_DIR, '/%2e%2e/%2e%2e/etc/passwd'), null);
  assert.equal(resolveStatic(WEB_DIR, '/nope.js'), null);
  assert.equal(resolveStatic(WEB_DIR, '/'), path.join(WEB_DIR, 'index.html'));
  assert.equal(resolveStatic(WEB_DIR, '/js/app.js'), path.join(WEB_DIR, 'js', 'app.js'));
});

test('content types cover what the app serves', () => {
  assert.match(mimeFor('/js/app.js'), /javascript/);
  assert.match(mimeFor('/css/app.css'), /text\/css/);
  assert.equal(mimeFor('/x/cloud.splat'), 'application/octet-stream');
  assert.equal(mimeFor('/x/thumbnail.png'), 'image/png');
});
