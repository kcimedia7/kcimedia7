import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fsp from 'node:fs/promises';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * The standalone build inlines server and viewer modules and patches the few
 * Node-isms that cannot run in a browser. Those patches are string matches
 * against real source, so they rot silently when the source moves — unless
 * something checks. The build script throws on a missing target; this runs it
 * and inspects what came out.
 */

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(await fsp.mkdtemp(path.join(os.tmpdir(), 'splat-build-')), 'page.html');

execFileSync(process.execPath, [path.join(ROOT, 'tools', 'build-demo.mjs'), '--out', out], {
  cwd: ROOT,
  stdio: 'pipe',
});
const page = await fsp.readFile(out, 'utf8');

test.after(() => fs.rmSync(path.dirname(out), { recursive: true, force: true }));

test('the build produces one self-contained page', () => {
  assert.ok(page.length > 50_000, 'the page should carry the whole bundle');
  assert.match(page, /<title>SplatWorks<\/title>/);
  assert.match(page, /<script type="module">/);
});

test('no external resource is referenced except Google Fonts', () => {
  const urls = [...page.matchAll(/(?:src|href)="(https?:\/\/[^"]+)"/g)].map((m) => m[1]);
  for (const url of urls) {
    // preconnect hints carry no path, so the trailing slash is optional.
    assert.match(url, /^https:\/\/fonts\.(googleapis|gstatic)\.com(\/|$)/, `unexpected external URL: ${url}`);
  }
});

test('the page is pure ASCII, since it cannot declare a charset', () => {
  const stray = page.match(/[^\x00-\x7F]/g);
  assert.equal(stray, null, `non-ASCII characters would mojibake: ${[...new Set(stray || [])].join(' ')}`);
});

test('no Node-only API survives into the browser bundle', () => {
  // Buffer, require and process would all throw on a page.
  const script = page.slice(page.indexOf('<script type="module">'));
  for (const pattern of [/\bBuffer\./, /\brequire\(/, /\bprocess\.env\b/, /\bmodule\.exports\b/]) {
    assert.equal(script.match(pattern), null, `bundle still contains ${pattern}`);
  }
});

test('module syntax is stripped so everything shares one scope', () => {
  const script = page.slice(page.indexOf('<script type="module">'));
  assert.equal(script.match(/^import\s/m), null, 'leftover import');
  assert.equal(script.match(/^export\s/m), null, 'leftover export');
});

test('the inlined pipeline and renderer are actually present', () => {
  for (const symbol of [
    'function reconstructPreview', 'function encodeSplatBuffer', 'function decodeSplatBuffer',
    'function encodePly', 'function applyEdits', 'class SplatViewer', 'class OrbitCamera',
    'VERTEX_SHADER', 'FRAGMENT_SHADER',
  ]) {
    assert.ok(page.includes(symbol), `bundle is missing ${symbol}`);
  }
});

test('the sort worker is embedded rather than fetched', () => {
  assert.match(page, /SORT_WORKER_URL = URL\.createObjectURL/);
  assert.match(page, /new Worker\(SORT_WORKER_URL\)/);
  assert.equal(page.match(/new URL\('\.\/sortWorker\.js'/), null, 'still points at a module URL');
});

test('helpers that would collide in one scope were renamed apart', () => {
  const script = page.slice(page.indexOf('<script type="module">'));
  for (const name of ['clamp01', 'normalize', 'cross', 'boundsOf']) {
    const declarations = script.match(new RegExp(`^function ${name}\\(`, 'gm')) || [];
    assert.ok(declarations.length <= 1, `${name} is declared ${declarations.length} times in one scope`);
  }
});

test('netlify.toml points at what the build actually writes', async () => {
  const toml = await fsp.readFile(path.join(ROOT, 'netlify.toml'), 'utf8');
  assert.match(toml, /command = "node tools\/build-demo\.mjs"/);
  assert.match(toml, /publish = "dist"/);

  // The default build must put a page where netlify.toml publishes from.
  execFileSync(process.execPath, [path.join(ROOT, 'tools', 'build-demo.mjs')], { cwd: ROOT, stdio: 'pipe' });
  assert.ok(fs.existsSync(path.join(ROOT, 'dist', 'index.html')), 'dist/index.html was not written');
});

test('the shipped CSP permits everything the page needs', async () => {
  const toml = await fsp.readFile(path.join(ROOT, 'netlify.toml'), 'utf8');
  const csp = /Content-Security-Policy = "([^"]+)"/.exec(toml)[1];
  // Each of these backs a capability the page actually uses.
  assert.match(csp, /worker-src [^;]*blob:/, 'the depth sorter runs from a blob URL');
  assert.match(csp, /media-src [^;]*blob:/, 'video files are read as blob URLs');
  assert.match(csp, /img-src [^;]*data:/, 'library thumbnails are data URLs');
  assert.match(csp, /script-src [^;]*'unsafe-inline'/, 'the module script is inlined');
  assert.match(csp, /font-src [^;]*fonts\.gstatic\.com/, 'webfonts load from gstatic');
  assert.match(csp, /style-src [^;]*fonts\.googleapis\.com/, 'the font stylesheet loads from googleapis');
});
