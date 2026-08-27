#!/usr/bin/env node
/**
 * Build the hosted browser demo: one self-contained HTML file.
 *
 * The demo runs the same reconstruction and the same renderer as the app, so
 * this script INLINES the real modules rather than restating them — anything
 * that changes in server/pipeline or web/js/viewer shows up in the demo the
 * next time this runs. What cannot cross into a single-file browser page is
 * patched here, explicitly, so the differences stay visible:
 *
 *   - `import`/`export` are stripped, since everything shares one scope
 *   - Node's Buffer becomes a typed array
 *   - a handful of module-private helpers are renamed where two modules
 *     happen to use the same name (they no longer have separate scopes)
 *   - the sort worker is embedded as a blob URL instead of a module URL
 *
 * Usage:
 *   node tools/build-demo.mjs                 # both targets
 *   node tools/build-demo.mjs --out FILE      # one file, default (web) target
 *
 * Targets differ only in where the page can put a finished file, which the page
 * detects at runtime -- the bundle itself is identical.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const outFlag = argv.indexOf('--out');
const explicitOut = outFlag !== -1 ? argv[outFlag + 1] : argv.find((a) => !a.startsWith('--'));

/**
 * `dist/` is what Netlify publishes; the second copy is the file published as
 * a Claude artifact. Same bytes, two homes.
 */
const TARGETS = explicitOut
  ? [path.resolve(explicitOut)]
  : [
    path.join(ROOT, 'dist', 'index.html'),
    path.join(ROOT, 'tools', 'demo', 'build', 'splatworks-demo.html'),
  ];

const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

/** Strip module syntax so several files can share one scope. */
function flatten(source) {
  return source
    .replace(/^import\s+[^;]*;\s*$/gm, '')
    .replace(/^export\s+(?=(async\s+)?(function|class|const|let)\b)/gm, '')
    .replace(/^export\s*\{[^}]*\}\s*;?\s*$/gm, '')
    .trim();
}

function patch(source, replacements, label) {
  let result = source;
  for (const [from, to] of replacements) {
    if (typeof from === 'string' && !result.includes(from)) {
      throw new Error(`${label}: patch target not found: ${from.slice(0, 70)}`);
    }
    result = result.replaceAll(from, to);
  }
  return result;
}

// --- gaussian cloud core -----------------------------------------------------
const splat = patch(flatten(read('server/pipeline/splat.js')), [
  // Node returns a Buffer; the renderer wants the raw ArrayBuffer.
  ['return Buffer.from(out);', 'return out;'],
], 'splat.js');

// --- image helpers -----------------------------------------------------------
const imageops = patch(flatten(read('server/pipeline/imageops.js')), [
  ['const out = Buffer.alloc(w * h * 4);', 'const out = new Uint8ClampedArray(w * h * 4);'],
], 'imageops.js');

// --- preview reconstruction --------------------------------------------------
const preview = patch(flatten(read('server/pipeline/preview.js')), [
  // These collide with the identically-named helpers exported by mat.js.
  ['function normalize(v) {', 'function pvNormalize(v) {'],
  ['function cross(a, b) {', 'function pvCross(a, b) {'],
  ['normalize([', 'pvNormalize(['],
  ['= cross(forward, worldUp)', '= pvCross(forward, worldUp)'],
  ['normalize(cross(right, forward))', 'pvNormalize(pvCross(right, forward))'],
  ['right = normalize(right);', 'right = pvNormalize(right);'],
], 'preview.js');

// --- PLY writer and the edit engine (export path) ----------------------------
const ply = patch(flatten(read('server/pipeline/ply.js')), [
  // edits.js declares its own clamp01, and both land in one scope here.
  ['function clamp01(v) {', 'function plyClamp01(v) {'],
  ['clamp01(shToColor(', 'plyClamp01(shToColor('],
  // Node Buffer -> plain bytes. The header is ASCII, so the encoding is a wash.
  ["const headerBuf = Buffer.from(header, 'latin1');", 'const headerBuf = new TextEncoder().encode(header);'],
  ['const body = Buffer.allocUnsafe(cloud.count * stride);', 'const body = new Uint8Array(cloud.count * stride);'],
  ['return Buffer.concat([headerBuf, body]);', 'return concatBytes(headerBuf, body);'],
], 'ply.js');

// edits.js imports defaultEdits from the server's store; flatten() drops that
// import and the app shell supplies its own (hoisted) definition.
const edits = flatten(read('server/edits.js'));

// --- viewer ------------------------------------------------------------------
const mat = flatten(read('web/js/viewer/mat.js'));
const camera = flatten(read('web/js/viewer/camera.js'));
const shaders = flatten(read('web/js/viewer/shaders.js'));

const renderer = patch(flatten(read('web/js/viewer/renderer.js')), [
  // splat.js already defines a boundsOf that takes a cloud, not raw positions.
  ['function boundsOf(positions, count) {', 'function boundsOfPositions(positions, count) {'],
  ['this.bounds = boundsOf(positions, count);', 'this.bounds = boundsOfPositions(positions, count);'],
  // No separate module URL exists in a single-file page.
  [
    "new Worker(new URL('./sortWorker.js', import.meta.url), { type: 'module' })",
    'new Worker(SORT_WORKER_URL)',
  ],
], 'renderer.js');

const sortWorker = read('web/js/viewer/sortWorker.js');

const bundle = [
  '// ---- server/pipeline/splat.js ----', splat,
  '// ---- server/pipeline/imageops.js ----', imageops,
  '// ---- server/pipeline/preview.js ----', preview,
  '// ---- server/pipeline/ply.js ----', ply,
  '// ---- server/edits.js ----', edits,
  'function concatBytes(a, b) {',
  '  const out = new Uint8Array(a.length + b.length);',
  '  out.set(a, 0); out.set(b, a.length);',
  '  return out;',
  '}',
  '// ---- web/js/viewer/mat.js ----', mat,
  '// ---- web/js/viewer/camera.js ----', camera,
  '// ---- web/js/viewer/shaders.js ----', shaders,
  '// ---- web/js/viewer/sortWorker.js (blob) ----',
  `const SORT_WORKER_URL = URL.createObjectURL(new Blob([${JSON.stringify(sortWorker)}], { type: 'text/javascript' }));`,
  '// ---- web/js/viewer/renderer.js ----', renderer,
  '// ---- tools/demo/app.js ----', flatten(read('tools/demo/app.js')),
].join('\n\n');

/**
 * An artifact's page content is injected into a document whose <head> we do not
 * control, so the page cannot declare its own charset. Emitting pure ASCII
 * makes rendering independent of whatever encoding the host assumes.
 */
const TYPOGRAPHY = [
  [/[\u2014\u2013]/g, '--'],
  [/\u00b7/g, '.'],
  [/[\u201c\u201d]/g, '"'],
  [/[\u2018\u2019]/g, "'"],
  [/\u2026/g, '...'],
  [/\u03c3/g, 'sigma'],
  [/\u00d7/g, 'x'],
  [/\u00b0/g, ' deg'],
];

function toAscii(source, label) {
  let result = source;
  for (const [pattern, replacement] of TYPOGRAPHY) result = result.replace(pattern, replacement);
  const stray = result.match(/[^\x00-\x7F]/g);
  if (stray) {
    throw new Error(`${label}: non-ASCII characters survived: ${[...new Set(stray)].join(' ')}`);
  }
  return result;
}

const html = read('tools/demo/shell.html')
  .replace('/*{{BUNDLE}}*/', () => toAscii(bundle, 'bundle'));

toAscii(html, 'page');   // the markup is entity-encoded; this proves it

const kb = (n) => `${(n / 1024).toFixed(1)} KB`;
for (const target of TARGETS) {
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, html);
  console.log(`wrote ${path.relative(ROOT, target)} (${kb(html.length)})`);
}
console.log(`bundle ${kb(bundle.length)}`);
