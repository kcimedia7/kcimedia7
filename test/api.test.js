import test from 'node:test';
import assert from 'node:assert/strict';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

/**
 * End-to-end test against a real HTTP server: upload frames, wait for the
 * conversion, then exercise the library operations the UI depends on.
 *
 * The data directory is set before the server modules are imported, because
 * config.js reads it at module load.
 */

const dataDir = await fsp.mkdtemp(path.join(os.tmpdir(), 'splatworks-test-'));
process.env.SPLAT_DATA_DIR = dataDir;
process.env.SPLAT_BACKEND = 'preview';

const { createServer } = await import('../server/index.js');
const store = await import('../server/store.js');
const { encodePng } = await import('../server/pipeline/png.js');
const { decodePly } = await import('../server/pipeline/ply.js');

await store.init();
const server = createServer();
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const base = `http://127.0.0.1:${server.address().port}`;

const jobs = await import('../server/jobs.js');

test.after(async () => {
  // Closing the listener does not stop a conversion already in flight, and a
  // job still writing into the data directory turns the cleanup below into an
  // ENOTEMPTY race that only shows up on some runs.
  const drained = await jobs.drain();
  await new Promise((resolve) => server.close(resolve));
  await fsp.rm(dataDir, { recursive: true, force: true });
  assert.ok(drained, 'a conversion was still running when the suite finished');
});

function testFrame(seed) {
  const width = 80;
  const height = 60;
  const data = Buffer.alloc(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      data[i] = 38; data[i + 1] = 40; data[i + 2] = 46; data[i + 3] = 255;
      if (Math.hypot(x - width / 2, y - height / 2) < 18) {
        data[i] = 200 + seed; data[i + 1] = 110; data[i + 2] = 60 + seed * 3;
      }
    }
  }
  return encodePng({ width, height, data });
}

async function uploadCapture({ name = 'Test capture', frames = 6, settings = {}, kind = 'photos' } = {}) {
  const form = new FormData();
  form.append('name', name);
  form.append('kind', kind);
  form.append('settings', JSON.stringify({ detail: 48, ...settings }));
  for (let i = 0; i < frames; i++) {
    form.append('frame', new Blob([testFrame(i * 6)], { type: 'image/png' }), `frame_${i}.png`);
  }
  const res = await fetch(`${base}/api/assets`, { method: 'POST', body: form });
  // Read the body once: the assertion message and the parse share it.
  const text = await res.text();
  assert.equal(res.status, 201, text);
  return JSON.parse(text).asset;
}

async function waitForStatus(id, wanted, timeoutMs = 45_000) {
  const deadline = Date.now() + timeoutMs;
  let last = null;
  while (Date.now() < deadline) {
    const res = await fetch(`${base}/api/assets/${id}`);
    const { asset } = await res.json();
    last = asset;
    if (wanted.includes(asset.status)) return asset;
    await new Promise((r) => setTimeout(r, 150));
  }
  // Say what it was doing instead. Test files run in parallel, so a real
  // conversion can be waiting behind another one for a worker rather than
  // being stuck -- and "never reached ready" alone cannot tell those apart.
  const queue = await fetch(`${base}/api/health`).then((r) => r.json())
    .then((h) => JSON.stringify(h.queue)).catch(() => 'unavailable');
  throw new Error(`asset ${id} never reached ${wanted.join('/')} in ${timeoutMs}ms. `
    + `last status: ${last?.status} (${last?.stage}) "${last?.message}" `
    + `error: ${last?.error ?? 'none'}. queue: ${queue}`);
}

test('health answers immediately, before backend detection finishes', async () => {
  // The server binds its socket before probing backends, because detection
  // shells out to `import torch` and can take tens of seconds. Health must
  // therefore answer straight away, flagging that capabilities are pending.
  const res = await fetch(`${base}/api/health`);
  assert.equal(res.status, 200);
  const body = await res.json();
  assert.equal(body.ok, true);
  assert.equal(typeof body.ready, 'boolean');
  assert.ok(Array.isArray(body.plans.colmap));
  assert.ok(Array.isArray(body.plans.gaussian));
  if (!body.ready) assert.equal(body.capabilities, null);
});

test('health reports the backend once detection lands', async () => {
  let body = null;
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    body = await (await fetch(`${base}/api/health`)).json();
    if (body.ready) break;
    await new Promise((r) => setTimeout(r, 200));
  }
  assert.equal(body.ready, true, 'capability detection never completed');
  assert.equal(body.capabilities.backend, 'preview');
  assert.equal(body.capabilities.reconstructs, false);
});

test('the library starts empty', async () => {
  const { assets } = await (await fetch(`${base}/api/assets`)).json();
  assert.deepEqual(assets, []);
});

test('a full conversion lifecycle produces downloadable splat files', async () => {
  const created = await uploadCapture({ name: 'Orbit one' });
  assert.equal(created.name, 'Orbit one');
  assert.equal(created.source.frameCount, 6);

  const ready = await waitForStatus(created.id, ['ready', 'failed']);
  assert.equal(ready.status, 'ready', ready.error || '');
  assert.ok(ready.result.splatCount > 0);
  assert.equal(ready.backend, 'preview');
  assert.ok(ready.log.length > 0, 'the conversion should be logged');

  // The compact form the viewer streams.
  const splat = await fetch(`${base}/api/assets/${created.id}/splat`);
  assert.equal(splat.status, 200);
  const splatBuf = Buffer.from(await splat.arrayBuffer());
  assert.equal(splatBuf.length, ready.result.splatCount * 32);

  // The interchange form other tools read.
  const ply = await fetch(`${base}/api/assets/${created.id}/export.ply`);
  assert.equal(ply.status, 200);
  assert.match(ply.headers.get('content-disposition'), /filename="Orbit one\.ply"/);
  const cloud = decodePly(Buffer.from(await ply.arrayBuffer()));
  assert.equal(cloud.count, ready.result.splatCount);

  const thumb = await fetch(`${base}/api/assets/${created.id}/thumbnail`);
  assert.equal(thumb.status, 200);
  assert.equal(thumb.headers.get('content-type'), 'image/png');
});

test('an upload with no frames is refused', async () => {
  const form = new FormData();
  form.append('name', 'Nothing');
  const res = await fetch(`${base}/api/assets`, { method: 'POST', body: form });
  assert.equal(res.status, 400);
  assert.match((await res.json()).error, /no frames/);
});

test('a frame that is not a PNG is refused', async () => {
  const form = new FormData();
  form.append('kind', 'photos');
  form.append('frame', new Blob([Buffer.from('this is not a png')]), 'bad.png');
  const res = await fetch(`${base}/api/assets`, { method: 'POST', body: form });
  assert.equal(res.status, 400);
  assert.match((await res.json()).error, /must be PNG/);
});

test('a rejected upload leaves nothing behind on disk', async () => {
  const before = await fsp.readdir(path.join(dataDir, 'assets'));
  const form = new FormData();
  form.append('frame', new Blob([Buffer.from('nope')]), 'bad.png');
  await fetch(`${base}/api/assets`, { method: 'POST', body: form });
  assert.deepEqual(await fsp.readdir(path.join(dataDir, 'assets')), before);
});

test('metadata edits round trip through the library', async () => {
  const asset = await uploadCapture({ name: 'Taggable' });
  await waitForStatus(asset.id, ['ready', 'failed']);

  const res = await fetch(`${base}/api/assets/${asset.id}`, {
    method: 'PATCH',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      name: 'Renamed capture',
      notes: 'Shot handheld at dusk.',
      tags: ['Kitchen', 'kitchen', ' Test '],
    }),
  });
  assert.equal(res.status, 200);
  const updated = (await res.json()).asset;
  assert.equal(updated.name, 'Renamed capture');
  assert.equal(updated.notes, 'Shot handheld at dusk.');
  // Tags are lowercased and de-duplicated.
  assert.deepEqual(updated.tags, ['kitchen', 'test']);
});

test('edits change the export but never the stored reconstruction', async () => {
  const asset = await uploadCapture({ name: 'Editable' });
  const ready = await waitForStatus(asset.id, ['ready', 'failed']);
  assert.equal(ready.status, 'ready');

  const rawFirst = decodePly(Buffer.from(
    await (await fetch(`${base}/api/assets/${asset.id}/export.ply?raw=1`)).arrayBuffer()));

  await fetch(`${base}/api/assets/${asset.id}`, {
    method: 'PATCH',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ edits: { translate: [5, 0, 0], pruneBelowOpacity: 0.9 } }),
  });

  const edited = decodePly(Buffer.from(
    await (await fetch(`${base}/api/assets/${asset.id}/export.ply`)).arrayBuffer()));
  assert.ok(edited.count < rawFirst.count, 'pruning should remove gaussians');
  for (let i = 0; i < edited.count; i++) {
    assert.ok(edited.positions[i * 3] > 3, 'translation should be baked into the export');
  }

  // The untouched original is still there.
  const rawAgain = decodePly(Buffer.from(
    await (await fetch(`${base}/api/assets/${asset.id}/export.ply?raw=1`)).arrayBuffer()));
  assert.equal(rawAgain.count, rawFirst.count);

  // And resetting brings the full export back.
  await fetch(`${base}/api/assets/${asset.id}`, {
    method: 'PATCH',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ resetEdits: true }),
  });
  const reset = decodePly(Buffer.from(
    await (await fetch(`${base}/api/assets/${asset.id}/export.ply`)).arrayBuffer()));
  assert.equal(reset.count, rawFirst.count);
});

test('duplicating copies the result so the copy can diverge', async () => {
  const asset = await uploadCapture({ name: 'Original' });
  await waitForStatus(asset.id, ['ready', 'failed']);

  const res = await fetch(`${base}/api/assets/${asset.id}/duplicate`, { method: 'POST' });
  assert.equal(res.status, 201);
  const copy = (await res.json()).asset;
  assert.notEqual(copy.id, asset.id);
  assert.equal(copy.name, 'Original (copy)');
  assert.equal(copy.status, 'ready');

  // The copy serves its own splat, not a reference to the original's.
  const copySplat = await fetch(`${base}/api/assets/${copy.id}/splat`);
  assert.equal(copySplat.status, 200);

  // Editing the copy leaves the original alone.
  await fetch(`${base}/api/assets/${copy.id}`, {
    method: 'PATCH',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: 'Diverged' }),
  });
  const { asset: original } = await (await fetch(`${base}/api/assets/${asset.id}`)).json();
  assert.equal(original.name, 'Original');
});

test('re-running a conversion clears the old result and produces a new one', async () => {
  const asset = await uploadCapture({ name: 'Rerun' });
  await waitForStatus(asset.id, ['ready', 'failed']);

  const res = await fetch(`${base}/api/assets/${asset.id}/reconvert`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ settings: { detail: 64 } }),
  });
  assert.equal(res.status, 202);

  const again = await waitForStatus(asset.id, ['ready', 'failed']);
  assert.equal(again.status, 'ready');
  assert.equal(again.settings.detail, 64);
  assert.ok(again.result.splatCount > 0);
});

test('deleting removes the entry and its files', async () => {
  const asset = await uploadCapture({ name: 'Doomed' });
  await waitForStatus(asset.id, ['ready', 'failed']);
  const dir = path.join(dataDir, 'assets', asset.id);
  assert.ok(await fsp.stat(dir).catch(() => null), 'files exist before deletion');

  const res = await fetch(`${base}/api/assets/${asset.id}`, { method: 'DELETE' });
  assert.equal(res.status, 200);

  assert.equal((await fetch(`${base}/api/assets/${asset.id}`)).status, 404);
  assert.equal(await fsp.stat(dir).catch(() => null), null, 'files are gone too');
});

test('unknown ids and endpoints return 404 rather than crashing', async () => {
  assert.equal((await fetch(`${base}/api/assets/does-not-exist`)).status, 404);
  assert.equal((await fetch(`${base}/api/assets/does-not-exist/splat`)).status, 404);
  assert.equal((await fetch(`${base}/api/nonsense`)).status, 404);
});

test('an id cannot traverse out of the asset directory', async () => {
  const res = await fetch(`${base}/api/assets/${encodeURIComponent('../../library.json')}`);
  assert.equal(res.status, 404);
});

test('unknown client routes fall through to the app shell', async () => {
  const res = await fetch(`${base}/a/some-id`);
  assert.equal(res.status, 200);
  assert.match(res.headers.get('content-type'), /text\/html/);
});

test('the splat endpoint honours byte ranges', async () => {
  const asset = await uploadCapture({ name: 'Ranged' });
  const ready = await waitForStatus(asset.id, ['ready', 'failed']);
  assert.equal(ready.status, 'ready');

  const res = await fetch(`${base}/api/assets/${asset.id}/splat`, { headers: { range: 'bytes=0-31' } });
  assert.equal(res.status, 206);
  assert.equal(res.headers.get('content-range'), `bytes 0-31/${ready.result.splatBytes}`);
  assert.equal((await res.arrayBuffer()).byteLength, 32);
});

test('a 360 capture is stored and reported as its own kind', async () => {
  // Panorama captures are resampled into perspective views before upload, so
  // the server sees ordinary frames. Only the kind distinguishes them, and if
  // it were silently folded into 'photos' the library would mislabel every
  // 360 conversion.
  const asset = await uploadCapture({ name: '360 room', kind: 'pano', frames: 6 });
  assert.equal(asset.kind, 'pano');

  const fetched = await waitForStatus(asset.id, ['ready', 'failed']);
  assert.equal(fetched.kind, 'pano', 'the kind must survive a round trip through the store');
});

test('an unknown capture kind falls back to photos rather than being stored', async () => {
  // The kind reaches the UI as a label, so an arbitrary string from a client
  // must not end up rendered there.
  const asset = await uploadCapture({ name: 'odd', kind: 'not-a-kind', frames: 4 });
  assert.equal(asset.kind, 'photos');
  await waitForStatus(asset.id, ['ready', 'failed']);
});
