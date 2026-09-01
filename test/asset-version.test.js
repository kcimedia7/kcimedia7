import test from 'node:test';
import assert from 'node:assert/strict';
import { api, assetVersion } from '../web/js/api.js';

/**
 * Splat files are served with a one-year `immutable` cache, which is right for
 * a file that is megabytes and normally never changes. But a re-run replaces
 * the file behind the same asset id, and `immutable` means the browser will
 * not revalidate even on reload -- so without a version in the URL the viewer
 * keeps showing the previous model while the details panel shows the new one.
 */

test('re-running a conversion changes the URL its model is fetched from', () => {
  const before = {
    id: 'abc', finishedAt: '2026-08-30T23:28:44.388Z', result: { splatCount: 14080 },
  };
  const after = {
    id: 'abc', finishedAt: '2026-08-31T02:11:07.001Z', result: { splatCount: 7896 },
  };
  assert.notEqual(api.splatUrl(before.id, assetVersion(before)),
    api.splatUrl(after.id, assetVersion(after)),
    'the same URL would be served from cache and show the old model');
  assert.notEqual(api.thumbnailUrl(before.id, assetVersion(before)),
    api.thumbnailUrl(after.id, assetVersion(after)));
});

test('the same conversion keeps one URL so the cache still does its job', () => {
  // Busting the cache on every render would re-download megabytes each time
  // the page is opened, which is the problem the long cache exists to avoid.
  const asset = { id: 'abc', finishedAt: '2026-08-30T23:28:44.388Z', result: { splatCount: 14080 } };
  assert.equal(api.splatUrl(asset.id, assetVersion(asset)),
    api.splatUrl(asset.id, assetVersion({ ...asset })));
});

test('a count that changed alone still moves the version', () => {
  // Two runs could conceivably record the same finish time; the model they
  // produced is what actually has to differ.
  const a = { finishedAt: '2026-08-30T23:28:44.388Z', result: { splatCount: 14080 } };
  const b = { finishedAt: '2026-08-30T23:28:44.388Z', result: { splatCount: 7896 } };
  assert.notEqual(assetVersion(a), assetVersion(b));
});

test('an asset with no finished conversion yields a plain URL', () => {
  // A queued asset has nothing to version, and the URL must stay valid.
  assert.equal(assetVersion(null), '');
  assert.equal(assetVersion({ id: 'abc' }), '');
  assert.equal(api.splatUrl('abc', ''), '/api/assets/abc/splat');
  assert.ok(api.splatUrl('a b/c', assetVersion({ finishedAt: 'x' })).startsWith('/api/assets/a%20b%2Fc/splat?v='),
    'the id must still be encoded');
});

test('a version falls back to updatedAt while a run is still in flight', () => {
  const running = { id: 'abc', updatedAt: '2026-08-31T02:00:00.000Z' };
  assert.ok(assetVersion(running).includes('2026-08-31T02:00:00.000Z'));
});
