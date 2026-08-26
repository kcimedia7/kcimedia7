import test from 'node:test';
import assert from 'node:assert/strict';
import { Readable } from 'node:stream';
import { parseMultipart, boundaryOf } from '../server/http/multipart.js';

/** Build a multipart body and feed it in chunks of `chunkSize` bytes. */
function makeRequest(boundary, parts, chunkSize = Infinity) {
  const pieces = [];
  for (const part of parts) {
    let headers = `--${boundary}\r\nContent-Disposition: form-data; name="${part.name}"`;
    if (part.filename !== undefined) headers += `; filename="${part.filename}"`;
    headers += `\r\nContent-Type: ${part.type || 'text/plain'}\r\n\r\n`;
    pieces.push(Buffer.from(headers, 'latin1'));
    pieces.push(Buffer.isBuffer(part.body) ? part.body : Buffer.from(part.body));
    pieces.push(Buffer.from('\r\n', 'latin1'));
  }
  pieces.push(Buffer.from(`--${boundary}--\r\n`, 'latin1'));
  const body = Buffer.concat(pieces);

  const chunks = [];
  for (let i = 0; i < body.length; i += Math.min(chunkSize, body.length)) {
    chunks.push(body.subarray(i, i + Math.min(chunkSize, body.length)));
  }
  const req = Readable.from(chunks);
  req.headers = { 'content-type': `multipart/form-data; boundary=${boundary}` };
  return req;
}

async function collect(req) {
  const fields = {};
  const files = [];
  await parseMultipart(req, {
    onField(name, value) { fields[name] = value; },
    async onFile(part) {
      const buf = [];
      files.push({ name: part.name, filename: part.filename, type: part.contentType, chunks: buf });
      return {
        async write(chunk) { buf.push(Buffer.from(chunk)); },
        async end() {},
      };
    },
  });
  return {
    fields,
    files: files.map((f) => ({ ...f, body: Buffer.concat(f.chunks) })),
  };
}

test('boundaryOf reads plain and quoted boundaries', () => {
  assert.equal(boundaryOf('multipart/form-data; boundary=abc123'), 'abc123');
  assert.equal(boundaryOf('multipart/form-data; boundary="a b c"'), 'a b c');
  assert.equal(boundaryOf('application/json'), null);
});

test('fields and files are parsed out of one body', async () => {
  const req = makeRequest('XbryZ', [
    { name: 'name', body: 'Kitchen orbit' },
    { name: 'kind', body: 'video' },
    { name: 'frame', filename: 'frame_1.png', type: 'image/png', body: Buffer.from([1, 2, 3, 4, 5]) },
    { name: 'frame', filename: 'frame_2.png', type: 'image/png', body: Buffer.from([9, 8, 7]) },
  ]);
  const { fields, files } = await collect(req);

  assert.equal(fields.name, 'Kitchen orbit');
  assert.equal(fields.kind, 'video');
  assert.equal(files.length, 2);
  assert.equal(files[0].filename, 'frame_1.png');
  assert.equal(files[0].type, 'image/png');
  assert.deepEqual([...files[0].body], [1, 2, 3, 4, 5]);
  assert.deepEqual([...files[1].body], [9, 8, 7]);
});

test('a boundary split across chunk edges is still found', async () => {
  // Byte-at-a-time is the worst case for a streaming boundary scanner.
  const payload = Buffer.from('the quick brown fox jumps over the lazy dog');
  for (const chunkSize of [1, 2, 3, 7, 13]) {
    const req = makeRequest('SplitMe', [
      { name: 'a', body: 'first' },
      { name: 'blob', filename: 'x.bin', body: payload },
    ], chunkSize);
    const { fields, files } = await collect(req);
    assert.equal(fields.a, 'first', `chunk size ${chunkSize}`);
    assert.deepEqual(files[0].body, payload, `chunk size ${chunkSize}`);
  }
});

test('file content that contains boundary-like bytes is preserved', async () => {
  // A body containing the delimiter text must not end the part early.
  const tricky = Buffer.from('data--Bnd not a boundary\r\n--Bnd-still-not\r\nend');
  const req = makeRequest('Bnd', [{ name: 'f', filename: 'f.bin', body: tricky }], 5);
  const { files } = await collect(req);
  assert.deepEqual(files[0].body, tricky);
});

test('an empty file part yields an empty body rather than being skipped', async () => {
  const req = makeRequest('E', [{ name: 'f', filename: 'empty.bin', body: Buffer.alloc(0) }]);
  const { files } = await collect(req);
  assert.equal(files.length, 1);
  assert.equal(files[0].body.length, 0);
});

test('a body larger than maxBytes is rejected', async () => {
  const req = makeRequest('Big', [{ name: 'f', filename: 'f.bin', body: Buffer.alloc(4096) }]);
  await assert.rejects(
    parseMultipart(req, { maxBytes: 512, onFile: async () => ({ write: async () => {}, end: async () => {} }) }),
    /size limit/,
  );
});

test('a truncated body is reported rather than silently accepted', async () => {
  const req = Readable.from([Buffer.from('--T\r\nContent-Disposition: form-data; name="f"; filename="a"\r\n\r\npartial')]);
  req.headers = { 'content-type': 'multipart/form-data; boundary=T' };
  await assert.rejects(collect(req), /ended before the final boundary/);
});

test('a request that is not multipart is rejected', async () => {
  const req = Readable.from([Buffer.from('{}')]);
  req.headers = { 'content-type': 'application/json' };
  await assert.rejects(collect(req), /not a multipart request/);
});
