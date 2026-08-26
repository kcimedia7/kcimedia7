/**
 * Streaming multipart/form-data parser.
 *
 * Uploads here are whole videos, so nothing is buffered whole: file parts are
 * handed to the caller as a stream of chunks and written straight to disk.
 * Only small text fields are collected in memory.
 */

const DASH = 0x2d; // '-'
const CR = 0x0d;
const LF = 0x0a;

export function boundaryOf(contentType) {
  if (!contentType) return null;
  const m = /boundary=(?:"([^"]+)"|([^;]+))/i.exec(contentType);
  if (!m) return null;
  return (m[1] || m[2]).trim();
}

function parseHeaders(block) {
  const headers = {};
  for (const line of block.toString('utf8').split('\r\n')) {
    if (!line) continue;
    const i = line.indexOf(':');
    if (i === -1) continue;
    headers[line.slice(0, i).trim().toLowerCase()] = line.slice(i + 1).trim();
  }
  const disposition = headers['content-disposition'] || '';
  const name = /name="([^"]*)"/.exec(disposition);
  const filename = /filename\*?=(?:UTF-8'')?"?([^";]*)"?/.exec(disposition);
  return {
    headers,
    name: name ? decodeURIComponent(name[1]) : null,
    filename: filename && filename[1] ? decodeURIComponent(filename[1]) : null,
    contentType: headers['content-type'] || 'application/octet-stream',
  };
}

/**
 * @param {import('node:http').IncomingMessage} req
 * @param {object} handlers
 * @param {(part: {name: string, filename: string, contentType: string}) => Promise<{write: Function, end: Function}>} handlers.onFile
 * @param {(name: string, value: string) => void} handlers.onField
 * @param {number} [handlers.maxBytes]
 */
export async function parseMultipart(req, handlers) {
  const boundary = boundaryOf(req.headers['content-type']);
  if (!boundary) throw Object.assign(new Error('not a multipart request'), { status: 400 });

  const delim = Buffer.from(`\r\n--${boundary}`);
  const maxBytes = handlers.maxBytes ?? Infinity;
  const maxFieldBytes = handlers.maxFieldBytes ?? 1024 * 1024;

  // Prefixing with CRLF lets the very first boundary be found by the same scan
  // as every subsequent one.
  let buf = Buffer.from('\r\n');
  let state = 'preamble';
  let total = 0;
  let part = null;
  let sink = null;
  let fieldChunks = null;
  let fieldBytes = 0;

  const finishPart = async () => {
    if (sink) {
      await sink.end();
      sink = null;
    } else if (part && fieldChunks) {
      handlers.onField?.(part.name, Buffer.concat(fieldChunks).toString('utf8'));
    }
    part = null;
    fieldChunks = null;
    fieldBytes = 0;
  };

  const pushBody = async (chunk) => {
    if (!chunk.length) return;
    if (sink) {
      await sink.write(chunk);
    } else if (fieldChunks) {
      fieldBytes += chunk.length;
      if (fieldBytes > maxFieldBytes) {
        throw Object.assign(new Error(`field "${part?.name}" is too large`), { status: 413 });
      }
      fieldChunks.push(chunk);
    }
  };

  const startPart = async () => {
    const end = buf.indexOf('\r\n\r\n');
    if (end === -1) return false;
    part = parseHeaders(buf.subarray(0, end));
    buf = buf.subarray(end + 4);
    if (part.filename !== null) {
      sink = await handlers.onFile(part);
      fieldChunks = null;
    } else {
      sink = null;
      fieldChunks = [];
    }
    state = 'body';
    return true;
  };

  for await (const chunk of req) {
    total += chunk.length;
    if (total > maxBytes) {
      throw Object.assign(new Error('upload exceeds the size limit'), { status: 413 });
    }
    buf = buf.length ? Buffer.concat([buf, chunk]) : chunk;

    let progressed = true;
    while (progressed && state !== 'done') {
      progressed = false;

      if (state === 'headers') {
        if (!(await startPart())) break;
        progressed = true;
        continue;
      }

      const at = buf.indexOf(delim);
      if (at === -1) {
        if (state === 'body') {
          // Hold back enough bytes that a boundary split across chunks is still found.
          const keep = Math.min(buf.length, delim.length - 1);
          const emit = buf.subarray(0, buf.length - keep);
          buf = buf.subarray(buf.length - keep);
          await pushBody(emit);
        } else if (buf.length > delim.length) {
          buf = buf.subarray(buf.length - delim.length); // discard preamble noise
        }
        break;
      }

      const verdict = classifyDelimiter(buf, at + delim.length);

      if (verdict.kind === 'need') {
        // Not enough bytes yet to tell a real boundary from body content that
        // merely looks like one; emit what is definitely body and wait.
        if (state === 'body') {
          await pushBody(buf.subarray(0, at));
          buf = buf.subarray(at);
        }
        break;
      }

      if (verdict.kind === 'false') {
        // Body content that happens to contain the delimiter text. RFC 2046
        // only treats it as a boundary when "--" or CRLF follows.
        if (state === 'body') await pushBody(buf.subarray(0, at + delim.length));
        buf = buf.subarray(at + delim.length);
        progressed = true;
        continue;
      }

      if (state === 'body') {
        await pushBody(buf.subarray(0, at));
        await finishPart();
      }
      buf = buf.subarray(at + delim.length + verdict.skip);
      state = verdict.kind === 'close' ? 'done' : 'headers';
      progressed = true;
    }

    if (state === 'done') break;
  }

  if (state !== 'done' && (sink || fieldChunks)) {
    // The stream ended mid-part; close what we opened so no handle leaks.
    await finishPart();
    throw Object.assign(new Error('upload ended before the final boundary'), { status: 400 });
  }
}

/**
 * Decide whether the delimiter just matched at `pos - delim.length` really ends
 * a part. RFC 2046 requires "--" (final boundary) or optional linear whitespace
 * then CRLF (next part); anything else means the bytes are part content.
 */
function classifyDelimiter(buf, pos) {
  if (pos + 2 > buf.length) return { kind: 'need' };
  if (buf[pos] === DASH && buf[pos + 1] === DASH) return { kind: 'close', skip: 2 };

  let i = pos;
  while (i < buf.length && (buf[i] === 0x20 || buf[i] === 0x09)) i++;
  if (i + 2 > buf.length) return { kind: 'need' };
  if (buf[i] === CR && buf[i + 1] === LF) return { kind: 'next', skip: i - pos + 2 };
  return { kind: 'false' };
}
