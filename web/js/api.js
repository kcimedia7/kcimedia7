/** Thin wrapper over the REST API, with upload progress and a live event feed. */

async function request(path, options = {}) {
  const res = await fetch(path, {
    headers: options.body && !(options.body instanceof FormData)
      ? { 'content-type': 'application/json' }
      : undefined,
    ...options,
  });
  const text = await res.text();
  let body = null;
  if (text) {
    try { body = JSON.parse(text); } catch { body = { error: text }; }
  }
  if (!res.ok) throw new Error(body?.error || `${res.status} ${res.statusText}`);
  return body;
}

export const api = {
  health: () => request('/api/health'),
  list: () => request('/api/assets'),
  get: (id) => request(`/api/assets/${encodeURIComponent(id)}`),
  patch: (id, patch) => request(`/api/assets/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    body: JSON.stringify(patch),
  }),
  remove: (id) => request(`/api/assets/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  cancel: (id) => request(`/api/assets/${encodeURIComponent(id)}/cancel`, { method: 'POST' }),
  duplicate: (id) => request(`/api/assets/${encodeURIComponent(id)}/duplicate`, { method: 'POST' }),
  reconvert: (id, settings) => request(`/api/assets/${encodeURIComponent(id)}/reconvert`, {
    method: 'POST',
    body: JSON.stringify({ settings }),
  }),
  /**
   * Asset files are cached hard -- a splat is megabytes and never changes for a
   * given conversion. But a re-run replaces the file behind the same id, and
   * `immutable` tells the browser not to revalidate even on reload, so without
   * a version in the URL the viewer would keep showing the previous model
   * indefinitely while the details panel showed the new one.
   */
  splatUrl: (id, version) =>
    `/api/assets/${encodeURIComponent(id)}/splat${versionQuery(version)}`,
  thumbnailUrl: (id, version) =>
    `/api/assets/${encodeURIComponent(id)}/thumbnail${versionQuery(version)}`,
  exportUrl: (id, format, raw) =>
    `/api/assets/${encodeURIComponent(id)}/export.${format}${raw ? '?raw=1' : ''}`,

  /**
   * Uploads use XHR rather than fetch because only XHR reports upload progress,
   * and a multi-gigabyte video needs a progress bar.
   */
  upload(formData, { onProgress, signal } = {}) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/assets');
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) onProgress?.(e.loaded / e.total);
      });
      xhr.addEventListener('load', () => {
        let body = null;
        try { body = JSON.parse(xhr.responseText); } catch { /* handled below */ }
        if (xhr.status >= 200 && xhr.status < 300 && body) resolve(body);
        else reject(new Error(body?.error || `Upload failed (${xhr.status})`));
      });
      xhr.addEventListener('error', () => reject(new Error('The upload connection failed.')));
      xhr.addEventListener('abort', () => reject(new DOMException('Upload cancelled', 'AbortError')));
      signal?.addEventListener('abort', () => xhr.abort(), { once: true });
      xhr.send(formData);
    });
  },
};

/**
 * Server-sent events, with automatic reconnection. Returns an unsubscribe fn.
 */
export function subscribe(onUpdate) {
  let source = null;
  let closed = false;
  let retry = null;

  const connect = () => {
    if (closed) return;
    source = new EventSource('/api/events');
    source.addEventListener('update', (e) => {
      try { onUpdate(JSON.parse(e.data)); } catch { /* ignore a malformed frame */ }
    });
    source.onerror = () => {
      source.close();
      if (closed) return;
      // EventSource retries on its own for network blips, but not after the
      // server restarts, so reconnect explicitly.
      retry = setTimeout(connect, 2000);
    };
  };
  connect();

  return () => {
    closed = true;
    clearTimeout(retry);
    source?.close();
  };
}

/** Cache-busting token for an asset's files: whenever its conversion finished. */
function versionQuery(version) {
  if (!version) return '';
  return `?v=${encodeURIComponent(String(version))}`;
}

/**
 * A token that changes whenever a conversion produced a new model.
 *
 * `finishedAt` moves on every run; the splat count is a second signal in case
 * two runs somehow land on the same timestamp.
 */
export function assetVersion(asset) {
  if (!asset) return '';
  return [asset.finishedAt || asset.updatedAt || '', asset.result?.splatCount ?? '']
    .filter(Boolean).join('-');
}
