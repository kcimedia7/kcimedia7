import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));

export const ROOT = path.resolve(here, '..');
export const WEB_DIR = path.join(ROOT, 'web');
export const DATA_DIR = process.env.SPLAT_DATA_DIR
  ? path.resolve(process.env.SPLAT_DATA_DIR)
  : path.join(ROOT, 'data');

/** Per-asset working area: data/assets/<id>/{source,frames,work,output} */
export const ASSETS_DIR = path.join(DATA_DIR, 'assets');
export const DB_FILE = path.join(DATA_DIR, 'library.json');

export const PORT = Number(process.env.PORT || 8787);
export const HOST = process.env.HOST || '127.0.0.1';

/** How many conversions may train at once. Real trainers want a GPU each. */
export const MAX_CONCURRENT_JOBS = Number(process.env.SPLAT_CONCURRENCY || 1);

/** Upload ceiling per request body (bytes). Videos are the big ones. */
export const MAX_UPLOAD_BYTES = Number(process.env.SPLAT_MAX_UPLOAD || 2 * 1024 * 1024 * 1024);

/**
 * Preview-backend tuning. These bound the dependency-free reconstruction so a
 * conversion stays interactive: at most N frames, each sampled to a grid of
 * roughly GRID x GRID gaussians.
 */
export const PREVIEW_MAX_FRAMES = Number(process.env.SPLAT_PREVIEW_FRAMES || 24);
export const PREVIEW_GRID = Number(process.env.SPLAT_PREVIEW_GRID || 160);

/** Force a specific pipeline backend: 'auto' | 'colmap' | 'preview'. */
export const BACKEND = process.env.SPLAT_BACKEND || 'auto';
