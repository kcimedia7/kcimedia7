# SplatWorks

Turn photos and video into 3D Gaussian Splats, then keep every conversion in a
library you can re-open, edit and export from later.

It runs as a single Node process with **no npm dependencies** and no build step:

```bash
node server/index.js     # http://127.0.0.1:8787
```

---

## What it does

1. **Upload** photos or a video. Frames are decoded and sampled *in the browser*,
   so any format your device can play works — HEIC, HEVC, whatever your phone
   produces — and the server never needs ffmpeg.
2. **Convert** them to a gaussian splat. The pipeline uses COLMAP and a real 3DGS
   trainer when they are installed, and falls back to a built-in preview
   reconstruction when they are not.
3. **Browse** everything you have converted, with live progress, search, tags and
   notes.
4. **View** any splat in a WebGL2 renderer written for this project — real EWA
   splatting with per-frame depth sorting in a worker.
5. **Edit** non-destructively: transform, colour-grade, crop, prune. Edits are
   parameters, never baked into stored data, so nothing is ever lost.
6. **Export** to `.ply` (the standard 3DGS interchange format) or `.splat`, with
   or without your edits applied.

---

## The two backends

Real gaussian splatting needs structure-from-motion to recover camera poses and
a CUDA trainer to optimise the gaussians. Both are large external installs, so
SplatWorks detects them rather than bundling them.

| | `colmap` backend | `preview` backend |
|---|---|---|
| Needs | COLMAP + a 3DGS trainer + GPU | nothing |
| Camera poses | solved from the images | **assumed** (turntable orbit) |
| Time | minutes to hours | about a second |
| Output | a true reconstruction | a fast proxy |

The badge in the top-right of the UI always tells you which one is active, and
the "New conversion" page says so plainly before you convert anything.

### What the preview backend actually does

It is **not** structure-from-motion, and it is labelled that way throughout the
app. Since it cannot recover camera poses, it assumes the capture pattern the UI
asks you to shoot — an orbit around a subject — and for each frame:

- estimates a depth field from three cheap monocular cues: how far each pixel's
  colour sits from the backdrop colour, local gradient energy (in-focus reads as
  nearer), and a mild centre bias;
- back-projects the pixels through that frame's assumed camera into oriented
  gaussian discs facing it;
- merges every frame's relief into one cloud and normalises it.

The result is a real, editable, exportable gaussian cloud built from your own
pixels that looks like your subject from the angles you shot it. It is a proxy
for review and framing, not a metrically accurate reconstruction. Use it to see
the shape of a capture in a second; re-run the same capture through the COLMAP
backend when you want the real thing — the original files are kept for exactly
that.

### Enabling real reconstruction

Install [COLMAP](https://colmap.github.io/) so it is on `PATH`, install a 3DGS
trainer, and point SplatWorks at it:

```bash
export SPLAT_TRAINER_CMD="python /opt/gaussian-splatting/train.py \
  -s {source} -m {output} --iterations {iterations}"

node server/index.js
```

The template is expanded with these placeholders:

| Placeholder | Value |
|---|---|
| `{source}` | COLMAP dataset directory (`images/` + `sparse/0/`) |
| `{images}` | the `images/` directory inside it |
| `{output}` | where the trainer should write its model |
| `{iterations}` | iteration count from the conversion settings |

Anything that reads a COLMAP dataset and writes a `.ply` works — INRIA's
`train.py`, gsplat, nerfstudio. A typo like `{sources}` is rejected at launch
rather than passed through to the trainer.

---

## Shooting a capture that converts well

This matters more than any setting:

- Walk a **full circle** around the subject, keeping it centred in frame.
- **Overlap generously** — each shot should share most of its view with the last.
- Keep lighting constant; avoid mirrors, glass and blank walls.
- Move steadily. Motion blur costs more detail than a lower frame count.
- 20–60 photos, or a slow 10–30 second orbit video, is a good target.

---

## Configuration

All optional.

| Variable | Default | Meaning |
|---|---|---|
| `PORT` / `HOST` | `8787` / `127.0.0.1` | listen address |
| `SPLAT_DATA_DIR` | `./data` | where the library and assets live |
| `SPLAT_BACKEND` | `auto` | force `preview` or `colmap` |
| `SPLAT_TRAINER_CMD` | — | trainer command template (see above) |
| `SPLAT_COLMAP_CMD` | `colmap` | COLMAP executable |
| `SPLAT_CONCURRENCY` | `1` | conversions running at once |
| `SPLAT_MAX_UPLOAD` | 2 GiB | per-request upload ceiling |
| `SPLAT_PREVIEW_FRAMES` | `24` | frames the preview backend uses |
| `SPLAT_PREVIEW_GRID` | `160` | preview detail (≈ gaussians per axis) |

---

## How it is put together

```
server/
  index.js            HTTP server, static files, SPA fallback
  api.js              REST endpoints
  jobs.js             conversion queue, progress, cancellation
  store.js            the library — one atomically-rewritten JSON document
  edits.js            non-destructive edits, applied on export
  http/
    router.js         routing, byte ranges, safe static paths
    multipart.js      streaming multipart parser (files go straight to disk)
  pipeline/
    index.js          stage orchestration and progress weighting
    backends.js       capability detection
    colmap.js         COLMAP + trainer invocation
    preview.js        the dependency-free reconstruction
    splat.js          gaussian cloud type, .splat codec, quaternion maths
    ply.js            binary/ascii PLY reader and writer
    png.js            PNG codec built on node:zlib
    imageops.js       resampling, blur, gradients
web/
  js/frames.js        browser-side frame extraction from photos and video
  js/viewer/
    renderer.js       WebGL2 splat renderer
    shaders.js        EWA splatting shaders
    sortWorker.js     depth sorting (16-bit counting sort, off-thread)
    camera.js         orbit camera with mouse and touch
```

### Notes on the renderer

Splat attributes live in two textures indexed by splat id, uploaded once. Each
frame draws one instanced quad per gaussian: the vertex shader projects the 3D
covariance to a screen-space 2D covariance, eigen-decomposes it for the ellipse
axes, and sizes the quad to 3σ. The fragment shader outputs premultiplied alpha
for back-to-front `over` compositing.

Correct blending needs back-to-front order, which changes whenever the camera or
the edit transform moves. Depth is affine in the original position, so the main
thread sends the sort worker three coefficients and a bias rather than matrices,
and positions never have to be re-uploaded.

Edits are shader uniforms, so dragging a slider costs nothing — but export goes
through `server/edits.js`, which applies the same operations on the CPU, so a
downloaded file matches what you saw.

### Data on disk

```
data/
  library.json                  every conversion's metadata
  assets/<id>/
    source/                     your original uploads, kept for re-runs
    frames/                     normalised PNG frames
    work/                       COLMAP database and trainer scratch
    output/                     point_cloud.ply, cloud.splat, thumbnail.png
```

Deleting a conversion in the UI removes its whole directory.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | backend capabilities, queue state |
| `GET` | `/api/assets` | list the library |
| `POST` | `/api/assets` | upload frames, queue a conversion |
| `GET` | `/api/assets/:id` | one conversion |
| `PATCH` | `/api/assets/:id` | name, notes, tags, edits |
| `DELETE` | `/api/assets/:id` | delete it and its files |
| `POST` | `/api/assets/:id/reconvert` | run again with new settings |
| `POST` | `/api/assets/:id/duplicate` | copy so a variant can diverge |
| `POST` | `/api/assets/:id/cancel` | stop a running conversion |
| `GET` | `/api/assets/:id/splat` | the cloud, for the viewer |
| `GET` | `/api/assets/:id/export.ply` | export (`?raw=1` skips edits) |
| `GET` | `/api/assets/:id/export.splat` | export (`?raw=1` skips edits) |
| `GET` | `/api/assets/:id/thumbnail` | preview image |
| `GET` | `/api/events` | server-sent progress stream |

---

## Tests

```bash
npm test
```

67 tests over the codecs (PNG, PLY, `.splat`), the streaming multipart parser,
the edit engine, the reconstruction pipeline, the HTTP layer, and a full
lifecycle integration test that runs a real server: upload → convert → edit →
export → duplicate → delete.

---

## The hosted browser demo

`tools/build-demo.mjs` bundles the reconstruction and the renderer into one
self-contained HTML file that runs the whole preview pipeline client-side — no
server, nothing uploaded, conversions kept in IndexedDB on the viewer's device:

```bash
node tools/build-demo.mjs          # -> tools/demo/build/splatworks-demo.html
```

The build inlines the real modules from `server/pipeline/` and `web/js/viewer/`
rather than restating them, so the demo cannot drift from the tested code. What
can't cross into a single-file page is patched explicitly in the build script:
Node's `Buffer` becomes a typed array, the sort worker is embedded as a blob
URL, and a few module-private helpers are renamed where merging scopes would
collide. The output is asserted to be pure ASCII, because an embedded page
can't declare its own charset.

The demo covers converting, viewing, editing and the library. It cannot do
COLMAP pose solving or `.ply`/`.splat` export — both of those are the local app.

## Limitations worth knowing

- The preview backend assumes an orbit and cannot recover true geometry. It says
  so in the UI; do not treat its output as measurement.
- The viewer evaluates spherical harmonics at degree 0. Higher-order terms in an
  imported `.ply` are parsed but not rendered, so view-dependent highlights from
  a trained model appear flat. Exports are written in the degree-0 form, which
  every viewer reads.
- Conversions run in-process. It is a local-first tool for one person, not a
  multi-tenant service — there is no authentication.
- Very large clouds are bounded by texture size; roughly 5.5M gaussians is the
  ceiling on a 4096-wide texture.
