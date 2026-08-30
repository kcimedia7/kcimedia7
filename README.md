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

## The three backends

A real 3D Gaussian Splat needs two things: structure-from-motion to recover
where each photo was taken, then gradient-descent optimisation of the gaussians
against those posed photos. SplatWorks detects what the machine can do and picks
the best available route.

| | `colmap` | `gaussian` | `preview` |
|---|---|---|---|
| Needs | COLMAP binary + a CUDA trainer | `pip install -r trainer/requirements.txt` | nothing |
| Camera poses | solved (COLMAP) | solved (pycolmap) | **assumed** |
| Gaussians | optimised on GPU | optimised on CPU | not optimised |
| Output | a true reconstruction | a true reconstruction | a fast proxy |
| Time | minutes | minutes to hours | about a second |

The first two are the same algorithm at different speeds. `preview` is a
different thing entirely, and the UI says so rather than letting you assume
otherwise.

### The bundled trainer

`trainer/` is a complete 3D Gaussian Splatting implementation that needs no GPU
and no system installs -- two pip wheels and it runs:

```bash
pip install -r trainer/requirements.txt
```

- **Poses** come from real COLMAP, via the `pycolmap` wheel.
- **Optimisation** uses a differentiable EWA rasterizer written in PyTorch:
  gaussians are projected to screen-space ellipses, binned into tiles, and
  alpha-composited front to back. Autograd supplies the backward pass that the
  reference CUDA implementation derives by hand -- and the gradients are checked
  against finite differences in the test suite, to ~1e-7 relative error.
- **Density control** follows the paper: clone under-reconstructed gaussians,
  split over-reconstructed ones, prune the transparent, and periodically reset
  opacity so floaters can be culled instead of lingering.
- **Loss** is the paper's 0.8 x L1 + 0.2 x D-SSIM.
- **Output** is a standard 3DGS `.ply` that any splat viewer reads.

Run it directly on a folder of images:

```bash
cd trainer
python -m splatworks_train.train --images ./frames --output ./model \
    --iterations 3000 --resolution 320
```

It prints per-iteration loss, PSNR and gaussian count, and writes
`point_cloud.ply` plus a `report.json` of metrics.

**On CPU this is slow.** That is the honest trade: it is the same algorithm as
the CUDA implementation, not a cheaper approximation of it. The reference runs
30,000 iterations on a GPU; a CPU run of a few thousand at reduced resolution
gets you a real but softer reconstruction. Point `SPLAT_TRAINER_CMD` at a GPU
trainer when you want the full thing.

### Enabling the external GPU path


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

### 360 panoramas

Equirectangular 360 photos are accepted directly, as JPEG/PNG or as Radiance
`.hdr`. Each one is resampled in the browser into six overlapping perspective
views before upload, because an equirectangular image is a spherical projection
and structure-from-motion models a pinhole camera — feeding one in unchanged
does not fail, it fits the wrong model and returns plausible nonsense.

The thing to understand before shooting:

- **One panorama cannot produce real geometry.** Depth comes from parallax, and
  a single 360 shot has one optical centre. You get a textured shell around the
  camera, not measured structure. The app says so when you select one file.
- **Shoot two or more, a step or two apart.** A 360 camera carried through a
  room, stopping every metre or so, is excellent capture data — each stop covers
  the whole room at once, so coverage is far better than a phone can manage.
- **Panoramas cannot be mixed with ordinary photos** in one conversion. They
  become square views, and the solver fits one shared camera to a capture, which
  it enforces by skipping images of any other shape.
- **HDR is tone mapped on the way in.** Splats store 8-bit colour, so the range
  has to collapse somewhere; doing it deliberately keeps detail in both the
  windows and the shadows, where a plain conversion would clip the highlights to
  featureless white.
- **OpenEXR is not supported.** Save as `.hdr` instead.

---

## Configuration

All optional.

| Variable | Default | Meaning |
|---|---|---|
| `PORT` / `HOST` | `8787` / `127.0.0.1` | listen address |
| `SPLAT_DATA_DIR` | `./data` | where the library and assets live |
| `SPLAT_BACKEND` | `auto` | force `preview` or `colmap` |
| `SPLAT_TRAINER_CMD` | — | trainer command template (see above) |
| `SPLAT_TRAIN_DEVICE` | `cpu` | `cuda` to train on an NVIDIA GPU with the bundled trainer |
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

## The standalone build

`tools/build-demo.mjs` bundles the reconstruction and the renderer into one
self-contained HTML file that runs the whole preview pipeline client-side — no
server, nothing uploaded, conversions kept in IndexedDB on the viewer's device,
and `.ply` / `.splat` / PNG exported straight to disk:

```bash
node tools/build-demo.mjs     # -> dist/index.html (+ a copy for embedding)
open dist/index.html          # no server required
```

The build inlines the real modules from `server/pipeline/` and `web/js/viewer/`
rather than restating them, so the demo cannot drift from the tested code. What
can't cross into a single-file page is patched explicitly in the build script:
Node's `Buffer` becomes a typed array, the sort worker is embedded as a blob
URL, and a few module-private helpers are renamed where merging scopes would
collide. The output is asserted to be pure ASCII, because an embedded page
can't declare its own charset.

It covers converting, viewing, editing, the library and export. The one thing
it cannot do is COLMAP pose solving, which needs a GPU host — that is the server
app's job.

`dist/` is what Netlify publishes; the second copy is the page published as a
Claude artifact, where the sandbox blocks page-initiated downloads. The build is
identical and the page detects at runtime which home it is in, offering the
export buttons only where they can actually work.

## Deploying

```bash
netlify deploy --build --prod    # the standalone build; netlify.toml is committed
docker build -t splatworks . && docker run -p 80:8787 -v splatworks-data:/data splatworks
```

Netlify hosts the standalone build, AWS hosts the server app, and
[docs/DEPLOY.md](docs/DEPLOY.md) explains why that split is the right one — the
short version being that multi-gigabyte uploads, a persistent library, a job
queue and an SSE stream do not fit inside a 10-second Lambda.

## When a conversion finishes but shows nothing

The gaussian count in the viewer comes from the file header, so it reads
correctly whatever the values are. A conversion that diverged writes a valid
PLY, loads, sorts and reports a frame rate — and draws nothing. To find out
which it is:

```
npm run inspect            # the newest conversion
npm run inspect --list     # everything in the library
npm run inspect <id>       # a specific conversion, or any .ply path
```

Run it from the project directory — `npm run` needs the project's
`package.json`, and a shell opened somewhere else (PowerShell starts in
`C:\windows\system32`) will fail with `ENOENT ... package.json`.

It reports non-finite values, the opacity and scale distributions, and the scene
extent, then names what would make the render blank. Note that the thumbnail on
each library card is the **middle source frame**, not a render of the result —
so a good-looking thumbnail says the upload worked, nothing more.

The most common cause is a capture with too little parallax. Walking straight
down a road or straight towards a subject moves the camera along its own view
direction, which barely constrains depth; the solve is poorly conditioned and
can diverge. Orbiting around the subject fixes it.

---

## Limitations worth knowing

- The preview backend, used only when neither trainer is available, assumes an
  orbit and cannot recover true geometry. It says so in the UI.
- The bundled trainer optimises degree-0 spherical harmonics, so colour is
  view-independent: no specular highlights that change with the viewing angle.
  The geometry is real anisotropic 3D gaussians either way.
- The viewer evaluates spherical harmonics at degree 0. Higher-order terms in an
  imported `.ply` are parsed but not rendered, so view-dependent highlights from
  a trained model appear flat. Exports are written in the degree-0 form, which
  every viewer reads.
- Conversions run in-process. It is a local-first tool for one person, not a
  multi-tenant service — there is no authentication.
- Very large clouds are bounded by texture size; roughly 5.5M gaussians is the
  ceiling on a 4096-wide texture.
