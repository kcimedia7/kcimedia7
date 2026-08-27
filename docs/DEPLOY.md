# Deploying SplatWorks

The project has two deployable shapes, and they want different hosts.

| | Standalone build | Server app |
|---|---|---|
| Host | **Netlify** (any static host) | **AWS** (EC2 / Fargate) |
| Backend | preview reconstruction, in the browser | preview, or real COLMAP + trainer |
| Library | the visitor's IndexedDB | server disk, shared |
| Uploads | never leave the device | up to 2 GiB per request |
| Export | `.ply`, `.splat`, PNG | `.ply`, `.splat`, PNG |
| Cost | free | an instance |

**Why not both on Netlify?** The server app needs multi-gigabyte uploads, a
persistent data directory, a job queue that outlives a request, and an SSE
stream. Netlify Functions are Lambdas: ~6 MB request payloads, a 10 s
synchronous ceiling, and an ephemeral filesystem. The conversion pipeline does
not fit inside that, and pretending otherwise would produce a deploy that
breaks on the first real video.

**Why Netlify is still the right home for the standalone build.** In that mode
the app genuinely has no server: the browser decodes frames, reconstruction runs
on the page, and results are exported straight to disk. A static host is not a
compromise there — it is the whole architecture.

---

## Netlify — the standalone build

`netlify.toml` is committed and ready. The build is `node tools/build-demo.mjs`,
publishing `dist/`, with no dependencies to install.

**Connect the repository** (recommended, gives you deploy previews):

1. Netlify → *Add new site* → *Import an existing project* → pick this repo.
2. Netlify reads `netlify.toml`; the build command and publish directory are
   already correct. Nothing to fill in.
3. Deploy.

**Or from your machine:**

```bash
npm i -g netlify-cli
netlify deploy --build --prod
```

**Or with no Netlify account at all** — the build output is a single file, so
any static host works, including opening it directly:

```bash
node tools/build-demo.mjs
open dist/index.html
```

### What the committed headers do

`netlify.toml` ships a Content-Security-Policy tight enough to be worth
explaining, since each exception is load-bearing:

- `script-src 'unsafe-inline'` — the module script is inlined into the page
- `worker-src blob:` — the depth sorter is constructed from a blob URL
- `media-src blob:` — video files are read via `URL.createObjectURL`
- `img-src data: blob:` — library thumbnails are data URLs
- `fonts.googleapis.com` / `fonts.gstatic.com` — the only external host contacted

The full flow (video decode, conversion, thumbnails, `.ply` export) has been
verified against exactly these headers.

---

## AWS — the server app

The app is stateful: the library and every capture's frames live in
`SPLAT_DATA_DIR`. Pick a shape that gives it a real disk.

- **EC2 + EBS** — simplest, and the only option if you want real reconstruction,
  since COLMAP and a 3DGS trainer need a GPU instance anyway. Start here.
- **ECS Fargate + EFS** — if you want it managed. Mount EFS at `/data`.
- **App Runner** — works, but its storage is ephemeral: the library is lost on
  every deploy. Only sensible if you treat conversions as disposable.

### Preview backend (CPU, no GPU)

```bash
docker build -t splatworks .
docker run -d --name splatworks \
  -p 80:8787 \
  -v splatworks-data:/data \
  --restart unless-stopped \
  splatworks
```

On a `t3.small` that is enough to run conversions with the preview backend.
Put it behind an ALB or Caddy for TLS; **the app has no authentication**, so do
not expose it to the internet without putting something in front of it.

### Real reconstruction (GPU)

Real 3D Gaussian Splatting needs COLMAP for camera poses and a CUDA trainer.
Use a GPU instance (`g4dn.xlarge` upward) with the NVIDIA drivers and container
toolkit installed, an image that carries COLMAP and your trainer of choice, and
point the app at the trainer:

```bash
docker run -d --gpus all \
  -p 80:8787 -v splatworks-data:/data \
  -e SPLAT_TRAINER_CMD="python /opt/gaussian-splatting/train.py -s {source} -m {output} --iterations {iterations}" \
  your-image-with-colmap-and-trainer
```

The app detects COLMAP on `PATH` plus `SPLAT_TRAINER_CMD` and switches backends
on its own — the badge in the UI reports which one is live. With neither, it
falls back to the preview backend rather than failing.

Sizing: training is minutes to hours per capture and wants a whole GPU, so keep
`SPLAT_CONCURRENCY=1` (the default) unless you have several.

### Configuration

Every setting is an environment variable; see the table in the main README.
The ones that matter in a container:

| Variable | Set it to |
|---|---|
| `HOST` | `0.0.0.0` (the image already does this) |
| `SPLAT_DATA_DIR` | a mounted volume, e.g. `/data` |
| `SPLAT_MAX_UPLOAD` | raise or lower the 2 GiB default |
| `SPLAT_TRAINER_CMD` | your trainer, to enable the COLMAP backend |

### Health check

`GET /api/health` returns 200 with the active backend and queue state — use it
as the ALB/ECS health check. The image has a `HEALTHCHECK` that calls it.

---

## Notes

Neither deploy was executed from the session that wrote these files: Netlify's
API is unreachable from it, and creating AWS infrastructure in someone's account
is not something to do unasked. What *was* verified is the part that usually
breaks — the standalone build running end to end under the exact CSP shipped in
`netlify.toml`, and the container's runtime contract (binding `0.0.0.0`, the
health check, and `SPLAT_DATA_DIR`). The Docker image itself has not been built.
