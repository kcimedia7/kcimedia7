"""Build a splat from a single 360 photo, using predicted depth.

Run as: python -m splatworks_train.from_panorama --images DIR --output DIR

The images are the six views the browser already reprojected out of the
panorama, named after the face they came from.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .depth import FACE_NAMES, build_cloud
from .ply_io import write_ply


def log(message: str) -> None:
    print(message, flush=True)


def load_faces(image_dir: Path):
    """Read the six views and work out which direction each one looked.

    Two ways, because the two callers name their files differently. The server
    stores what it receives as frame_00001.png upwards, discarding the original
    names, so upload order is the only thing left that says which view is
    which -- and the browser emits them in the canonical face order, which is
    the contract those two halves share. A filename that does name its face
    wins, so the backend can also be pointed at a directory by hand.

    Getting this wrong does not fail: the views are all valid images and the
    cloud builds, rotated to somewhere the photograph never looked.
    """
    paths = sorted(p for p in image_dir.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})

    named = {}
    for path in paths:
        for name in FACE_NAMES:
            if name in path.stem.lower():
                named.setdefault(name, path)

    if len(named) == len(FACE_NAMES):
        chosen = [(name, named[name]) for name in FACE_NAMES]
    elif len(paths) == len(FACE_NAMES):
        # Upload order, which is the order the browser reprojects them in.
        chosen = list(zip(FACE_NAMES, paths))
    else:
        raise SystemExit(
            f"expected {len(FACE_NAMES)} panorama views, found {len(paths)} image(s) "
            f"in {image_dir}. This backend converts one 360 photo, which the browser "
            "resamples into exactly six views.")

    images = []
    for name, path in chosen:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        if rgb.shape[0] != rgb.shape[1]:
            raise SystemExit(f"{path.name} is {rgb.shape[1]}x{rgb.shape[0]}, not square. "
                             "The panorama views are square by construction, so this "
                             "directory holds something else.")
        images.append((name, rgb))
    return images


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fov", type=float, default=100.0,
                   help="field of view each view was reprojected at")
    p.add_argument("--max-points", type=int, default=2_000_000,
                   help="thinning below the source resolution costs sharpness, "
                        "so this is high enough to keep a 2k panorama whole")
    p.add_argument("--device", default="cpu")
    p.add_argument("--model", default=None)
    p.add_argument("--opacity", type=float, default=0.9)
    p.add_argument("--far-ratio", type=float, default=60.0,
                   help="how much further the sky sits than the nearest surface")
    args = p.parse_args(argv)

    image_dir = Path(args.images)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    log(f"depth: reading {len(FACE_NAMES)} views from {image_dir}")
    images = load_faces(image_dir)

    from .depth_model import DEFAULT_MODEL, estimate, load_estimator
    model_name = args.model or DEFAULT_MODEL
    log(f"depth: loading {model_name} on {args.device}")
    estimator = load_estimator(model_name, device=args.device)

    disparities = []
    for index, (name, rgb) in enumerate(images, start=1):
        log(f"depth: estimating {name} ({index}/{len(images)})")
        disparities.append(estimate(estimator, rgb))

    log("depth: aligning the views and building the cloud")
    positions, colours, radii = build_cloud(
        images, disparities, args.fov,
        max_points=args.max_points, far_ratio=args.far_ratio,
    )
    count = positions.shape[0]
    if not count:
        raise SystemExit("the depth estimate produced no usable points")

    # Isotropic gaussians, sized to the footprint of the pixel they came from,
    # facing nowhere in particular: there are no surface normals to orient them
    # by, and a splat this dense does not need them.
    quats = np.zeros((count, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    log_scales = np.log(np.maximum(radii, 1e-6))[:, None].repeat(3, axis=1)
    opacity = np.full((count, 1), _logit(args.opacity), dtype=np.float32)
    features_dc = ((colours - 0.5) / 0.28209479177387814)[:, None, :]

    ply_path = output_dir / "point_cloud.ply"
    size = write_ply(
        ply_path, positions, features_dc,
        np.zeros((count, 0, 3), dtype=np.float32),
        opacity, log_scales, quats,
    )
    elapsed = time.time() - started
    log(f"wrote {ply_path} ({count} gaussians, {size} bytes)")

    report = {
        "gaussians": int(count),
        "views": len(images),
        "fov_deg": args.fov,
        "model": model_name,
        "device": args.device,
        "seconds": round(elapsed, 2),
        # Said plainly, because everything downstream shows it beside numbers
        # from reconstructions and the difference matters.
        "depth": "inferred",
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    log(f"depth: done in {elapsed:.1f}s")
    return 0


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


if __name__ == "__main__":
    sys.exit(main())
