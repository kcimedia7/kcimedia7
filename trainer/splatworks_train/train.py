"""Train a 3D Gaussian Splatting model from a folder of images.

Images in, standard 3DGS .ply out. Camera poses come from COLMAP (via
pycolmap), the gaussians are optimised against the posed images with the
differentiable rasterizer in this package, and adaptive density control follows
the paper: clone under-reconstructed gaussians, split over-reconstructed ones,
prune the transparent, and periodically reset opacity so floaters can die.

Runs on CPU. That is slower than the reference CUDA implementation, not
different from it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .losses import photometric_loss, psnr
from .model import GaussianModel
from .ply_io import write_ply
from .rasterizer import build_covariance, rasterize
from .sfm import run_sfm, scene_extent

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_views(views, image_dir, max_dim, device):
    """Load each posed image, rescaling it and its intrinsics together."""
    from PIL import Image

    loaded = []
    for view in views:
        path = Path(image_dir) / view.name
        img = Image.open(path).convert("RGB")
        scale = min(1.0, max_dim / max(img.width, img.height))
        width = max(1, int(round(img.width * scale)))
        height = max(1, int(round(img.height * scale)))
        if (width, height) != (img.width, img.height):
            img = img.resize((width, height), Image.LANCZOS)

        # Intrinsics must follow the resample, or reprojection drifts.
        sx = width / view.width
        sy = height / view.height
        loaded.append({
            "name": view.name,
            "image": torch.tensor(np.asarray(img, dtype=np.float32) / 255.0, device=device),
            "viewmat": torch.tensor(view.world_to_cam, dtype=torch.float32, device=device),
            "K": (view.fx * sx, view.fy * sy, view.cx * sx, view.cy * sy),
            "width": width,
            "height": height,
        })
    return loaded


def exponential_lr(step, total, lr_init, lr_final):
    """Log-linear decay from lr_init to lr_final, as the reference does."""
    if total <= 1:
        return lr_final
    t = min(max(step / total, 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


def train(args, log=print):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)

    image_dir = Path(args.images)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work or (output_dir / "sfm"))

    frames = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if len(frames) < 3:
        raise SystemExit(f"need at least 3 images to reconstruct, found {len(frames)}")
    log(f"reconstructing from {len(frames)} images")

    started = time.time()
    views, points, colors = run_sfm(image_dir, work_dir, matcher=args.matcher, log=log)
    sfm_seconds = time.time() - started

    extent = scene_extent(views)
    log(f"scene extent {extent:.3f}")

    data = load_views(views, image_dir, args.resolution, device)
    log(f"training at {data[0]['width']}x{data[0]['height']}")

    model = GaussianModel.from_points(points, colors, device=device, max_points=args.max_init)
    log(f"initialised {model.count} gaussians from the sparse cloud")

    position_lr_init = 0.00016 * extent
    position_lr_final = 0.0000016 * extent
    optimizer = torch.optim.Adam(model.parameter_groups(position_lr_init), lr=0.0, eps=1e-15)

    background = torch.tensor(
        [args.background] * 3 if isinstance(args.background, float) else args.background,
        dtype=torch.float32, device=device,
    )

    # The paper's schedule assumes 30k iterations; scale it to the budget so a
    # short CPU run still gets several rounds of densification.
    total = args.iterations
    densify_from = max(50, int(0.10 * total))
    densify_until = int(0.60 * total)
    densify_every = max(25, total // 25)
    opacity_reset_every = max(300, int(0.30 * total))

    history = []
    order = []
    train_started = time.time()

    for step in range(1, total + 1):
        for group in optimizer.param_groups:
            if group["name"] == "means":
                group["lr"] = exponential_lr(step, total, position_lr_init, position_lr_final)

        if not order:
            order = list(range(len(data)))
            random.shuffle(order)
        view = data[order.pop()]

        cov3d = build_covariance(model.scales(), model.quats)
        image, info = rasterize(
            model.means, cov3d, model.colors(), model.opacities(),
            view["viewmat"], view["K"], view["width"], view["height"],
            background=background, max_per_tile=args.max_per_tile,
        )
        info["means2d"].retain_grad()

        loss, l1, d_ssim = photometric_loss(image, view["image"], args.lambda_dssim)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if info["means2d"].grad is not None:
            model.record_step(info["means2d"].grad, info["radius"].detach(), info["visible"],
                              viewport=(view["width"], view["height"]))

        optimizer.step()

        if step % max(1, total // 20) == 0 or step == 1:
            with torch.no_grad():
                quality = psnr(image, view["image"])
            loss_value, l1_value = float(loss.detach()), float(l1.detach())
            history.append({"step": step, "loss": loss_value, "l1": l1_value,
                            "psnr": quality, "gaussians": model.count,
                            "overflow_tiles": info.get("overflow_tiles", 0)})
            # A non-zero overflow means the per-tile cap is discarding gaussians
            # and the render no longer matches what the model actually says.
            overflow = info.get("overflow_tiles", 0)
            warn = f"  [!] {overflow} tiles over cap (peak {info.get('max_occupancy', 0)})" if overflow else ""
            log(f"iter {step}/{total}  loss {loss_value:.4f}  l1 {l1_value:.4f}  "
                f"psnr {quality:.2f}dB  gaussians {model.count}{warn}")

        if densify_from <= step <= densify_until and step % densify_every == 0:
            added, pruned = model.densify_and_prune(
                optimizer, args.densify_grad_threshold, extent,
                min_opacity=args.min_opacity, max_count=args.max_gaussians,
            )
            log(f"  densify: +{added} -{pruned} -> {model.count} gaussians")

        if args.opacity_reset and step % opacity_reset_every == 0 and step < densify_until:
            model.reset_opacity(optimizer)
            log("  opacity reset")

    train_seconds = time.time() - train_started

    # Final quality over every training view.
    with torch.no_grad():
        scores = []
        for view in data:
            cov3d = build_covariance(model.scales(), model.quats)
            image, _ = rasterize(
                model.means, cov3d, model.colors(), model.opacities(),
                view["viewmat"], view["K"], view["width"], view["height"],
                background=background, max_per_tile=args.max_per_tile,
            )
            scores.append(psnr(image, view["image"]))
    mean_psnr = float(np.mean(scores))
    log(f"final PSNR over {len(scores)} training views: {mean_psnr:.2f} dB")

    ply_path = output_dir / "point_cloud.ply"
    size = write_ply(
        ply_path,
        model.means.detach().cpu().numpy(),
        model.features.detach().cpu().numpy(),
        model.logit_opacity.detach().cpu().numpy(),
        model.log_scales.detach().cpu().numpy(),
        model.quats.detach().cpu().numpy(),
    )
    log(f"wrote {ply_path} ({model.count} gaussians, {size} bytes)")

    report = {
        "gaussians": model.count,
        "images": len(data),
        "registered_views": len(views),
        "iterations": total,
        "psnr": mean_psnr,
        "per_view_psnr": scores,
        "scene_extent": extent,
        "sfm_seconds": round(sfm_seconds, 2),
        "train_seconds": round(train_seconds, 2),
        "resolution": [data[0]["width"], data[0]["height"]],
        "history": history,
        "ply": str(ply_path),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def build_parser():
    p = argparse.ArgumentParser(
        prog="splatworks-train",
        description="Reconstruct a 3D Gaussian Splatting model from images.",
    )
    p.add_argument("--images", "-s", required=True, help="directory of input frames")
    p.add_argument("--output", "-m", required=True, help="directory for the .ply and report")
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--resolution", type=int, default=320,
                   help="longest edge used for training; lower is much faster on CPU")
    p.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    p.add_argument("--max-gaussians", type=int, default=120_000)
    p.add_argument("--max-init", type=int, default=60_000,
                   help="cap on sparse points used to seed the model")
    p.add_argument("--max-per-tile", type=int, default=4096,
                   help="gaussians composited per tile; below real occupancy this truncates")
    p.add_argument("--densify-grad-threshold", type=float, default=0.0002)
    p.add_argument("--min-opacity", type=float, default=0.005)
    p.add_argument("--lambda-dssim", type=float, default=0.2)
    p.add_argument("--background", type=float, default=0.0)
    p.add_argument("--opacity-reset", action="store_true", default=True)
    p.add_argument("--no-opacity-reset", dest="opacity_reset", action="store_false")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--work", default=None, help="scratch directory for COLMAP")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = train(args)
    print(json.dumps({"psnr": report["psnr"], "gaussians": report["gaussians"]}))


if __name__ == "__main__":
    main()
