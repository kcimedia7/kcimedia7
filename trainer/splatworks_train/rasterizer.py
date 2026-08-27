"""Differentiable 3D Gaussian rasterizer (EWA splatting).

This is the forward half of real 3DGS: gaussians are projected to screen space,
binned into tiles, and alpha-composited front to back. Every operation is a
torch op, so autograd supplies the backward pass that the reference CUDA
implementation hand-derives.

Conventions follow COLMAP/OpenCV so poses from pycolmap drop straight in:
the camera looks down +Z, x right, y down, and a point is in front of the
camera when its camera-space z is positive.
"""

from __future__ import annotations

import math
import torch
from torch.utils.checkpoint import checkpoint

TILE = 16
# Screen-space dilation, as in the paper: keeps a sub-pixel gaussian from
# falling between samples and vanishing.
LOW_PASS = 0.3
# A gaussian is drawn out to 3 sigma; past that its contribution is < 1.2%.
CUTOFF = 3.0


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """[N,4] (w,x,y,z), not necessarily normalised -> [N,3,3]."""
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quat.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def build_covariance(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """World-space 3D covariance Sigma = R S S^T R^T, from [N,3] and [N,4]."""
    rot = quat_to_rotmat(quats)
    scaled = rot * scales.unsqueeze(1)          # R @ diag(s), broadcasting columns
    return scaled @ scaled.transpose(1, 2)


def project_gaussians(means, cov3d, viewmat, K, width, height):
    """World -> screen. Returns 2D means, 2D covariances, depths and radii."""
    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    cam = means @ R.T + t                        # [N,3] camera space
    depth = cam[:, 2]

    fx, fy, cx, cy = K
    # Guard the projection against points at or behind the eye; they are culled
    # by the caller, but the maths must stay finite for autograd.
    z = depth.clamp_min(1e-4)
    inv_z = 1.0 / z
    u = fx * cam[:, 0] * inv_z + cx
    v = fy * cam[:, 1] * inv_z + cy
    means2d = torch.stack([u, v], dim=-1)

    # Perspective Jacobian, evaluated at each gaussian's centre.
    zeros = torch.zeros_like(z)
    J = torch.stack([
        torch.stack([fx * inv_z, zeros, -fx * cam[:, 0] * inv_z * inv_z], dim=-1),
        torch.stack([zeros, fy * inv_z, -fy * cam[:, 1] * inv_z * inv_z], dim=-1),
    ], dim=1)                                    # [N,2,3]

    W = R.unsqueeze(0).expand(means.shape[0], 3, 3)
    T = J @ W                                    # [N,2,3]
    cov2d = T @ cov3d @ T.transpose(1, 2)        # [N,2,2]

    a = cov2d[:, 0, 0] + LOW_PASS
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1] + LOW_PASS

    # Largest eigenvalue of the 2x2 covariance gives the on-screen extent.
    mid = 0.5 * (a + c)
    disc = (mid * mid - (a * c - b * b)).clamp_min(1e-9).sqrt()
    lambda1 = mid + disc
    radius = CUTOFF * lambda1.clamp_min(1e-9).sqrt()

    return means2d, torch.stack([a, b, c], dim=-1), depth, radius


def rasterize(
    means, cov3d, colors, opacities, viewmat, K, width, height,
    background=None, max_per_tile=4096, tile=TILE, tile_chunk=8,
):
    """Render one view. Returns an [H,W,3] image plus per-gaussian diagnostics.

    Gaussians are binned into tiles and composited front to back within each
    tile, which is what makes the cost independent of scene size: a tile only
    pays for the gaussians that actually touch it.
    """
    device = means.device
    n = means.shape[0]
    if background is None:
        background = torch.zeros(3, device=device, dtype=means.dtype)

    means2d, cov2d, depth, radius = project_gaussians(means, cov3d, viewmat, K, width, height)

    visible = (
        (depth > 0.01)
        & (radius > 0.5)
        & (means2d[:, 0] > -radius) & (means2d[:, 0] < width + radius)
        & (means2d[:, 1] > -radius) & (means2d[:, 1] < height + radius)
    )
    idx_visible = visible.nonzero(as_tuple=True)[0]
    if idx_visible.numel() == 0:
        img = background.view(1, 1, 3).expand(height, width, 3)
        return img, {"means2d": means2d, "radius": radius, "visible": visible}

    # Depth order, front to back, established once and inherited by every tile.
    order = torch.argsort(depth[idx_visible])
    sel = idx_visible[order]

    g_mean = means2d[sel]
    g_cov = cov2d[sel]
    g_radius = radius[sel]
    g_color = colors[sel]
    g_opacity = opacities[sel]

    tiles_x = (width + tile - 1) // tile
    tiles_y = (height + tile - 1) // tile
    n_tiles = tiles_x * tiles_y

    ty = torch.arange(tiles_y, device=device)
    tx = torch.arange(tiles_x, device=device)
    tile_cx = (tx.to(means.dtype) + 0.5) * tile
    tile_cy = (ty.to(means.dtype) + 0.5) * tile
    tile_centre = torch.stack(
        torch.meshgrid(tile_cy, tile_cx, indexing="ij"), dim=-1
    ).reshape(-1, 2)                              # [T,2] as (y,x)

    half = tile * 0.5
    # A gaussian touches a tile when their bounding boxes overlap on both axes.
    dx = (tile_centre[:, 1].unsqueeze(1) - g_mean[:, 0].unsqueeze(0)).abs()
    dy = (tile_centre[:, 0].unsqueeze(1) - g_mean[:, 1].unsqueeze(0)).abs()
    reach = g_radius.unsqueeze(0) + half
    touches = (dx <= reach) & (dy <= reach)       # [T,K_all]

    k_all = sel.shape[0]
    rank = torch.arange(k_all, device=device).unsqueeze(0).expand(n_tiles, k_all)
    # Non-touching entries are pushed past every real rank so topk skips them.
    score = torch.where(touches, rank, torch.full_like(rank, k_all))
    k = min(max_per_tile, k_all)
    picked = torch.topk(score, k, dim=1, largest=False).values   # [T,K] ranks
    valid = picked < k_all
    picked = picked.clamp_max(k_all - 1)

    # Compositing is done in chunks of tiles, each recomputed during backward
    # instead of being kept. Memory is what forces a per-tile cap, and a cap is
    # not a free approximation: at 192 it silently dropped most of the gaussians
    # covering 93% of tiles (mean occupancy ~900, peak ~3000), so the optimiser
    # was fitting a truncated image. Checkpointing buys the headroom to raise the
    # cap far above real occupancy, at the cost of one extra forward pass.
    def composite(picked_chunk, valid_chunk, pix_x_chunk, pix_y_chunk):
        t_mean = g_mean[picked_chunk]                 # [C,K,2]
        t_cov = g_cov[picked_chunk]                   # [C,K,3]
        t_color = g_color[picked_chunk]               # [C,K,3]
        t_opacity = g_opacity[picked_chunk].squeeze(-1)

        d_x = pix_x_chunk - t_mean[..., 0].unsqueeze(-1)
        d_y = pix_y_chunk - t_mean[..., 1].unsqueeze(-1)

        a = t_cov[..., 0].unsqueeze(-1)
        b = t_cov[..., 1].unsqueeze(-1)
        c = t_cov[..., 2].unsqueeze(-1)
        det = (a * c - b * b).clamp_min(1e-9)
        power = -0.5 * (c * d_x * d_x - 2 * b * d_x * d_y + a * d_y * d_y) / det

        alpha = (t_opacity.unsqueeze(-1) * power.exp()).clamp(0.0, 0.999)
        alpha = alpha * valid_chunk.unsqueeze(-1)
        alpha = torch.where(power < -0.5 * CUTOFF * CUTOFF, torch.zeros_like(alpha), alpha)

        # Front-to-back "over": T_i is the product of (1-alpha) strictly before i.
        trans = torch.cumprod(1.0 - alpha, dim=1)
        trans_excl = torch.cat([torch.ones_like(trans[:, :1]), trans[:, :-1]], dim=1)
        weight = alpha * trans_excl

        rgb_chunk = (weight.unsqueeze(-1) * t_color.unsqueeze(2)).sum(dim=1)
        return rgb_chunk, trans[:, -1]

    # Pixel coordinates within a tile, shared by every tile.
    py, px = torch.meshgrid(
        torch.arange(tile, device=device, dtype=means.dtype),
        torch.arange(tile, device=device, dtype=means.dtype),
        indexing="ij",
    )
    tile_origin = torch.stack([
        (tile_centre[:, 1] - half), (tile_centre[:, 0] - half),
    ], dim=-1)                                    # [T,2] as (x,y)
    pix_x = (tile_origin[:, 0].view(-1, 1) + px.reshape(1, -1) + 0.5).unsqueeze(1)  # [T,1,P]
    pix_y = (tile_origin[:, 1].view(-1, 1) + py.reshape(1, -1) + 0.5).unsqueeze(1)

    chunk_size = max(1, min(n_tiles, max(1, tile_chunk)))
    rgb_parts, rem_parts = [], []
    for start in range(0, n_tiles, chunk_size):
        stop = min(start + chunk_size, n_tiles)
        args = (picked[start:stop], valid[start:stop], pix_x[start:stop], pix_y[start:stop])
        if means.requires_grad and torch.is_grad_enabled():
            part_rgb, part_rem = checkpoint(composite, *args, use_reentrant=False)
        else:
            part_rgb, part_rem = composite(*args)
        rgb_parts.append(part_rgb)
        rem_parts.append(part_rem)

    rgb = torch.cat(rgb_parts, dim=0)              # [T,P,3]
    remaining = torch.cat(rem_parts, dim=0)        # [T,P]
    rgb = rgb + remaining.unsqueeze(-1) * background.view(1, 1, 3)

    # Tiles back to an image, dropping the padding a non-multiple size adds.
    canvas = rgb.reshape(tiles_y, tiles_x, tile, tile, 3)
    canvas = canvas.permute(0, 2, 1, 3, 4).reshape(tiles_y * tile, tiles_x * tile, 3)
    image = canvas[:height, :width, :]

    # How much the cap actually bit, so truncation is measurable rather than
    # an invisible quality ceiling.
    with torch.no_grad():
        occupancy = touches.sum(dim=1)
        overflow_tiles = int((occupancy > k).sum().item())

    return image, {
        "means2d": means2d,
        "radius": radius,
        "visible": visible,
        "n_rendered": int(valid.sum().item()),
        "overflow_tiles": overflow_tiles,
        "max_occupancy": int(occupancy.max().item()) if occupancy.numel() else 0,
    }
