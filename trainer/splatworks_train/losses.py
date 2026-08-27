"""Training losses: L1 plus D-SSIM, in the paper's 0.8/0.2 mix."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_window(size: int, sigma: float, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(3, 1, size, size).contiguous()


def ssim(a: torch.Tensor, b: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Structural similarity over two [H,W,3] images in 0..1."""
    x = a.permute(2, 0, 1).unsqueeze(0)
    y = b.permute(2, 0, 1).unsqueeze(0)
    window = _gaussian_window(window_size, 1.5, x.device, x.dtype)
    pad = window_size // 2

    mu_x = F.conv2d(x, window, padding=pad, groups=3)
    mu_y = F.conv2d(y, window, padding=pad, groups=3)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_x = F.conv2d(x * x, window, padding=pad, groups=3) - mu_x2
    sigma_y = F.conv2d(y * y, window, padding=pad, groups=3) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=3) - mu_xy

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    value = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / \
            ((mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2))
    return value.mean()


def photometric_loss(rendered, target, lambda_dssim: float = 0.2):
    l1 = (rendered - target).abs().mean()
    d_ssim = 1.0 - ssim(rendered, target)
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * d_ssim, l1, d_ssim


def psnr(rendered, target):
    mse = ((rendered - target) ** 2).mean().clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))
