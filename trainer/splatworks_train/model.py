"""The optimisable gaussian cloud, with the paper's adaptive density control.

Parameters are stored in the same activated form the reference implementation
uses -- log scales, logit opacity, unnormalised quaternions, degree-0 spherical
harmonics -- so a trained model writes out as a standard 3DGS .ply.
"""

from __future__ import annotations

import math
import numpy as np
import torch

SH_C0 = 0.28209479177387814


def inverse_sigmoid(x: float) -> float:
    return math.log(x / (1 - x))


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    return sh * SH_C0 + 0.5


def mean_nn_distance(points: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean distance to the k nearest neighbours, used to size new gaussians.

    Chunked so a large sparse cloud does not need an N^2 matrix all at once.
    """
    n = points.shape[0]
    out = np.zeros(n, dtype=np.float32)
    chunk = max(1, min(2048, n))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = np.linalg.norm(points[start:stop, None, :] - points[None, :, :], axis=-1)
        d[np.arange(stop - start), np.arange(start, stop)] = np.inf
        take = min(k, max(1, n - 1))
        nearest = np.partition(d, take - 1, axis=1)[:, :take]
        out[start:stop] = nearest.mean(axis=1)
    return np.clip(out, 1e-7, None)


class GaussianModel:
    def __init__(self, means, scales, quats, opacity, features, device="cpu"):
        self.device = device
        self.means = means.to(device).requires_grad_(True)
        self.log_scales = scales.to(device).requires_grad_(True)
        self.quats = quats.to(device).requires_grad_(True)
        self.logit_opacity = opacity.to(device).requires_grad_(True)
        self.features = features.to(device).requires_grad_(True)   # SH degree 0
        self._reset_accumulators()

    # ---------------------------------------------------------------- factory

    @classmethod
    def from_points(cls, points: np.ndarray, colors: np.ndarray, device="cpu",
                    initial_opacity=0.1, max_points=None):
        """Initialise from an SfM sparse cloud, exactly as the paper does."""
        if max_points and points.shape[0] > max_points:
            keep = np.random.default_rng(0).choice(points.shape[0], max_points, replace=False)
            points, colors = points[keep], colors[keep]

        dist = mean_nn_distance(points)
        n = points.shape[0]
        quats = np.zeros((n, 4), dtype=np.float32)
        quats[:, 0] = 1.0

        return cls(
            means=torch.tensor(points, dtype=torch.float32),
            # An isotropic gaussian roughly filling the gap to its neighbours.
            scales=torch.tensor(np.log(dist)[:, None].repeat(3, axis=1), dtype=torch.float32),
            quats=torch.tensor(quats),
            opacity=torch.full((n, 1), inverse_sigmoid(initial_opacity), dtype=torch.float32),
            features=rgb_to_sh(torch.tensor(colors, dtype=torch.float32)),
            device=device,
        )

    # ------------------------------------------------------------ activations

    @property
    def count(self) -> int:
        return self.means.shape[0]

    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_opacity)

    def colors(self) -> torch.Tensor:
        return sh_to_rgb(self.features).clamp(0.0, 1.0)

    def parameter_groups(self, position_lr):
        return [
            {"params": [self.means], "lr": position_lr, "name": "means"},
            {"params": [self.features], "lr": 2.5e-3, "name": "features"},
            {"params": [self.logit_opacity], "lr": 0.05, "name": "opacity"},
            {"params": [self.log_scales], "lr": 5e-3, "name": "scales"},
            {"params": [self.quats], "lr": 1e-3, "name": "quats"},
        ]

    # ------------------------------------------------- density control state

    def _reset_accumulators(self):
        self.grad_accum = torch.zeros(self.count, device=self.device)
        self.grad_denom = torch.zeros(self.count, device=self.device)
        self.max_radius = torch.zeros(self.count, device=self.device)

    def record_step(self, means2d_grad, radius, visible, viewport=None):
        """Accumulate the screen-space positional gradient that drives densification.

        The gradient arrives in pixels, but the paper's 0.0002 threshold is in
        normalised device coordinates -- a factor of width/2 apart. Measured on a
        160x120 view, the pixel-space gradient selects 0% of gaussians at that
        threshold while the NDC-space one selects 45%, which is why densification
        silently did nothing until this rescale was added. Convert here so the
        published constant means what it says.
        """
        with torch.no_grad():
            grad = means2d_grad[visible]
            if viewport is not None:
                width, height = viewport
                grad = grad * torch.tensor(
                    [width * 0.5, height * 0.5], device=grad.device, dtype=grad.dtype
                )
            grad_norm = grad.norm(dim=-1)
            self.grad_accum[visible] += grad_norm
            self.grad_denom[visible] += 1
            self.max_radius[visible] = torch.maximum(self.max_radius[visible], radius[visible])

    # ------------------------------------------------------------- mutations

    def _tensors(self):
        return ["means", "features", "logit_opacity", "log_scales", "quats"]

    ATTR_FOR_GROUP = {
        "means": "means", "features": "features", "opacity": "logit_opacity",
        "scales": "log_scales", "quats": "quats",
    }

    def _replace(self, optimizer, new_values, state_fn):
        """Swap parameters, carrying Adam's moments across with them.

        `state_fn` maps an old moment tensor to the new one. Getting this wrong
        is expensive and silent: rebuilding the moments from zeros on every
        densification round throws away the momentum of every surviving gaussian,
        and with densification running every ~25 iterations that is most of the
        optimiser's memory, most of the time.
        """
        for group in optimizer.param_groups:
            name = group["name"]
            old = group["params"][0]
            state = optimizer.state.pop(old, None)
            replacement = new_values[name].clone().requires_grad_(True)
            if state is not None:
                state["exp_avg"] = state_fn(state["exp_avg"])
                state["exp_avg_sq"] = state_fn(state["exp_avg_sq"])
                optimizer.state[replacement] = state
            group["params"][0] = replacement
            setattr(self, self.ATTR_FOR_GROUP[name], replacement)

    def _current_values(self):
        return {
            "means": self.means.detach(),
            "features": self.features.detach(),
            "opacity": self.logit_opacity.detach(),
            "scales": self.log_scales.detach(),
            "quats": self.quats.detach(),
        }

    def prune(self, optimizer, keep_mask):
        values = {k: v[keep_mask] for k, v in self._current_values().items()}
        # Surviving rows keep their moments; dropped rows take theirs with them.
        self._replace(optimizer, values, lambda t: t[keep_mask])
        self.grad_accum = self.grad_accum[keep_mask]
        self.grad_denom = self.grad_denom[keep_mask]
        self.max_radius = self.max_radius[keep_mask]

    def append(self, optimizer, extra):
        current = self._current_values()
        values = {k: torch.cat([current[k], extra[k]], 0) for k in current}
        n_new = extra["means"].shape[0]
        # Existing rows keep their moments; only the new gaussians start cold.
        self._replace(
            optimizer, values,
            lambda t: torch.cat([t, t.new_zeros((n_new, *t.shape[1:]))], 0),
        )
        self._reset_accumulators()

    def densify_and_prune(self, optimizer, grad_threshold, extent, min_opacity=0.005,
                          percent_dense=0.01, max_count=None):
        """One round of the paper's density control: clone, split, then prune."""
        grads = torch.where(self.grad_denom > 0, self.grad_accum / self.grad_denom.clamp_min(1),
                            torch.zeros_like(self.grad_accum))
        scales = self.scales().detach()
        big = scales.max(dim=1).values > percent_dense * extent
        wants_more = grads >= grad_threshold

        headroom = None if max_count is None else max(0, max_count - self.count)

        # Under-reconstructed regions: copy the gaussian and let the pair separate.
        clone_mask = wants_more & ~big
        # Over-reconstructed regions: replace one large gaussian with two smaller.
        split_mask = wants_more & big
        if headroom is not None:
            allowed = torch.zeros_like(clone_mask)
            picks = (clone_mask | split_mask).nonzero(as_tuple=True)[0][:headroom]
            allowed[picks] = True
            clone_mask &= allowed
            split_mask &= allowed

        extra = {k: [] for k in ["means", "features", "opacity", "scales", "quats"]}

        if clone_mask.any():
            extra["means"].append(self.means.detach()[clone_mask])
            extra["features"].append(self.features.detach()[clone_mask])
            extra["opacity"].append(self.logit_opacity.detach()[clone_mask])
            extra["scales"].append(self.log_scales.detach()[clone_mask])
            extra["quats"].append(self.quats.detach()[clone_mask])

        split_count = int(split_mask.sum().item())
        if split_count:
            from .rasterizer import quat_to_rotmat
            src_scales = scales[split_mask]
            rot = quat_to_rotmat(self.quats.detach()[split_mask])
            # New centres are drawn from the gaussian being replaced.
            samples = torch.randn(split_count, 3, device=self.device) * src_scales
            offset = torch.bmm(rot, samples.unsqueeze(-1)).squeeze(-1)
            centre = self.means.detach()[split_mask]
            for sign in (1.0, -1.0):
                extra["means"].append(centre + sign * offset * 0.5)
                extra["features"].append(self.features.detach()[split_mask])
                extra["opacity"].append(self.logit_opacity.detach()[split_mask])
                extra["scales"].append(torch.log(src_scales / 1.6))
                extra["quats"].append(self.quats.detach()[split_mask])

        added = 0
        if any(len(v) for v in extra.values()):
            merged = {k: torch.cat(v, 0) for k, v in extra.items() if v}
            added = merged["means"].shape[0]
            self.append(optimizer, merged)

        # Splitting replaces the original, so drop it along with faint gaussians.
        keep = self.opacities().detach().squeeze(-1) > min_opacity
        keep &= self.scales().detach().max(dim=1).values < 0.5 * extent
        if split_count:
            drop_original = torch.zeros(self.count, dtype=torch.bool, device=self.device)
            drop_original[:split_mask.shape[0]] = split_mask
            keep &= ~drop_original
        pruned = int((~keep).sum().item())
        if pruned and int(keep.sum().item()) > 0:
            self.prune(optimizer, keep)

        self._reset_accumulators()
        return added, pruned

    def reset_opacity(self, optimizer, value=0.01):
        """Periodic knock-down that lets floaters be pruned instead of lingering."""
        values = self._current_values()
        values["opacity"] = torch.minimum(
            values["opacity"], torch.full_like(values["opacity"], inverse_sigmoid(value)),
        )
        # Shapes are unchanged, but the moments describe the old opacities, so
        # clear them rather than fighting the reset.
        self._replace(optimizer, values, lambda t: torch.zeros_like(t))
