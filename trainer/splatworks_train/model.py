"""The optimisable gaussian cloud, with the paper's adaptive density control.

Parameters are stored in the same activated form the reference implementation
uses -- log scales, logit opacity, unnormalised quaternions, degree-0 spherical
harmonics -- so a trained model writes out as a standard 3DGS .ply.
"""

from __future__ import annotations

import math
import numpy as np
import torch

from .sh import MAX_DEGREE, coefficient_count, eval_sh

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
    def __init__(self, means, scales, quats, opacity, features_dc, features_rest,
                 device="cpu", sh_degree=MAX_DEGREE):
        self.device = device
        self.means = means.to(device).requires_grad_(True)
        self.log_scales = scales.to(device).requires_grad_(True)
        self.quats = quats.to(device).requires_grad_(True)
        self.logit_opacity = opacity.to(device).requires_grad_(True)
        # Colour is split the way the reference splits it: the constant term
        # and the view-dependent bands are separate parameters because they
        # want very different learning rates. The higher bands are a correction
        # to a colour that is already roughly right, and letting them move as
        # fast as the base colour makes training unstable.
        self.features_dc = features_dc.to(device).requires_grad_(True)      # (N, 1, 3)
        self.features_rest = features_rest.to(device).requires_grad_(True)  # (N, 15, 3)
        self.sh_degree = sh_degree
        # Bands are switched on one at a time during training, not all at once.
        self.active_sh_degree = 0
        self._reset_accumulators()

    # ---------------------------------------------------------------- factory

    @classmethod
    def from_points(cls, points: np.ndarray, colors: np.ndarray, device="cpu",
                    initial_opacity=0.1, max_points=None, sh_degree=MAX_DEGREE):
        """Initialise from an SfM sparse cloud, exactly as the paper does."""
        if max_points and points.shape[0] > max_points:
            keep = np.random.default_rng(0).choice(points.shape[0], max_points, replace=False)
            points, colors = points[keep], colors[keep]

        dist = mean_nn_distance(points)
        n = points.shape[0]
        quats = np.zeros((n, 4), dtype=np.float32)
        quats[:, 0] = 1.0

        # The sparse cloud gives one colour per point, which is exactly the
        # constant term. The view-dependent bands start at zero: the model
        # begins as the flat-shaded scene it used to be and earns the rest.
        rest_count = coefficient_count(sh_degree) - 1
        return cls(
            means=torch.tensor(points, dtype=torch.float32),
            # An isotropic gaussian roughly filling the gap to its neighbours.
            scales=torch.tensor(np.log(dist)[:, None].repeat(3, axis=1), dtype=torch.float32),
            quats=torch.tensor(quats),
            opacity=torch.full((n, 1), inverse_sigmoid(initial_opacity), dtype=torch.float32),
            features_dc=rgb_to_sh(torch.tensor(colors, dtype=torch.float32)).unsqueeze(1),
            features_rest=torch.zeros((n, rest_count, 3), dtype=torch.float32),
            device=device,
            sh_degree=sh_degree,
        )

    # ------------------------------------------------------------ activations

    @property
    def count(self) -> int:
        return self.means.shape[0]

    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_opacity)

    def features(self) -> torch.Tensor:
        """All coefficients as one (N, C, 3) tensor, constant term first."""
        return torch.cat([self.features_dc, self.features_rest], dim=1)

    def colors(self, dirs: torch.Tensor | None = None) -> torch.Tensor:
        """Colour per gaussian, view-dependent when a direction is supplied.

        Without directions this returns the constant term alone -- the diffuse
        appearance, which is what a viewer that ignores the higher bands shows
        and what the exported `.splat` carries.
        """
        if dirs is None or self.active_sh_degree == 0:
            return sh_to_rgb(self.features_dc[:, 0]).clamp(0.0, 1.0)
        value = eval_sh(self.active_sh_degree, self.features(), dirs)
        # The 0.5 shift is part of the encoding, not a tone curve: an
        # all-zero model is mid grey rather than black.
        return (value + 0.5).clamp(0.0, 1.0)

    def raise_sh_degree(self) -> bool:
        """Switch on the next band, if there is one left."""
        if self.active_sh_degree >= self.sh_degree:
            return False
        self.active_sh_degree += 1
        return True

    def tensors(self):
        """Every optimised tensor, in no particular order."""
        return [self.means, self.features_dc, self.features_rest,
                self.logit_opacity, self.log_scales, self.quats]

    def sanitise_gradients(self) -> "torch.Tensor":
        """Zero any gradient entry that is not finite, and count how many.

        A single non-finite gradient is not a passing problem. Adam keeps a
        running mean and variance per parameter, so once NaN reaches that state
        every later step for that parameter is NaN too -- the damage is
        permanent and silent, and it is why a run can end with a small
        percentage of gaussians ruined rather than none or all of them.

        The count comes back as a tensor so the caller can decide when to pay
        for a device sync rather than stalling on every iteration.
        """
        bad = torch.zeros((), dtype=torch.long, device=self.device)
        for t in self.tensors():
            if t.grad is None:
                continue
            with torch.no_grad():
                bad += (~torch.isfinite(t.grad)).sum()
                torch.nan_to_num_(t.grad, nan=0.0, posinf=0.0, neginf=0.0)
        return bad

    def clamp_scales(self, extent: float) -> None:
        """Hold gaussian sizes inside a range the rasterizer can integrate.

        A gaussian whose scale collapses gives a covariance with a vanishing
        determinant. The rasterizer floors that determinant to stay finite, but
        the gradient still carries a 1/det**2 term, so the collapse produces
        enormous gradients that throw the whole model apart -- the failure looks
        like exploding positions when it started with shrinking scales.

        The bounds are proportional to the scene, so this is a guard rail rather
        than a change in behaviour: a healthy gaussian sits orders of magnitude
        inside them.
        """
        lo = math.log(max(extent, 1e-6) * 1e-6)
        hi = math.log(max(extent, 1e-6) * 0.5)
        with torch.no_grad():
            self.log_scales.clamp_(lo, hi)

    def non_finite_count(self) -> "torch.Tensor":
        """How many parameter entries have gone non-finite, as a tensor."""
        bad = torch.zeros((), dtype=torch.long, device=self.device)
        with torch.no_grad():
            for t in self.tensors():
                bad += (~torch.isfinite(t)).sum()
        return bad

    def parameter_groups(self, position_lr):
        return [
            {"params": [self.means], "lr": position_lr, "name": "means"},
            {"params": [self.features_dc], "lr": 2.5e-3, "name": "features_dc"},
            # A twentieth of the base colour's rate, as the reference uses. The
            # higher bands correct a colour that is already close, so letting
            # them move at full speed trades stability for nothing.
            {"params": [self.features_rest], "lr": 2.5e-3 / 20.0, "name": "features_rest"},
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
        return ["means", "features_dc", "features_rest", "logit_opacity",
                "log_scales", "quats"]

    ATTR_FOR_GROUP = {
        "means": "means", "features_dc": "features_dc",
        "features_rest": "features_rest", "opacity": "logit_opacity",
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
            "features_dc": self.features_dc.detach(),
            "features_rest": self.features_rest.detach(),
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
                          percent_dense=0.01, max_count=None, max_screen_size=None,
                          max_world_fraction=0.1):
        """One round of the paper's density control: clone, split, then prune.

        `max_world_fraction` and `max_screen_size` are what keep the result
        looking like a scene rather than a bundle of needles. A gaussian that
        grows to a large fraction of the scene, or that covers a large part of
        the frame, is not describing a surface any more -- it is a smear that
        happens to reduce the loss, and it hides everything behind it. The paper
        prunes both; without them the model stays cheap and wrong.
        """
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

        extra = {k: [] for k in ["means", "features_dc", "features_rest", "opacity",
                                 "scales", "quats"]}

        if clone_mask.any():
            extra["means"].append(self.means.detach()[clone_mask])
            extra["features_dc"].append(self.features_dc.detach()[clone_mask])
            extra["features_rest"].append(self.features_rest.detach()[clone_mask])
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
                extra["features_dc"].append(self.features_dc.detach()[split_mask])
                extra["features_rest"].append(self.features_rest.detach()[split_mask])
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
        # The paper's threshold is a tenth of the scene. Half, which this used
        # to allow, lets a single gaussian span the whole reconstruction: those
        # are the streaks that read as motion blur across an otherwise sensible
        # scene.
        keep &= self.scales().detach().max(dim=1).values < max_world_fraction * extent
        if max_screen_size is not None:
            # Screen extent catches what world extent cannot: a gaussian close
            # to a camera can be small in the world and still cover half the
            # frame.
            #
            # Cloning and splitting above have already grown the model, so the
            # recorded radii cover only the gaussians that existed when the
            # frames were rendered. The new ones have never been drawn, so they
            # get a radius of zero and survive this round on their merits.
            radii = torch.zeros(keep.shape[0], device=self.device)
            seen = min(self.max_radius.shape[0], keep.shape[0])
            radii[:seen] = self.max_radius[:seen]
            keep &= radii < max_screen_size
        if split_count:
            drop_original = torch.zeros(self.count, dtype=torch.bool, device=self.device)
            drop_original[:split_mask.shape[0]] = split_mask
            keep &= ~drop_original
        # Refuse to empty the model: if every gaussian fails the tests, the
        # thresholds are wrong for this scene and deleting everything turns a
        # poor reconstruction into no reconstruction. Report what was actually
        # removed rather than what was selected, or the log claims prunes that
        # never happened.
        survivors = int(keep.sum().item())
        pruned = 0
        if survivors and survivors < keep.shape[0]:
            pruned = keep.shape[0] - survivors
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
