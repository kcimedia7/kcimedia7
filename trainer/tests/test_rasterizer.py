"""The rasterizer is the piece everything else trusts, so check its gradients.

Autograd supplies the backward pass, but only for the maths actually written in
the forward pass. Comparing against finite differences is what catches a wrong
Jacobian or a mis-transposed covariance -- errors that still produce a plausible
picture and quietly ruin training.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splatworks_train.rasterizer import (  # noqa: E402
    build_covariance, quat_to_rotmat, rasterize, project_gaussians,
)


def make_scene(n=6, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    means = torch.randn(n, 3, dtype=dtype) * 0.35
    scales = torch.full((n, 3), 0.12, dtype=dtype) + torch.rand(n, 3, dtype=dtype) * 0.05
    quats = torch.randn(n, 4, dtype=dtype)
    colors = torch.rand(n, 3, dtype=dtype)
    opacity = torch.full((n, 1), 0.7, dtype=dtype)
    view = torch.eye(4, dtype=dtype)
    view[2, 3] = 3.0
    return means, scales, quats, colors, opacity, view


def render(means, scales, quats, colors, opacity, view, w=48, h=36):
    K = (40.0, 40.0, w / 2, h / 2)
    image, _ = rasterize(means, build_covariance(scales, quats), colors, opacity, view, K, w, h)
    return image


def test_render_is_finite_and_in_range():
    means, scales, quats, colors, opacity, view = make_scene()
    image = render(means, scales, quats, colors, opacity, view)
    assert image.shape == (36, 48, 3)
    assert torch.isfinite(image).all()
    assert image.min() >= -1e-9 and image.max() <= 1.0 + 1e-6


def test_quaternion_to_rotation_is_orthonormal():
    q = torch.randn(8, 4, dtype=torch.float64)
    R = quat_to_rotmat(q)
    eye = torch.eye(3, dtype=torch.float64).expand(8, 3, 3)
    assert torch.allclose(R @ R.transpose(1, 2), eye, atol=1e-10)
    assert torch.allclose(torch.linalg.det(R), torch.ones(8, dtype=torch.float64), atol=1e-10)


def test_covariance_is_symmetric_positive_semidefinite():
    _, scales, quats, *_ = make_scene(12)
    cov = build_covariance(scales, quats)
    assert torch.allclose(cov, cov.transpose(1, 2), atol=1e-12)
    eigenvalues = torch.linalg.eigvalsh(cov)
    assert (eigenvalues > 0).all()


def test_projection_places_a_centred_gaussian_at_the_principal_point():
    means = torch.zeros(1, 3, dtype=torch.float64)
    cov = build_covariance(torch.full((1, 3), 0.1, dtype=torch.float64),
                           torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64))
    view = torch.eye(4, dtype=torch.float64)
    view[2, 3] = 4.0
    means2d, _, depth, radius = project_gaussians(means, cov, view, (50.0, 50.0, 32.0, 24.0), 64, 48)
    assert torch.allclose(means2d[0], torch.tensor([32.0, 24.0], dtype=torch.float64))
    assert depth.item() == pytest.approx(4.0)
    assert radius.item() > 0


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
def test_gradients_match_finite_differences(index):
    """Every optimised parameter group gets its analytic gradient checked."""
    torch.manual_seed(index)
    means, scales, quats, colors, opacity, view = make_scene(seed=index)
    tensors = [means, scales, quats, colors, opacity]
    names = ["means", "scales", "quats", "colors", "opacity"]
    for t in tensors:
        t.requires_grad_(True)

    target = torch.rand(36, 48, 3, dtype=torch.float64)

    def loss_fn():
        return ((render(means, scales, quats, colors, opacity, view) - target) ** 2).mean()

    loss_fn().backward()

    tensor = tensors[index]
    assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    assert tensor.grad.norm() > 0, f"{names[index]} received no gradient"

    analytic = tensor.grad.reshape(-1).clone()
    flat = tensor.detach().reshape(-1)
    eps = 1e-6
    checked = 0
    for i in torch.randperm(flat.numel())[:4]:
        original = flat[i].item()
        flat[i] = original + eps
        with torch.no_grad():
            plus = loss_fn().item()
        flat[i] = original - eps
        with torch.no_grad():
            minus = loss_fn().item()
        flat[i] = original

        numeric = (plus - minus) / (2 * eps)
        expected = analytic[i].item()
        scale = max(abs(numeric), abs(expected), 1e-8)
        assert abs(numeric - expected) / scale < 1e-4, (
            f"{names[index]}[{i}]: analytic {expected:.6e} vs numeric {numeric:.6e}"
        )
        checked += 1
    assert checked == 4


def test_a_gaussian_behind_the_camera_is_culled():
    means = torch.tensor([[0.0, 0.0, 5.0]], dtype=torch.float64)
    cov = build_covariance(torch.full((1, 3), 0.2, dtype=torch.float64),
                           torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64))
    view = torch.eye(4, dtype=torch.float64)
    # +Z is forward, so the point is behind the eye only once its camera-space
    # z goes negative: 5 - 6 = -1.
    view[2, 3] = -6.0
    image, info = rasterize(means, cov, torch.ones(1, 3, dtype=torch.float64),
                            torch.ones(1, 1, dtype=torch.float64), view,
                            (40.0, 40.0, 24.0, 18.0), 48, 36)
    assert not info["visible"].any()
    assert image.abs().max() == 0.0


def test_opaque_gaussian_occludes_what_is_behind_it():
    """Front-to-back compositing must let the nearer gaussian win."""
    means = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    scales = torch.full((2, 3), 0.30, dtype=torch.float64)
    quats = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]], dtype=torch.float64)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    opacity = torch.full((2, 1), 0.999, dtype=torch.float64)
    view = torch.eye(4, dtype=torch.float64)
    view[2, 3] = 3.0

    image = render(means, scales, quats, colors, opacity, view)
    centre = image[18, 24]
    assert centre[0] > 0.8, "the near red gaussian should dominate"
    assert centre[1] < 0.2, "the far green gaussian should be hidden"
