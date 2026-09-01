"""Spherical harmonics, checked against their defining properties.

Comparing an SH implementation against itself proves nothing: a transposed
band or a wrong constant produces a self-consistent but wrong renderer, and the
symptom is subtle colour shifts as the camera moves rather than an obvious
fault. These check orthonormality and rotational behaviour instead, which a
mistake in any coefficient breaks.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splatworks_train.sh import (                                    # noqa: E402
    MAX_DEGREE, C0, coefficient_count, eval_sh, camera_centre, view_directions,
)


def fibonacci_directions(n):
    """Near-uniform points on the sphere, for integrating over directions."""
    i = torch.arange(n, dtype=torch.float64) + 0.5
    phi = torch.acos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    return torch.stack([
        torch.sin(phi) * torch.cos(theta),
        torch.sin(phi) * torch.sin(theta),
        torch.cos(phi),
    ], dim=1).float()


def basis_matrix(dirs, degree=MAX_DEGREE):
    """Evaluate every basis function separately at each direction."""
    n = dirs.shape[0]
    count = coefficient_count(degree)
    out = torch.zeros(n, count)
    for k in range(count):
        sh = torch.zeros(n, count, 3)
        sh[:, k, :] = 1.0
        # Every channel carries the same coefficient, so any channel is the
        # basis function's own value.
        out[:, k] = eval_sh(degree, sh, dirs)[:, 0]
    return out


def test_the_basis_is_orthonormal_over_the_sphere():
    # The defining property. A wrong constant on any band shows up on the
    # diagonal; a swapped pair shows up off it.
    dirs = fibonacci_directions(20000)
    basis = basis_matrix(dirs).double()
    # Monte-Carlo integral over the sphere: mean * 4pi.
    gram = (basis.T @ basis) / dirs.shape[0] * (4 * math.pi)
    identity = torch.eye(gram.shape[0], dtype=torch.float64)
    error = (gram - identity).abs().max()
    assert error < 0.02, f"basis is not orthonormal, worst entry off by {error:.4f}"


def test_the_constant_band_is_the_average_over_all_directions():
    # Degree 0 must be direction-independent, and its constant is what makes an
    # all-zero model mid grey rather than black.
    dirs = fibonacci_directions(512)
    sh = torch.zeros(512, 16, 3)
    sh[:, 0, :] = 1.0
    value = eval_sh(0, sh, dirs)
    assert torch.allclose(value, torch.full_like(value, C0), atol=1e-6)
    # And with the higher bands present but degree 0 requested, they are ignored.
    sh[:, 1:, :] = 5.0
    assert torch.allclose(eval_sh(0, sh, dirs), torch.full_like(value, C0), atol=1e-6)


def test_odd_bands_flip_sign_with_direction_and_even_bands_do_not():
    # Band parity: degree 1 and 3 are odd functions of direction, 0 and 2 even.
    # Getting a band into the wrong group mirrors the lighting front-to-back.
    dirs = fibonacci_directions(256)
    for degree, odd in [(1, True), (2, False), (3, True)]:
        count = coefficient_count(degree)
        first = coefficient_count(degree - 1)
        sh = torch.zeros(256, 16, 3)
        sh[:, first:count, :] = torch.randn(1, count - first, 1)
        forward = eval_sh(degree, sh, dirs) - eval_sh(degree - 1, sh, dirs)
        backward = eval_sh(degree, sh, -dirs) - eval_sh(degree - 1, sh, -dirs)
        if odd:
            assert torch.allclose(forward, -backward, atol=1e-5), f"degree {degree} is not odd"
        else:
            assert torch.allclose(forward, backward, atol=1e-5), f"degree {degree} is not even"


def test_colour_varies_with_view_only_once_higher_bands_are_active():
    from splatworks_train.model import GaussianModel
    points = np.random.default_rng(0).normal(size=(64, 3)).astype(np.float32)
    model = GaussianModel.from_points(points, np.full((64, 3), 0.5, dtype=np.float32))
    with torch.no_grad():
        model.features_rest.normal_(0, 0.5)

    front = torch.nn.functional.normalize(torch.randn(64, 3), dim=1)
    back = -front

    # Degree 0 is the whole point of "flat": the same colour from anywhere.
    assert torch.allclose(model.colors(front), model.colors(back))

    while model.raise_sh_degree():
        pass
    assert model.active_sh_degree == MAX_DEGREE
    # With the bands active the two sides must differ, or nothing was gained.
    spread = (model.colors(front) - model.colors(back)).abs().max()
    assert spread > 0.01, f"view made no difference to colour ({spread})"
    # And colour stays inside the range the renderer can store.
    for dirs in (front, back):
        c = model.colors(dirs).detach()
        assert float(c.min()) >= 0.0 and float(c.max()) <= 1.0


def test_a_degree_beyond_the_stored_bands_is_refused():
    # Silently reading past the coefficients would sample whatever memory
    # follows and produce colours that drift for no visible reason.
    dirs = fibonacci_directions(8)
    with pytest.raises(ValueError, match="needs"):
        eval_sh(3, torch.zeros(8, 4, 3), dirs)
    for bad in (-1, 4):
        with pytest.raises(ValueError, match="degree"):
            eval_sh(bad, torch.zeros(8, 16, 3), dirs)


def test_the_camera_centre_comes_back_out_of_its_own_matrix():
    # The bands are evaluated against this, so an error here rotates the
    # lighting rather than failing outright.
    angle = 0.7
    rot = torch.tensor([
        [math.cos(angle), 0.0, math.sin(angle)],
        [0.0, 1.0, 0.0],
        [-math.sin(angle), 0.0, math.cos(angle)],
    ])
    centre = torch.tensor([1.5, -2.0, 3.25])
    world_to_cam = torch.eye(4)
    world_to_cam[:3, :3] = rot
    world_to_cam[:3, 3] = -(rot @ centre)
    assert torch.allclose(camera_centre(world_to_cam), centre, atol=1e-5)

    # A point at the camera maps to the origin, which is what makes it the centre.
    homogeneous = torch.cat([centre, torch.ones(1)])
    assert torch.allclose((world_to_cam @ homogeneous)[:3], torch.zeros(3), atol=1e-5)


def test_view_directions_point_from_the_camera_at_the_gaussians():
    centre = torch.tensor([0.0, 0.0, -5.0])
    means = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -10.0], [3.0, 4.0, -5.0]])
    dirs = view_directions(means, centre)
    assert torch.allclose(dirs.norm(dim=1), torch.ones(3), atol=1e-6)
    assert torch.allclose(dirs[0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)
    assert torch.allclose(dirs[1], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)
    assert torch.allclose(dirs[2], torch.tensor([0.6, 0.8, 0.0]), atol=1e-6)

    # A gaussian sitting exactly on the camera must not divide by zero.
    degenerate = view_directions(centre.unsqueeze(0), centre)
    assert torch.isfinite(degenerate).all()


def test_the_written_ply_carries_every_band_where_readers_expect_it(tmp_path):
    """Band order in the file is not a detail.

    The reference writes the higher bands channel-major -- every coefficient
    for red, then green, then blue. Interleaving them instead produces a file
    that loads without complaint in any viewer and renders with the colours
    smeared across the bands, which looks like a subtle tint rather than a bug.
    """
    from splatworks_train.ply_io import write_ply, properties_for

    n = 7
    rng = np.random.default_rng(0)
    dc = rng.normal(size=(n, 1, 3)).astype(np.float32)
    rest = rng.normal(size=(n, 15, 3)).astype(np.float32)
    path = tmp_path / "model.ply"
    write_ply(path, np.zeros((n, 3), np.float32), dc, rest,
              np.zeros((n, 1), np.float32), np.zeros((n, 3), np.float32),
              np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)))

    raw = path.read_bytes()
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:header_end].decode("ascii")
    names = [line.split()[-1] for line in header.splitlines()
             if line.startswith("property float")]
    assert names == properties_for(15), "property order does not match the reference"
    assert "f_rest_44" in names and "f_rest_45" not in names

    values = np.frombuffer(raw[header_end:], dtype="<f4").reshape(n, len(names))
    at = {name: i for i, name in enumerate(names)}

    for i in range(n):
        for channel in range(3):
            assert values[i, at[f"f_dc_{channel}"]] == pytest.approx(dc[i, 0, channel])
        # Channel-major: red's fifteen coefficients occupy f_rest_0..14.
        for coefficient in range(15):
            for channel in range(3):
                index = channel * 15 + coefficient
                assert values[i, at[f"f_rest_{index}"]] == pytest.approx(
                    rest[i, coefficient, channel]), \
                    f"f_rest_{index} should be coefficient {coefficient} of channel {channel}"


def test_a_degree_zero_model_still_writes_a_file_without_higher_bands(tmp_path):
    # Asking for degree 0 must produce the file this trainer used to write, so
    # nothing downstream has to special-case it.
    from splatworks_train.ply_io import write_ply

    n = 4
    path = tmp_path / "flat.ply"
    write_ply(path, np.zeros((n, 3), np.float32),
              np.zeros((n, 1, 3), np.float32), np.zeros((n, 0, 3), np.float32),
              np.zeros((n, 1), np.float32), np.zeros((n, 3), np.float32),
              np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)))
    header = path.read_bytes().split(b"end_header")[0].decode("ascii")
    assert "f_dc_2" in header
    assert "f_rest" not in header
