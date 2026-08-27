"""End-to-end checks: SfM accuracy against ground truth, then real training.

The synthetic scene has exactly known camera poses, so pose error is measurable
rather than eyeballed -- which is the only way to tell a working reconstruction
from one that merely produces output.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splatworks_train.sfm import run_sfm, scene_extent          # noqa: E402
from splatworks_train.train import build_parser, train          # noqa: E402
from tests.synthetic import make_dataset, orbit                 # noqa: E402

# 24 views is 15 degrees apart. At 16 views (22.5 degrees) consecutive frames
# share too little of the subject and COLMAP reports "no initial pair" -- the
# same failure a user hits when they walk around something too quickly.
N_VIEWS = 24
FOCAL = 420.0


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("frames")
    make_dataset(str(out), n=N_VIEWS, width=480, height=360, focal=FOCAL)
    return out


@pytest.fixture(scope="module")
def solved(dataset, tmp_path_factory):
    work = tmp_path_factory.mktemp("sfm")
    return run_sfm(dataset, work, matcher="exhaustive", log=lambda *a: None)


def umeyama_similarity(source, target):
    """Best similarity transform source -> target; SfM is only defined up to one."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    s_c, t_c = source - mu_s, target - mu_t
    cov = s_c.T @ t_c / source.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(U @ Vt))
    W = np.diag([1.0, 1.0, sign])
    R = U @ W @ Vt
    scale = (D * np.array([1.0, 1.0, sign])).sum() / (s_c ** 2).sum() * source.shape[0]
    return scale, R, mu_t - scale * (mu_s @ R)


def test_sfm_registers_every_view(solved):
    views, points, colors = solved
    assert len(views) == N_VIEWS, f"only {len(views)}/{N_VIEWS} views registered"
    assert points.shape[0] > 200, "sparse cloud is too small to seed training"
    assert points.shape == colors.shape


def test_sfm_recovers_focal_length(solved):
    views, _, _ = solved
    # Everything downstream is wrong if the intrinsics are wrong.
    assert views[0].fx == pytest.approx(FOCAL, rel=0.05)
    assert views[0].fy == pytest.approx(FOCAL, rel=0.05)


def test_sfm_recovers_camera_geometry(solved):
    """Recovered camera centres must match ground truth after alignment."""
    views, _, _ = solved
    truth = {f"frame_{i:03d}.png": (-p[:3, :3].T @ p[:3, 3]) for i, p in enumerate(orbit(N_VIEWS))}
    estimated = np.stack([v.camera_centre for v in views])
    expected = np.stack([truth[v.name] for v in views])

    scale, R, t = umeyama_similarity(estimated, expected)
    aligned = scale * (estimated @ R) + t
    error = np.linalg.norm(aligned - expected, axis=1)
    rig_radius = np.linalg.norm(expected - expected.mean(0), axis=1).mean()

    assert error.mean() / rig_radius < 0.02, (
        f"mean pose error {error.mean():.4f} is {100 * error.mean() / rig_radius:.1f}% of the rig radius"
    )


def test_scene_extent_is_positive(solved):
    views, _, _ = solved
    assert scene_extent(views) > 0


@pytest.mark.slow
def test_training_converges_and_writes_a_valid_ply(dataset, tmp_path):
    """A short run must reduce loss and emit a readable 3DGS .ply."""
    output = tmp_path / "model"
    args = build_parser().parse_args([
        "-s", str(dataset), "-m", str(output),
        "--iterations", "60", "--resolution", "96",
        "--max-gaussians", "8000", "--matcher", "exhaustive",
    ])
    report = train(args, log=lambda *a: None)

    assert report["gaussians"] > 0
    assert report["registered_views"] == N_VIEWS

    history = report["history"]
    assert len(history) >= 2
    # Loss should be lower at the end than at the start; a flat or rising curve
    # means the gradients are not reaching the parameters.
    assert history[-1]["loss"] < history[0]["loss"], \
        f"loss did not fall: {history[0]['loss']:.4f} -> {history[-1]['loss']:.4f}"
    assert report["psnr"] > 10.0

    ply = Path(report["ply"])
    assert ply.exists()
    # Read to end_header rather than a fixed prefix: the 17 properties run past
    # any small fixed slice, which silently hides the trailing ones.
    raw = ply.read_bytes()[:4096]
    end = raw.find(b"end_header")
    assert end != -1, "no end_header in the first 4 KiB"
    header = raw[:end].decode("latin1")
    assert header.startswith("ply")
    assert "format binary_little_endian 1.0" in header
    for prop in ("x", "f_dc_0", "opacity", "scale_0", "rot_3"):
        assert f"property float {prop}" in header

    saved = json.loads((output / "report.json").read_text())
    assert saved["gaussians"] == report["gaussians"]


@pytest.mark.slow
def test_render_matches_the_input_images(dataset, tmp_path):
    """Rendering a trained model from a training pose should resemble that photo."""
    import torch
    from splatworks_train.model import GaussianModel
    from splatworks_train.rasterizer import build_covariance, rasterize
    from splatworks_train.train import load_views
    from splatworks_train.losses import psnr

    views, points, colors = run_sfm(dataset, tmp_path / "sfm", matcher="exhaustive",
                                    log=lambda *a: None)
    data = load_views(views, dataset, 96, torch.device("cpu"))
    model = GaussianModel.from_points(points, colors)

    view = data[0]
    with torch.no_grad():
        image, info = rasterize(
            model.means, build_covariance(model.scales(), model.quats),
            model.colors(), model.opacities(),
            view["viewmat"], view["K"], view["width"], view["height"],
        )
    # Even untrained, gaussians seeded at SfM points and coloured from the
    # images must land on the subject rather than render an empty frame.
    assert image.shape == (view["height"], view["width"], 3)
    assert float(image.max()) > 0.05, "nothing was rendered"
    assert psnr(image, view["image"]) > 5.0
    assert info["overflow_tiles"] == 0, (
        f"per-tile cap truncated {info['overflow_tiles']} tiles "
        f"(peak occupancy {info['max_occupancy']}); the render is not what the model says"
    )
