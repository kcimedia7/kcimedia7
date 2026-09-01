"""End-to-end checks: SfM accuracy against ground truth, then real training.

The synthetic scene has exactly known camera poses, so pose error is measurable
rather than eyeballed -- which is the only way to tell a working reconstruction
from one that merely produces output.
"""

import json
import math
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
    assert int(info["overflow_tiles"]) == 0, (
        f"per-tile cap truncated {int(info['overflow_tiles'])} tiles "
        f"(peak occupancy {int(info['max_occupancy'])}); the render is not what the model says"
    )


def test_mixed_image_sizes_fail_loudly_instead_of_dropping_frames(tmp_path):
    """A single shared camera makes COLMAP skip odd-sized images silently.

    It logs a warning and carries on, so the reconstruction quietly runs on a
    subset. Since 360 panoramas are resampled into square views, mixing them
    with ordinary photos would hit this every time -- and look like a weak
    capture rather than like dropped input.
    """
    import numpy as np
    import pycolmap
    from PIL import Image
    from splatworks_train.sfm import _check_nothing_was_dropped

    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        Image.fromarray(rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)).save(
            images / f"square_{i}.png")

    work = tmp_path / "work"
    work.mkdir()
    database = work / "db.db"
    pycolmap.extract_features(
        database_path=database, image_path=images,
        camera_mode=pycolmap.CameraMode.SINGLE,
    )
    # All one size: nothing was dropped, so the check must stay out of the way.
    _check_nothing_was_dropped(database, images)

    # Add an image of a different shape and the extractor starts skipping.
    Image.fromarray(rng.integers(0, 255, (72, 128, 3), dtype=np.uint8)).save(
        images / "wide_0.png")
    database.unlink()
    pycolmap.extract_features(
        database_path=database, image_path=images,
        camera_mode=pycolmap.CameraMode.SINGLE,
    )
    with pytest.raises(RuntimeError, match="same dimensions"):
        _check_nothing_was_dropped(database, images)


def _tiny_model(n=32, device="cpu"):
    from splatworks_train.model import GaussianModel
    points = np.random.default_rng(0).normal(size=(n, 3)).astype(np.float32)
    colors = np.full((n, 3), 0.5, dtype=np.float32)
    return GaussianModel.from_points(points, colors, device=device)


def test_a_non_finite_gradient_cannot_poison_the_optimiser():
    """Adam keeps running moments, so one NaN gradient ruins a parameter forever.

    This is why a diverged run ends with a few percent of gaussians broken
    rather than none or all of them: the damage is per-parameter and permanent
    from the step it first appears.
    """
    import torch
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameter_groups(1e-3), lr=0.0, eps=1e-15)

    # A gradient the way divergence delivers it: mostly fine, a few ruined.
    model.means.grad = torch.zeros_like(model.means)
    model.means.grad[0, 0] = float("nan")
    model.means.grad[1, 1] = float("inf")
    model.means.grad[2, 2] = 0.5

    bad = int(model.sanitise_gradients())
    assert bad == 2, f"expected both bad entries counted, got {bad}"

    optimizer.step()
    assert torch.isfinite(model.means).all(), "a suppressed gradient still reached the parameters"

    # And the state Adam carries forward must be clean, or the next step
    # reintroduces the NaN with no bad gradient in sight.
    for state in optimizer.state.values():
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                assert torch.isfinite(state[key]).all(), f"Adam's {key} was poisoned"

    # A second step with clean gradients must stay finite too.
    model.means.grad = torch.full_like(model.means, 0.01)
    model.sanitise_gradients()
    optimizer.step()
    assert torch.isfinite(model.means).all()


def test_scales_are_held_inside_a_range_the_rasterizer_can_integrate():
    """A collapsing scale is what starts the divergence.

    The covariance determinant vanishes, the rasterizer floors it to stay
    finite, and the gradient's 1/det**2 term then explodes. Bounding the scale
    stops the chain at its source.
    """
    import torch
    model = _tiny_model()
    with torch.no_grad():
        model.log_scales[0] = -80.0     # collapsed to nothing
        model.log_scales[1] = 40.0      # swallowing the scene
    extent = 4.0
    model.clamp_scales(extent)

    scales = model.scales()
    assert torch.isfinite(scales).all()
    smallest, largest = float(scales.min().detach()), float(scales.max().detach())
    assert smallest >= extent * 1e-6 * 0.999
    assert largest <= extent * 0.5 * 1.001

    # A healthy model must pass through untouched -- this is a guard rail, not
    # a change in behaviour.
    healthy = _tiny_model()
    before = healthy.log_scales.detach().clone()
    healthy.clamp_scales(extent)
    assert torch.equal(before, healthy.log_scales.detach())


def test_non_finite_parameters_are_detected():
    import torch
    model = _tiny_model()
    assert int(model.non_finite_count()) == 0
    with torch.no_grad():
        model.means[3, 1] = float("nan")
        model.log_scales[4, 2] = float("inf")
    assert int(model.non_finite_count()) == 2


def test_more_iterations_buy_more_densification():
    """Gaussian count is what decides whether a result looks like the photos.

    The interval used to be a fraction of the run, which pinned the number of
    density-control rounds at about a dozen however long you trained: asking
    for 9000 iterations rather than 3000 bought three times the wait and the
    same handful of gaussians.
    """
    from splatworks_train.train import densification_interval

    def rounds(total):
        span = int(0.60 * total) - max(50, int(0.10 * total))
        return span // densification_interval(total)

    counts = [rounds(t) for t in (1000, 3000, 9000, 30000)]
    assert counts == sorted(counts), f"rounds must rise with iterations, got {counts}"
    assert counts[-1] > counts[0] * 4, f"a 30x longer run should densify far more: {counts}"

    # The paper's interval over a paper-length run.
    assert densification_interval(30_000) == 100

    # And a short run must not come out worse than the schedule it replaced.
    for total in (500, 1000, 3000):
        old_interval = max(25, total // 25)
        assert densification_interval(total) <= old_interval, (
            f"{total} iterations would densify less often than before")


def test_oversized_gaussians_are_pruned():
    """Streaks across a scene are single gaussians grown far too large.

    They reduce the loss cheaply while describing no surface and hiding
    everything behind them, so the paper prunes by both world and screen
    extent. Allowing half the scene, as this once did, is what let them
    survive.
    """
    import torch
    from splatworks_train.model import GaussianModel
    # A dense cloud in a small volume, so ordinary gaussians sit well inside
    # the threshold and only the deliberately oversized ones fail it. Spreading
    # a handful of points across the whole extent instead would make every
    # gaussian oversized, and the model refuses to prune itself to nothing.
    points = np.random.default_rng(0).normal(size=(400, 3)).astype(np.float32) * 0.05
    model = GaussianModel.from_points(points, np.full((400, 3), 0.5, dtype=np.float32))
    optimizer = torch.optim.Adam(model.parameter_groups(1e-3), lr=0.0, eps=1e-15)
    extent = 4.0

    with torch.no_grad():
        # One gaussian a quarter of the scene across: under the old half-scene
        # threshold this survived.
        model.log_scales[0] = math.log(extent * 0.25)
        # And one that is small in the world but huge on screen.
        model.max_radius[1] = 500.0

    before = model.count
    _, pruned = model.densify_and_prune(optimizer, grad_threshold=1e9, extent=extent,
                                        max_screen_size=20.0)
    assert model.count == before - 2, f"expected both oversized gaussians pruned, count {model.count}"
    assert pruned == 2, f"the count reported must be what was removed, got {pruned}"

    scales = model.scales().detach()
    assert float(scales.max()) < 0.1 * extent, "a world-oversized gaussian survived"

    # And a model where everything fails the test is left alone rather than
    # emptied: that turns a poor reconstruction into no reconstruction.
    doomed = GaussianModel.from_points(
        np.random.default_rng(1).normal(size=(32, 3)).astype(np.float32),
        np.full((32, 3), 0.5, dtype=np.float32))
    opt2 = torch.optim.Adam(doomed.parameter_groups(1e-3), lr=0.0, eps=1e-15)
    _, none_pruned = doomed.densify_and_prune(opt2, grad_threshold=1e9, extent=0.001)
    assert doomed.count == 32, "the model pruned itself to nothing"
    assert none_pruned == 0, "reported a prune that did not happen"
