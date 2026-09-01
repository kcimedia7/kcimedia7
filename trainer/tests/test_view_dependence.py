"""Do the higher bands actually buy anything?

Note the step count these fits use. The bands train at a twentieth of the base
colour's rate, so they converge slowly: measured on the target below, degree 3
is *worse* than degree 0 at 400 steps and roughly ten times better by 2000. A
short run does not show the benefit, and a test that did not allow for it would
have concluded the bands were harmful.

The repository's synthetic scene is Lambertian: its colour depends only on
where a surface point is, never on where it is seen from. Comparing degree 0
against degree 3 on that scene measures nothing about view dependence -- there
is none to fit, and any difference would be the higher bands overfitting the
training views.

So this fits a target that genuinely varies with direction, which is the only
thing degree 3 can do that degree 0 cannot.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splatworks_train.model import GaussianModel                     # noqa: E402
from splatworks_train.sh import view_directions                      # noqa: E402


def specular_target(dirs, base, highlight):
    """A diffuse colour plus a lobe that peaks along one direction.

    This is the shape of a specular highlight: the same surface is brighter
    seen from some angles than others.
    """
    alignment = (dirs @ highlight).clamp(min=0.0).unsqueeze(1)
    return (base + 0.6 * alignment ** 4).clamp(0.0, 1.0)


def fit_colour(model, views, steps=2500):
    """Optimise colour alone against per-view targets, and report the error."""
    groups = [g for g in model.parameter_groups(0.0)
              if g["name"] in ("features_dc", "features_rest")]
    optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
    for _ in range(steps):
        loss = torch.zeros(())
        for dirs, target in views:
            loss = loss + (model.colors(dirs) - target).abs().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return float(sum((model.colors(d) - t).abs().mean() for d, t in views) / len(views))


def build(degree, n=32, seed=0):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n, 3)).astype(np.float32)
    model = GaussianModel.from_points(
        points, np.full((n, 3), 0.5, dtype=np.float32), sh_degree=degree)
    while model.raise_sh_degree():
        pass
    return model, torch.tensor(points)


def make_views(means, count=8, seed=1):
    """Cameras on a circle, each with the direction-dependent target it sees."""
    torch.manual_seed(seed)
    highlight = torch.nn.functional.normalize(torch.tensor([0.3, 0.9, 0.2]), dim=0)
    base = torch.full((means.shape[0], 3), 0.25)
    views = []
    for i in range(count):
        angle = 2 * np.pi * i / count
        centre = torch.tensor([4.0 * np.cos(angle), 1.0, 4.0 * np.sin(angle)],
                              dtype=torch.float32)
        dirs = view_directions(means, centre)
        views.append((dirs, specular_target(dirs, base, highlight)))
    return views


def test_higher_bands_fit_view_dependent_colour_that_degree_zero_cannot():
    flat, means = build(0)
    views = make_views(means)
    flat_error = fit_colour(flat, views)

    full, _ = build(3)
    full_error = fit_colour(full, views)

    # Degree 0 has one colour per gaussian for every direction, so the best it
    # can do is the average -- the highlight is error it cannot represent.
    assert full_error < flat_error * 0.4, (
        f"view-dependent bands should fit a specular target far better: "
        f"degree 0 {flat_error:.5f}, degree 3 {full_error:.5f}")


def test_the_bands_cost_nothing_on_a_target_that_does_not_vary_with_view():
    # The other half of the claim: on a Lambertian target -- which is what the
    # repository's synthetic scene is -- degree 3 must not do worse. If it did,
    # the extra parameters would be actively harmful on ordinary captures.
    flat, means = build(0)
    torch.manual_seed(2)
    base = torch.rand(means.shape[0], 3)
    views = []
    for i in range(12):
        angle = 2 * np.pi * i / 12
        centre = torch.tensor([4.0 * np.cos(angle), 1.0, 4.0 * np.sin(angle)],
                              dtype=torch.float32)
        views.append((view_directions(means, centre), base.clone()))

    flat_error = fit_colour(flat, views)
    full, _ = build(3)
    full_error = fit_colour(full, views)
    assert full_error <= flat_error * 1.1, (
        f"degree 3 should not be worse on a flat target: "
        f"degree 0 {flat_error:.4f}, degree 3 {full_error:.4f}")


def test_the_band_schedule_leaves_time_for_them_to_converge():
    """Switching a band on near the end is worse than not switching it on.

    The bands train at a twentieth of the base rate, so a band that arrives
    with a few hundred iterations left is still mid-flight when the run stops --
    and an unconverged band fits worse than no band at all.
    """
    from splatworks_train.train import sh_band_interval

    for total in (1000, 3000, 9000, 30_000):
        interval = sh_band_interval(total, 3)
        settling = total - interval * 3
        assert settling >= 0.7 * total, (
            f"{total} iterations leaves only {settling} after the last band")

    # A paper-length run should use the paper's own interval.
    assert sh_band_interval(30_000, 3) == 1000

    # Degree 0 asks for no bands, so the interval must never fire.
    assert sh_band_interval(3000, 0) >= 3000
