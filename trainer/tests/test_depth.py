"""Single-panorama depth geometry, checked against a scene whose answer is known.

The failure mode here is quiet. A sign error in the ray basis, or forgetting
that a model reports depth along the optical axis rather than along the ray,
produces a cloud that still looks like a scene from the original viewpoint --
it is only wrong once you move, which is the entire point of building it.

So these tests unproject synthetic depth from a shape whose geometry is known
exactly, and check the points land on it.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splatworks_train.depth import (                                  # noqa: E402
    CUBE_FACES, FACE_NAMES, align_scales, build_cloud, disparity_to_depth,
    face_basis, overlap_samples, ray_directions, unproject,
)

FOV = 100.0
SIZE = 48


def test_the_face_table_matches_the_one_the_browser_reprojects_with():
    """Two copies of the same table, in two languages.

    If they drift, the views are extracted with one geometry and unprojected
    with another. The cloud comes out folded inside out and nothing reports an
    error, so the check is mechanical rather than by eye.
    """
    source = (Path(__file__).resolve().parents[2] / "web" / "js" / "pano.js").read_text()
    block = re.search(r"export const CUBE_FACES = \[(.*?)\n\];", source, re.S)
    assert block, "could not find CUBE_FACES in pano.js"

    entries = re.findall(
        r"\{\s*name:\s*'(\w+)',\s*forward:\s*(\[[^\]]*\]),\s*right:\s*(\[[^\]]*\]),"
        r"\s*up:\s*(\[[^\]]*\])\s*\}",
        block.group(1))
    assert len(entries) == len(CUBE_FACES), f"found {len(entries)} faces in pano.js"

    for (name, forward, right, up), (py_name, py_f, py_r, py_u) in zip(entries, CUBE_FACES):
        assert name == py_name, f"face order differs: {name} vs {py_name}"
        assert json.loads(forward) == list(py_f), f"{name} forward differs"
        assert json.loads(right) == list(py_r), f"{name} right differs"
        assert json.loads(up) == list(py_u), f"{name} up differs"


def test_every_ray_is_a_unit_vector_and_the_centre_looks_along_the_axis():
    for name in FACE_NAMES:
        dirs = ray_directions(name, SIZE, FOV)
        lengths = np.linalg.norm(dirs, axis=2)
        assert np.allclose(lengths, 1.0, atol=1e-6), f"{name} rays are not unit length"
        forward, _, _ = face_basis(name)
        mid = SIZE // 2
        # The centre of an even-sized image sits half a pixel off the axis.
        assert dirs[mid, mid] @ forward > 0.999, f"{name} centre does not look forward"


def test_the_six_views_cover_every_direction():
    # A gap in coverage is a hole in the finished scene.
    dirs = np.concatenate([ray_directions(n, SIZE, FOV).reshape(-1, 3) for n in FACE_NAMES])
    probes = np.random.default_rng(0).normal(size=(2000, 3))
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    # Every probe direction should have some ray close to it.
    best = (probes @ dirs.T).max(axis=1)
    assert best.min() > 0.99, f"a direction is uncovered (best alignment {best.min():.3f})"


def test_a_sphere_of_constant_depth_unprojects_onto_a_sphere():
    """The check that catches the cosine.

    Constant depth *along the optical axis* is not a sphere -- it is a plane in
    front of each face. Constant radial distance is the sphere. If the cosine
    correction were missing, this would come back as a cube.
    """
    radius = 4.0
    for name in FACE_NAMES:
        dirs = ray_directions(name, SIZE, FOV)
        forward, _, _ = face_basis(name)
        # Depth along the axis for a point at constant radius.
        axis_depth = radius * (dirs @ forward)
        points = unproject(name, axis_depth, FOV)
        distances = np.linalg.norm(points, axis=2)
        assert np.allclose(distances, radius, atol=1e-4), (
            f"{name}: radii range {distances.min():.4f}..{distances.max():.4f}, "
            "which means the depth was treated as radial")


def test_a_flat_wall_stays_flat():
    # A plane at constant z in front of the front face must unproject to a
    # plane, not a bowl. This is the same cosine seen from the other side.
    distance = 3.0
    dirs = ray_directions("front", SIZE, FOV)
    axis_depth = np.full((SIZE, SIZE), distance)
    points = unproject("front", axis_depth, FOV)
    assert np.allclose(points[:, :, 2], distance, atol=1e-5), "the wall came out curved"


def test_disparity_becomes_depth_with_the_sky_pushed_back_but_finite():
    # Sky reads as near-zero disparity. Inverting it naively gives infinity,
    # which no renderer and no bounding box can cope with.
    disparity = np.array([[10.0, 5.0, 1.0, 1e-6, 0.0]])
    depth = disparity_to_depth(disparity, near=1.0, far_ratio=50.0)
    assert np.all(np.isfinite(depth)), "sky became infinite"
    assert depth[0, 0] < depth[0, 1] < depth[0, 2], "near things must be nearer"
    assert depth[0, 3] == pytest.approx(depth[0, 4]), "everything past the cap is the cap"
    # The far cap is measured from the anchor -- a high percentile of the
    # disparity, not its maximum, so one speckled pixel cannot set the scale.
    assert depth.max() == pytest.approx(1.0 * 50.0, rel=0.02)


def test_depth_that_says_nothing_is_refused():
    with pytest.raises(ValueError):
        disparity_to_depth(np.full((4, 4), np.nan))
    with pytest.raises(ValueError):
        disparity_to_depth(np.zeros((4, 4)))


def test_faces_estimated_on_different_scales_are_brought_together():
    """Each face is estimated alone, so each has its own arbitrary scale.

    Unaligned, the same wall sits at different distances in adjacent views and
    the fused scene has a seam at every edge.
    """
    truth = 5.0
    names = list(FACE_NAMES)
    # Every face sees the same sphere, but reports it scaled differently.
    applied = np.array([1.0, 2.5, 0.4, 1.7, 0.7, 3.1])
    depths = []
    for name, factor in zip(names, applied):
        dirs = ray_directions(name, SIZE, FOV)
        forward, _, _ = face_basis(name)
        depths.append(truth * (dirs @ forward) / factor)

    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = overlap_samples(names[i], names[j], SIZE, FOV, depths[i], depths[j])
            if a.size:
                pairs.append((i, j, a, b))
    assert pairs, "the views should overlap enough to align them"

    solved = align_scales(names, pairs)
    # Only ratios are recoverable, so compare against the truth up to a
    # constant -- pinned on the first face, which is what align_scales fixes.
    expected = applied / applied[0]
    recovered = solved / solved[0]
    error = np.abs(recovered / expected - 1.0).max()
    assert error < 0.05, f"scales off by up to {error:.1%}: {recovered} vs {expected}"


def test_alignment_leaves_matching_faces_alone_and_survives_no_overlap():
    names = list(FACE_NAMES)
    assert np.allclose(align_scales(names, []), 1.0), "no overlap must not invent scales"

    # Faces that already agree must not be moved.
    depths = []
    for name in names:
        dirs = ray_directions(name, SIZE, FOV)
        forward, _, _ = face_basis(name)
        depths.append(3.0 * (dirs @ forward))
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = overlap_samples(names[i], names[j], SIZE, FOV, depths[i], depths[j])
            if a.size:
                pairs.append((i, j, a, b))
    assert np.allclose(align_scales(names, pairs), 1.0, atol=0.02)


def test_a_whole_panorama_fuses_into_a_cloud_on_the_shape_it_came_from():
    # End to end over the pure half: a sphere in, a spherical shell out, with
    # colour carried through and gaussians sized to leave no gaps.
    radius = 6.0
    images, disparities = [], []
    for name in FACE_NAMES:
        dirs = ray_directions(name, SIZE, FOV)
        forward, _, _ = face_basis(name)
        axis_depth = radius * (dirs @ forward)
        disparities.append(1.0 / axis_depth)
        rgb = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        rgb[:, :, 0] = 200
        images.append((name, rgb))

    positions, colours, radii = build_cloud(images, disparities, FOV, max_points=0)
    assert positions.shape[0] == len(FACE_NAMES) * SIZE * SIZE
    assert np.isfinite(positions).all()

    distances = np.linalg.norm(positions, axis=1)
    # One scale for the whole cloud, so the shell has a constant radius even
    # though the absolute value is arbitrary.
    assert distances.std() / distances.mean() < 0.02, "the shell is not uniform"

    assert np.allclose(colours[:, 0], 200 / 255.0, atol=1e-3)
    assert np.allclose(colours[:, 1], 0.0)
    # On a shell of constant radius every gaussian is the same size, which is
    # itself the check that they are sized by radial distance rather than by
    # depth along each face's axis.
    assert radii.min() > 0
    assert radii.std() / radii.mean() < 0.02, "gaussian size varies across a uniform shell"


def test_gaussians_grow_with_distance_so_far_surfaces_have_no_gaps():
    # A scene with real depth variation: a wall much further away on one side.
    images, disparities = [], []
    for name in FACE_NAMES:
        dirs = ray_directions(name, SIZE, FOV)
        forward, _, _ = face_basis(name)
        cos_axis = dirs @ forward
        # Radial distance ramps from 2 to 20 across the scene.
        radial = 2.0 + 18.0 * (dirs[:, :, 0] * 0.5 + 0.5)
        disparities.append(1.0 / (radial * cos_axis))
        images.append((name, np.full((SIZE, SIZE, 3), 100, dtype=np.uint8)))

    positions, _, radii = build_cloud(images, disparities, FOV, max_points=0)
    distances = np.linalg.norm(positions, axis=1)
    correlation = np.corrcoef(distances, radii)[0, 1]
    assert correlation > 0.99, (
        f"gaussian size should track distance, correlation {correlation:.3f}")


def test_the_cloud_is_capped_without_biasing_where_the_points_come_from():
    images, disparities = [], []
    for name in FACE_NAMES:
        dirs = ray_directions(name, SIZE, FOV)
        forward, _, _ = face_basis(name)
        disparities.append(1.0 / (5.0 * (dirs @ forward)))
        images.append((name, np.full((SIZE, SIZE, 3), 128, dtype=np.uint8)))

    positions, colours, radii = build_cloud(images, disparities, FOV, max_points=1000)
    assert positions.shape[0] == 1000
    assert colours.shape[0] == 1000 and radii.shape[0] == 1000
    # A cap that took the first N would leave a cloud covering one face.
    assert np.abs(positions.mean(axis=0)).max() < 0.6, "the sample is lopsided"


def write_views(directory, namer):
    """Six tagged views, named by whichever scheme is being tested."""
    from PIL import Image
    for index, name in enumerate(FACE_NAMES):
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[:, :, 0] = index * 40          # a tag identifying which face this is
        Image.fromarray(rgb).save(directory / namer(index, name))


def test_views_stored_the_way_the_server_stores_them_are_read_correctly(tmp_path):
    """The naming this actually has to cope with.

    The server discards the uploaded filenames and writes frame_00001.png
    upwards, so nothing in the name says which direction a view looked. Reading
    them by name finds no match at all -- which is how this backend failed on
    every real upload while its test passed, because the test invented a naming
    scheme that suited it.
    """
    from splatworks_train.from_panorama import load_faces

    write_views(tmp_path, lambda index, name: f"frame_{index + 1:05d}.png")
    loaded = load_faces(tmp_path)
    assert [name for name, _ in loaded] == list(FACE_NAMES), "faces came back out of order"
    for index, (_, rgb) in enumerate(loaded):
        assert rgb[0, 0, 0] == index * 40, "a face was matched to the wrong image"


def test_a_filename_that_names_its_face_is_believed_over_its_position(tmp_path):
    # So the backend can be pointed at a directory by hand, in any order.
    from splatworks_train.from_panorama import load_faces

    reversed_names = list(reversed(FACE_NAMES))
    write_views(tmp_path, lambda index, name: f"{index:02d}_{reversed_names[index]}.png")
    loaded = load_faces(tmp_path)
    assert [name for name, _ in loaded] == list(FACE_NAMES)
    # 'front' is last in the file order here, so it must carry the last tag.
    front_tag = dict(loaded)["front"][0, 0, 0]
    assert front_tag == (len(FACE_NAMES) - 1) * 40, "the filename was ignored"


def test_the_wrong_number_of_views_is_refused_rather_than_guessed(tmp_path):
    from PIL import Image
    from splatworks_train.from_panorama import load_faces

    for index in range(4):
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / f"frame_{index:05d}.png")
    with pytest.raises(SystemExit, match="expected 6"):
        load_faces(tmp_path)


def test_frames_that_are_not_panorama_views_are_refused(tmp_path):
    # Six ordinary photographs would otherwise be read as a panorama and
    # unprojected into a scene that never existed.
    from PIL import Image
    from splatworks_train.from_panorama import load_faces

    for index in range(6):
        Image.fromarray(np.zeros((6, 10, 3), np.uint8)).save(tmp_path / f"frame_{index:05d}.png")
    with pytest.raises(SystemExit, match="not square"):
        load_faces(tmp_path)


def test_the_whole_panorama_path_runs_without_a_downloaded_model(tmp_path, monkeypatch):
    """Everything except the model itself, end to end.

    The model is the one piece that needs a download, so it is swapped for a
    known depth field. That leaves the parts that can actually be wrong --
    reading, unprojecting, aligning, writing -- under test.
    """
    from PIL import Image
    from splatworks_train import from_panorama

    images = tmp_path / "images"
    images.mkdir()
    for index, name in enumerate(FACE_NAMES):
        rgb = np.full((32, 32, 3), 60 + index * 20, dtype=np.uint8)
        # Named exactly as the server names them: no face in the filename.
        Image.fromarray(rgb).save(images / f"frame_{index + 1:05d}.png")

    # A sphere at radius 5, expressed the way a model would report it, and with
    # a different arbitrary scale per face so alignment has work to do.
    scale_by_call = iter([1.0, 2.0, 0.5, 1.5, 0.8, 1.2])

    def fake_estimate(_estimator, rgb):
        size = rgb.shape[0]
        name = FACE_NAMES[fake_estimate.calls]
        fake_estimate.calls += 1
        dirs = ray_directions(name, size, 100.0)
        forward, _, _ = face_basis(name)
        axis_depth = 5.0 * (dirs @ forward)
        return next(scale_by_call) / axis_depth
    fake_estimate.calls = 0

    # main() imports these inside the function, so patching the module they
    # come from is enough and no download happens.
    import splatworks_train.depth_model as depth_model
    monkeypatch.setattr(depth_model, "load_estimator", lambda *a, **k: object())
    monkeypatch.setattr(depth_model, "estimate", fake_estimate)

    out = tmp_path / "out"
    assert from_panorama.main([
        "--images", str(images), "--output", str(out), "--max-points", "0",
    ]) == 0

    report = json.loads((out / "report.json").read_text())
    assert report["gaussians"] == len(FACE_NAMES) * 32 * 32
    assert report["views"] == 6
    assert report["depth"] == "inferred", "the result must say it was not measured"

    # The written file must be a real 3DGS ply that the rest of the app reads.
    raw = (out / "point_cloud.ply").read_bytes()
    header = raw[:raw.index(b"end_header")].decode("ascii")
    assert "f_dc_0" in header and "opacity" in header and "scale_0" in header
    assert "f_rest" not in header, "a depth cloud has no view-dependent colour to store"

    names = [line.split()[-1] for line in header.splitlines()
             if line.startswith("property float")]
    values = np.frombuffer(raw[raw.index(b"end_header\n") + 11:],
                           dtype="<f4").reshape(-1, len(names))
    at = {name: i for i, name in enumerate(names)}
    xyz = values[:, [at["x"], at["y"], at["z"]]]
    assert np.isfinite(xyz).all()

    # The six faces were given different scales; if alignment worked they land
    # on one shell rather than six.
    distances = np.linalg.norm(xyz, axis=1)
    assert distances.std() / distances.mean() < 0.05, (
        f"the faces did not align: radii {distances.min():.2f}..{distances.max():.2f}")


def test_the_device_reaches_the_model_as_something_it_understands():
    """A device index cannot express "cuda:1".

    The older API took one, so asking for the second GPU quietly ran on the
    first. Passing the string through keeps that distinction.
    """
    from splatworks_train.depth_model import resolve_device

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("") == "cpu"
    assert resolve_device(None) == "cpu"
    assert resolve_device("CUDA") in ("cuda", "cpu")     # depends on the machine

    import torch
    if torch.cuda.is_available():
        assert resolve_device("cuda") == "cuda"
        assert resolve_device("cuda:1") == "cuda:1", "the device number must survive"
    else:
        # No CUDA: a one-shot estimate over six images should still run rather
        # than fail, unlike training where a silent fallback costs an hour.
        assert resolve_device("cuda") == "cpu"
        assert resolve_device("cuda:1") == "cpu"


def test_the_result_shape_this_code_expects_is_what_the_library_returns():
    """Pinned against the installed transformers, not against my memory of it.

    The pipeline returns both a raw tensor and an eight-bit image; reading the
    image instead would band a large scene visibly, so which key is preferred
    matters and should break loudly if the library changes it.
    """
    transformers = pytest.importorskip("transformers")
    import inspect
    from transformers.pipelines.depth_estimation import DepthEstimationPipeline

    source = inspect.getsource(DepthEstimationPipeline.postprocess)
    assert '"predicted_depth"' in source, "the raw tensor key is gone"
    assert '"depth"' in source, "the image key is gone"

    from transformers.pipelines import SUPPORTED_TASKS
    assert "depth-estimation" in SUPPORTED_TASKS


def test_thinning_the_cloud_grows_the_gaussians_to_match():
    """Keeping a fraction of a surface spreads the survivors out.

    Keep f of the points and they sit 1/sqrt(f) further apart. Gaussians left
    at their original size stop touching, and the scene renders as a field of
    separate dots with the background between them rather than as surfaces --
    which is exactly what it did at 32% of the points.
    """
    images, disparities = [], []
    size = 32
    for name in FACE_NAMES:
        dirs = ray_directions(name, size, FOV)
        forward, _, _ = face_basis(name)
        disparities.append(1.0 / (4.0 * (dirs @ forward)))
        images.append((name, np.full((size, size, 3), 120, dtype=np.uint8)))

    total = len(FACE_NAMES) * size * size
    _, _, full = build_cloud(images, disparities, FOV, max_points=0)
    assert full.shape[0] == total

    kept = total // 4
    _, _, thinned = build_cloud(images, disparities, FOV, max_points=kept)
    assert thinned.shape[0] == kept

    # A quarter of the points means twice the spacing, so twice the radius.
    ratio = float(np.median(thinned)) / float(np.median(full))
    assert abs(ratio - 2.0) < 0.05, (
        f"a quarter of the points should double the gaussian size, got {ratio:.2f}x")

    # A cloud small enough to keep whole must not be inflated.
    _, _, uncapped = build_cloud(images, disparities, FOV, max_points=total * 2)
    assert abs(float(np.median(uncapped)) - float(np.median(full))) < 1e-6
