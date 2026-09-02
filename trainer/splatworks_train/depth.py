"""Turn one 360 photo into a gaussian cloud using predicted depth.

Structure-from-motion cannot touch a single panorama: every ray it recorded
passes through one point, so there is no baseline and nothing to triangulate.
A monocular depth model sidesteps that by *guessing* -- it has seen enough
photographs to have an opinion about what is near and what is far, and that
opinion is usually convincing.

What comes out is inferred, not measured. Distances are plausible rather than
correct, the overall scale is arbitrary, and anything hidden behind something
else simply has no data. That is the honest description of the result, and it
is a different kind of object from a reconstruction.

Everything here is pure geometry over numpy arrays. The model call lives in
depth_model.py so this half can be tested against synthetic depth without
downloading anything.
"""

from __future__ import annotations

import numpy as np

#: The six view directions, mirroring web/js/pano.js exactly.
#:
#: `forward` is the optical axis, `right` is +x in the image and `up` is -y,
#: because image rows count downward. A cross-check against the JavaScript table
#: lives in the tests: if these two ever disagree, the extracted views and the
#: geometry that unprojects them describe different scenes, and the result is a
#: cloud folded inside out with no error anywhere.
CUBE_FACES = (
    ("front", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ("right", (1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    ("back", (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ("left", (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ("up", (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    ("down", (0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
)

FACE_NAMES = tuple(face[0] for face in CUBE_FACES)


def face_basis(name: str):
    for face_name, forward, right, up in CUBE_FACES:
        if face_name == name:
            return np.array(forward), np.array(right), np.array(up)
    raise KeyError(f"unknown cube face {name!r}")


def ray_directions(name: str, size: int, fov_deg: float) -> np.ndarray:
    """Unit ray through every pixel of one view, in world space.

    Returns (size, size, 3). The rays all start at the panorama's viewpoint,
    which is the origin -- that is the whole problem with a single panorama and
    also what makes this unprojection simple.
    """
    forward, right, up = face_basis(name)
    t = np.tan(np.radians(fov_deg) / 2.0)
    # Pixel centres, matching the reprojection that produced these views.
    span = (2.0 * (np.arange(size) + 0.5) / size - 1.0) * t
    a = span[None, :, None]          # across the image
    b = -span[:, None, None]         # down the image, so world up is -y
    dirs = forward + a * right + b * up
    return dirs / np.linalg.norm(dirs, axis=2, keepdims=True)


def unproject(name: str, depth: np.ndarray, fov_deg: float) -> np.ndarray:
    """Place every pixel of a view in space at its predicted depth.

    Monocular models report depth along the optical axis, not along the ray, so
    a pixel at the edge of the frame is further away than its depth value says.
    Dividing by the cosine to the axis corrects that; skipping it bends every
    flat wall into a bowl.
    """
    size = depth.shape[0]
    dirs = ray_directions(name, size, fov_deg)
    forward, _, _ = face_basis(name)
    cos_axis = np.clip(dirs @ forward, 1e-6, None)
    return dirs * (depth[:, :, None] / cos_axis[:, :, None])


def disparity_to_depth(disparity: np.ndarray, near: float = 1.0,
                       far_ratio: float = 60.0) -> np.ndarray:
    """Convert a model's relative inverse depth into distances.

    Depth models of this family predict disparity: large for near things, near
    zero for the sky. Inverting it puts the sky at infinity, which no renderer
    wants, so the far end is capped at a multiple of the nearest distance. The
    result is a shape, not a measurement -- the absolute numbers mean nothing.
    """
    d = np.asarray(disparity, dtype=np.float64)
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        raise ValueError("depth estimate contains no usable values")
    # Anchor on a high percentile rather than the maximum: one speckled pixel
    # of extreme disparity would otherwise set the scale for the whole image.
    strongest = np.percentile(finite, 99.0)
    if strongest <= 0:
        raise ValueError("depth estimate has no positive disparity")
    floor = strongest / far_ratio
    return near * strongest / np.clip(d, floor, None)


def align_scales(faces, overlap_pairs, iterations: int = 200):
    """Put every face's relative depth on one common scale.

    Each face is estimated on its own, so each comes back with its own
    arbitrary scale and offset -- run separately, the same wall is at 3 units in
    one view and 11 in the next. Fusing them unaligned produces six mismatched
    shells with a visible seam at every edge.

    The views deliberately overlap, which is what makes this solvable: where two
    faces see the same direction they must agree, and a per-face scale that
    minimises the disagreement is a small least-squares problem in log space.
    Face 0 is pinned at 1 because only the ratios are determined -- the overall
    scale of a single panorama is unknowable either way.

    Args:
        faces: names, in the order their depths are given.
        overlap_pairs: (i, j, depth_i, depth_j) for directions seen by both,
            as arrays of matched samples.
    Returns:
        A scale per face, positive, with the first fixed at 1.
    """
    n = len(faces)
    log_scale = np.zeros(n)
    if not overlap_pairs:
        return np.ones(n)

    # Gauss-Seidel over the pairs: each face is nudged towards agreeing with
    # its neighbours. Small, well conditioned, and it needs no dependencies.
    for _ in range(iterations):
        numerator = np.zeros(n)
        weight = np.zeros(n)
        for i, j, di, dj in overlap_pairs:
            good = np.isfinite(di) & np.isfinite(dj) & (di > 0) & (dj > 0)
            if not np.any(good):
                continue
            # Median rather than mean: a handful of pixels straddling an object
            # edge disagree wildly and would drag a mean with them.
            ratio = np.median(np.log(dj[good]) - np.log(di[good]))
            count = int(np.count_nonzero(good))
            # Aligned depths must agree: s_i * d_i == s_j * d_j, so
            # log s_i == log s_j + (log d_j - log d_i).
            numerator[i] += count * (log_scale[j] + ratio)
            weight[i] += count
            numerator[j] += count * (log_scale[i] - ratio)
            weight[j] += count
        moved = np.where(weight > 0, numerator / np.maximum(weight, 1e-9), log_scale)
        log_scale = 0.5 * log_scale + 0.5 * moved
        log_scale -= log_scale[0]      # pin the first face

    return np.exp(log_scale - log_scale[0])


def overlap_samples(name_a: str, name_b: str, size: int, fov_deg: float,
                    depth_a: np.ndarray, depth_b: np.ndarray, stride: int = 4):
    """Distances the two views report for the directions they both cover.

    Walks the pixels of A, turns each into a world ray, and asks where that ray
    lands in B. Rays outside B's frustum are dropped, which is most of them --
    the overlap is a band along the shared edge.

    Both are converted to radial distance before being returned. Depth along
    the optical axis is not comparable between faces: the same point in the
    world sits at a different axis depth in each view because their axes point
    elsewhere, so aligning on it would fit the scales to the angle between the
    faces rather than to any disagreement about the scene.
    """
    dirs = ray_directions(name_a, size, fov_deg)[::stride, ::stride].reshape(-1, 3)
    forward_a, _, _ = face_basis(name_a)
    cos_a = np.clip(dirs @ forward_a, 1e-6, None)
    radial_a = depth_a[::stride, ::stride].reshape(-1) / cos_a

    forward, right, up = face_basis(name_b)
    z = dirs @ forward
    ahead = z > 1e-6
    if not np.any(ahead):
        return np.empty(0), np.empty(0)

    t = np.tan(np.radians(fov_deg) / 2.0)
    x = np.full(z.shape, np.nan)
    y = np.full(z.shape, np.nan)
    x[ahead] = (dirs[ahead] @ right) / z[ahead]
    y[ahead] = (dirs[ahead] @ up) / z[ahead]
    inside = ahead & (np.abs(x) <= t) & (np.abs(y) <= t)
    if not np.any(inside):
        return np.empty(0), np.empty(0)

    col = ((x[inside] / t + 1.0) * 0.5 * size - 0.5).round().astype(int)
    row = ((-y[inside] / t + 1.0) * 0.5 * size - 0.5).round().astype(int)
    np.clip(col, 0, size - 1, out=col)
    np.clip(row, 0, size - 1, out=row)
    radial_b = depth_b[row, col] / z[inside]
    return radial_a[inside], radial_b


def build_cloud(images, depths, fov_deg: float, max_points: int = 600_000,
                near: float = 1.0, far_ratio: float = 60.0, seed: int = 0):
    """Fuse per-face colour and depth into one gaussian cloud.

    Returns (positions, colours, scales) with one row per gaussian.
    """
    names = [name for name, _ in images]
    size = images[0][1].shape[0]

    metric = [disparity_to_depth(d, near=near, far_ratio=far_ratio) for d in depths]

    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = overlap_samples(names[i], names[j], size, fov_deg, metric[i], metric[j])
            if a.size:
                pairs.append((i, j, a, b))
    scales = align_scales(names, pairs)

    positions, colours, radii = [], [], []
    # Angular size of one pixel, which is how large a gaussian has to be at a
    # given distance to leave no gaps between it and its neighbours.
    pixel_angle = np.radians(fov_deg) / size
    for index, (name, rgb) in enumerate(images):
        depth = metric[index] * scales[index]
        points = unproject(name, depth, fov_deg).reshape(-1, 3)
        positions.append(points)
        colours.append(rgb.reshape(-1, 3).astype(np.float32) / 255.0)
        # Sized from radial distance, not from depth along the axis: a pixel at
        # the corner of a view is further away than its depth value says, and
        # sizing it from that leaves gaps between gaussians towards the edges
        # of every face -- a grid of seams over the whole scene.
        radial = np.linalg.norm(points, axis=1)
        radii.append((radial * pixel_angle).astype(np.float32))

    positions = np.concatenate(positions).astype(np.float32)
    colours = np.concatenate(colours).astype(np.float32)
    radii = np.concatenate(radii).astype(np.float32)

    keep = np.isfinite(positions).all(axis=1) & np.isfinite(radii) & (radii > 0)
    positions, colours, radii = positions[keep], colours[keep], radii[keep]

    if max_points and positions.shape[0] > max_points:
        pick = np.random.default_rng(seed).choice(
            positions.shape[0], max_points, replace=False)
        positions, colours, radii = positions[pick], colours[pick], radii[pick]

    return positions, colours, radii
