"""A synthetic textured scene with known camera poses.

Reconstruction tests need ground truth, and they need input a feature detector
can actually work with -- a flat-shaded cube produces almost no SIFT keypoints.
So this ray-casts a procedurally textured cube, which gives dense corner
features on every face while keeping the geometry and the poses exactly known.
"""

from __future__ import annotations

import numpy as np

HALF = 0.5          # cube half-extent, centred on the origin


def _value_noise(u, v, freq, seed):
    """Smoothly interpolated lattice noise, evaluated on arrays of uv."""
    rng = np.random.default_rng(seed)
    grid = rng.random((freq + 1, freq + 1))
    x = np.clip(u, 0, 1) * freq
    y = np.clip(v, 0, 1) * freq
    x0 = np.floor(x).astype(int).clip(0, freq - 1)
    y0 = np.floor(y).astype(int).clip(0, freq - 1)
    fx = x - x0
    fy = y - y0
    sx = fx * fx * (3 - 2 * fx)          # smoothstep, so gradients stay continuous
    sy = fy * fy * (3 - 2 * fy)
    a = grid[y0, x0]
    b = grid[y0, x0 + 1]
    c = grid[y0 + 1, x0]
    d = grid[y0 + 1, x0 + 1]
    return (a * (1 - sx) + b * sx) * (1 - sy) + (c * (1 - sx) + d * sx) * sy


def _speckles(u, v, seed, count=170, rmin=0.008, rmax=0.030):
    """High-contrast blobs scattered over uv.

    SIFT is a difference-of-gaussians blob detector, so smooth value noise gives
    it almost nothing to latch onto -- measured at ~90 keypoints per image,
    against the thousands a real photo yields, which left SfM unable to register
    the sequence. Discrete blobs at mixed radii are exactly the structure the
    detector is built to find, across several scales.
    """
    rng = np.random.default_rng(seed)
    cx = rng.random(count)
    cy = rng.random(count)
    radius = rng.uniform(rmin, rmax, count)
    strength = rng.uniform(-0.85, 0.85, count)

    out = np.zeros_like(u, dtype=np.float64)
    for i in range(count):
        d2 = (u - cx[i]) ** 2 + (v - cy[i]) ** 2
        r2 = radius[i] ** 2
        near = d2 < r2 * 4.0
        if near.any():
            # A soft-edged disc: sharp enough to detect, smooth enough to be
            # stable as the camera moves.
            out[near] += strength[i] * np.exp(-d2[near] / (2 * (radius[i] / 1.6) ** 2))
    return out


FACE_TINT = [
    (0.90, 0.35, 0.30), (0.30, 0.55, 0.90), (0.35, 0.80, 0.45),
    (0.92, 0.75, 0.25), (0.70, 0.40, 0.85), (0.25, 0.78, 0.78),
]


def face_texture(face, u, v):
    """Per-face colour: multi-octave noise, deliberately NON-repeating.

    A regular checker is the wrong test fixture -- every tile looks like every
    other tile, so SIFT matches become ambiguous and RANSAC throws them out.
    (Measured: a checkered cube registered 3 of 24 views.) Octave noise gives
    locally unique blobs, which is what a feature matcher can actually anchor to
    and what a real textured surface looks like.
    """
    tint = np.array(FACE_TINT[face])
    shade = np.zeros_like(u, dtype=np.float64)
    amplitude, total = 1.0, 0.0
    for octave, freq in enumerate((5, 11, 23, 47)):
        shade += amplitude * _value_noise(u, v, freq, seed=face * 101 + octave * 7 + 3)
        total += amplitude
        amplitude *= 0.55
    shade = 0.30 + 0.55 * (shade / total)
    shade = shade + _speckles(u, v, seed=face * 977 + 41, count=340)
    # Per-channel jitter keeps the colour from being a pure luminance ramp.
    wobble = 0.80 + 0.40 * _value_noise(v, u, 9, seed=face * 53 + 29)
    return np.clip(tint[None, :] * shade[:, None] * wobble[:, None], 0, 1)


GROUND_Y = -HALF          # the cube sits on the plane


def ground_texture(x, z):
    """Texture for the ground plane, on world coordinates."""
    u = np.clip((x + 3.0) / 6.0, 0, 1)
    v = np.clip((z + 3.0) / 6.0, 0, 1)
    shade = np.zeros_like(u, dtype=np.float64)
    amplitude, total = 1.0, 0.0
    for octave, freq in enumerate((7, 15, 31)):
        shade += amplitude * _value_noise(u, v, freq, seed=900 + octave * 13)
        total += amplitude
        amplitude *= 0.5
    shade = 0.30 + 0.55 * (shade / total)
    shade = shade + _speckles(u, v, seed=4242, count=700, rmin=0.004, rmax=0.014)
    base = np.array([0.55, 0.52, 0.48])
    return np.clip(base[None, :] * shade[:, None], 0, 1)


def look_at(centre, target=(0.0, 0.0, 0.0), world_up=(0.0, 1.0, 0.0)):
    """World-to-camera 4x4 in COLMAP convention: +Z forward, +Y down."""
    centre = np.asarray(centre, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - centre
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(world_up, dtype=np.float64))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = -R @ centre
    return pose


def render(pose, width, height, fx, fy, cx=None, cy=None, background=(0.06, 0.07, 0.09)):
    """Ray-cast the textured cube through a pinhole camera. Returns [H,W,3] float."""
    cx = width / 2 if cx is None else cx
    cy = height / 2 if cy is None else cy

    R = pose[:3, :3]
    t = pose[:3, 3]
    origin = -R.T @ t

    px, py = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    dirs_cam = np.stack([(px - cx) / fx, (py - cy) / fy, np.ones_like(px)], axis=-1)
    dirs = dirs_cam.reshape(-1, 3) @ R          # R^T applied on the right
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # Slab test against the axis-aligned cube.
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t0 = (-HALF - origin) * inv
        t1 = (HALF - origin) * inv
    tmin = np.minimum(t0, t1).max(axis=1)
    tmax = np.maximum(t0, t1).min(axis=1)
    hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 0)

    image = np.tile(np.asarray(background, dtype=np.float64), (dirs.shape[0], 1))

    # Ground plane: gives parallax at a different depth from the cube, which is
    # what lets SfM pin down scale and focal length.
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ground = (GROUND_Y - origin[1]) / dirs[:, 1]
    ground_hit = np.isfinite(t_ground) & (t_ground > 0)
    if ground_hit.any():
        gp = origin + dirs[ground_hit] * t_ground[ground_hit][:, None]
        inside = (np.abs(gp[:, 0]) < 3.0) & (np.abs(gp[:, 2]) < 3.0)
        idx = np.nonzero(ground_hit)[0][inside]
        if idx.size:
            image[idx] = ground_texture(gp[inside, 0], gp[inside, 2])
            # The cube occludes the plane wherever it is nearer.
            t_ground_full = np.full(dirs.shape[0], np.inf)
            t_ground_full[idx] = t_ground[idx]
            hit = hit & ((tmin < t_ground_full) | ~np.isfinite(t_ground_full))

    if hit.any():
        point = origin + dirs[hit] * tmin[hit][:, None]
        # The face is whichever axis the hit sits on.
        axis = np.argmax(np.abs(point), axis=1)
        sign = point[np.arange(point.shape[0]), axis] > 0
        face = axis * 2 + sign.astype(int)
        other = np.array([[1, 2], [0, 2], [0, 1]])[axis]
        u = (point[np.arange(point.shape[0]), other[:, 0]] + HALF) / (2 * HALF)
        v = (point[np.arange(point.shape[0]), other[:, 1]] + HALF) / (2 * HALF)

        colour = np.zeros((point.shape[0], 3))
        for f in range(6):
            m = face == f
            if m.any():
                colour[m] = face_texture(f, u[m], v[m])
        # A touch of Lambertian shading so faces are not flat-lit.
        normal = np.zeros_like(point)
        normal[np.arange(point.shape[0]), axis] = np.where(sign, 1.0, -1.0)
        lambert = 0.65 + 0.35 * np.clip(normal @ np.array([0.5, 0.8, -0.3]), 0, 1)
        image[hit] = np.clip(colour * lambert[:, None], 0, 1)

    return image.reshape(height, width, 3)


def orbit(n, radius=2.4, elevation_deg=18.0):
    """n camera poses evenly spaced around the cube."""
    poses = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        elev = np.radians(elevation_deg)
        centre = (
            np.sin(angle) * radius * np.cos(elev),
            np.sin(elev) * radius,
            np.cos(angle) * radius * np.cos(elev),
        )
        poses.append(look_at(centre))
    return poses


def make_dataset(out_dir, n=24, width=240, height=180, focal=210.0, radius=2.4):
    """Render an orbit to PNG and return the ground-truth poses."""
    from PIL import Image
    import os
    os.makedirs(out_dir, exist_ok=True)
    poses = orbit(n, radius=radius)
    for i, pose in enumerate(poses):
        img = render(pose, width, height, focal, focal)
        Image.fromarray((img * 255).astype(np.uint8)).save(f"{out_dir}/frame_{i:03d}.png")
    return poses, (width, height, focal)
