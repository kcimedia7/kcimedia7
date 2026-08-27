"""Structure from motion via pycolmap -- real COLMAP, no system install.

Produces exactly what the trainer needs: intrinsics, world-to-camera poses, and
the sparse point cloud that seeds the gaussians.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap


@dataclass
class CameraView:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_to_cam: np.ndarray      # 4x4, COLMAP convention (+Z forward)

    @property
    def intrinsics(self):
        return (self.fx, self.fy, self.cx, self.cy)

    @property
    def camera_centre(self) -> np.ndarray:
        R = self.world_to_cam[:3, :3]
        t = self.world_to_cam[:3, 3]
        return -R.T @ t


def _pose_matrix(image) -> np.ndarray:
    """world-to-camera 4x4 from a pycolmap Image, across API spellings."""
    rigid = image.cam_from_world
    if callable(rigid):
        rigid = rigid()
    try:
        matrix = np.asarray(rigid.matrix(), dtype=np.float64)
    except Exception:
        rotation = rigid.rotation
        R = np.asarray(rotation.matrix() if callable(getattr(rotation, "matrix", None))
                       else rotation.matrix, dtype=np.float64)
        matrix = np.concatenate([R, np.asarray(rigid.translation).reshape(3, 1)], axis=1)
    out = np.eye(4)
    out[:3, :4] = matrix[:3, :4]
    return out


def _pinhole(camera):
    """(fx, fy, cx, cy) for whichever model COLMAP chose."""
    params = np.asarray(camera.params, dtype=np.float64)
    model = str(camera.model).split(".")[-1].upper()
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = params[0], params[1], params[2]
        return f, f, cx, cy
    # PINHOLE, OPENCV and friends all lead with fx, fy, cx, cy.
    return params[0], params[1], params[2], params[3]


def run_sfm(image_dir, work_dir, matcher="sequential", log=print):
    """Solve poses for every image in `image_dir`.

    Returns (views, points_xyz, points_rgb). Raises RuntimeError when COLMAP
    cannot register the images -- usually too little overlap or texture.
    """
    image_dir = Path(image_dir)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    database = work_dir / "database.db"
    if database.exists():
        database.unlink()
    sparse_dir = work_dir / "sparse"
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True)

    log("sfm: extracting features")
    pycolmap.extract_features(
        database_path=database,
        image_path=image_dir,
        # One physical camera shot every frame, which constrains intrinsics far
        # better than solving a separate camera per image.
        camera_mode=pycolmap.CameraMode.SINGLE,
    )

    log(f"sfm: matching ({matcher})")
    if matcher == "exhaustive":
        pycolmap.match_exhaustive(database_path=database)
    else:
        pycolmap.match_sequential(database_path=database)

    log("sfm: incremental mapping")
    reconstructions = pycolmap.incremental_mapping(
        database_path=database, image_path=image_dir, output_path=sparse_dir,
    )
    if not reconstructions:
        raise RuntimeError(
            "COLMAP could not register any images. The frames need more overlap, "
            "more texture, or less motion blur."
        )

    best = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    log(f"sfm: registered {best.num_reg_images()} images, {best.num_points3D()} points")
    if best.num_reg_images() < 2:
        raise RuntimeError("COLMAP registered fewer than two images; poses are unusable.")

    views = []
    for image in best.images.values():
        camera = best.camera(image.camera_id)
        fx, fy, cx, cy = _pinhole(camera)
        views.append(CameraView(
            name=image.name, width=camera.width, height=camera.height,
            fx=fx, fy=fy, cx=cx, cy=cy, world_to_cam=_pose_matrix(image),
        ))
    views.sort(key=lambda v: v.name)

    xyz, rgb = [], []
    for point in best.points3D.values():
        xyz.append(point.xyz)
        rgb.append(point.color)
    points = np.asarray(xyz, dtype=np.float32)
    colors = np.asarray(rgb, dtype=np.float32) / 255.0

    if points.shape[0] < 16:
        raise RuntimeError(f"COLMAP produced only {points.shape[0]} points; too sparse to train.")

    return views, points, colors


def scene_extent(views) -> float:
    """Radius of the camera rig, the scale the paper ties learning rates to."""
    centres = np.stack([v.camera_centre for v in views])
    centre = centres.mean(axis=0)
    return float(np.linalg.norm(centres - centre, axis=1).max() * 1.1) or 1.0
