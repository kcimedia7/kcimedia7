"""Real spherical harmonics up to degree 3, as 3DGS uses them for colour.

Degree 0 is a single coefficient per channel: one colour, the same from every
direction. That is all this trainer used to store, and it is why surfaces came
out looking uniformly matte -- a window, a painted wall and a polished car all
render as their average colour with no sense of the light moving across them.

The higher bands add the view-dependent part. Evaluated against the direction
from the camera to the gaussian, they let a surface be brighter head-on than at
a glance, which is most of what reads as "real" in a reconstruction.

The coefficients are the standard real SH basis, matching the reference
implementation exactly so a model written here loads in any 3DGS viewer.
"""

from __future__ import annotations

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)

MAX_DEGREE = 3

#: Coefficients per channel at each degree: 1, 4, 9, 16.
def coefficient_count(degree: int) -> int:
    return (degree + 1) ** 2


def eval_sh(degree: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate SH coefficients along unit directions.

    Args:
        degree: bands to use, 0..3. Bands beyond it are ignored rather than
            dropped, so the model can grow into them during training.
        sh: (N, C, 3) coefficients, C >= (degree + 1) ** 2.
        dirs: (N, 3) unit vectors, camera towards gaussian.

    Returns:
        (N, 3) colour offsets, before the 0.5 mid-grey shift.
    """
    if degree < 0 or degree > MAX_DEGREE:
        raise ValueError(f"spherical harmonics degree must be 0..{MAX_DEGREE}, got {degree}")
    needed = coefficient_count(degree)
    if sh.shape[1] < needed:
        raise ValueError(f"degree {degree} needs {needed} coefficients, got {sh.shape[1]}")

    result = C0 * sh[:, 0]
    if degree < 1:
        return result

    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    result = result - C1 * y * sh[:, 1] + C1 * z * sh[:, 2] - C1 * x * sh[:, 3]
    if degree < 2:
        return result

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    result = (
        result
        + C2[0] * xy * sh[:, 4]
        + C2[1] * yz * sh[:, 5]
        + C2[2] * (2.0 * zz - xx - yy) * sh[:, 6]
        + C2[3] * xz * sh[:, 7]
        + C2[4] * (xx - yy) * sh[:, 8]
    )
    if degree < 3:
        return result

    return (
        result
        + C3[0] * y * (3.0 * xx - yy) * sh[:, 9]
        + C3[1] * xy * z * sh[:, 10]
        + C3[2] * y * (4.0 * zz - xx - yy) * sh[:, 11]
        + C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[:, 12]
        + C3[4] * x * (4.0 * zz - xx - yy) * sh[:, 13]
        + C3[5] * z * (xx - yy) * sh[:, 14]
        + C3[6] * x * (xx - 3.0 * yy) * sh[:, 15]
    )


def camera_centre(world_to_cam: torch.Tensor) -> torch.Tensor:
    """Where the camera sits in world space, from a world-to-camera matrix.

    The matrix maps world points into the camera's frame, so the camera's own
    position is the world point that lands at the origin: -R^T t.
    """
    rot = world_to_cam[:3, :3]
    translation = world_to_cam[:3, 3]
    return -(rot.transpose(0, 1) @ translation)


def view_directions(means: torch.Tensor, centre: torch.Tensor) -> torch.Tensor:
    """Unit vectors from the camera towards each gaussian."""
    direction = means - centre
    return direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
