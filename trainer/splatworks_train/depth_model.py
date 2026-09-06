"""The monocular depth model, kept behind a small door.

This is the only part of the single-panorama path that needs a downloaded
model, so it is the only part that cannot be tested without one. Everything it
returns is plain numpy, which is what lets the geometry in depth.py be checked
against synthetic depth instead.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def load_estimator(model_name: str = DEFAULT_MODEL, device: str = "cpu"):
    """Build a depth estimator, or explain precisely what is missing.

    The two failures here look identical from the outside -- nothing happens --
    and have completely different fixes, so they are reported apart.
    """
    try:
        from transformers import pipeline
    except ImportError as exc:                                   # pragma: no cover
        raise RuntimeError(
            "The depth backend needs the transformers package: "
            "pip install transformers"
        ) from exc

    try:
        return pipeline("depth-estimation", model=model_name, device=resolve_device(device))
    except Exception as exc:                                     # pragma: no cover
        raise RuntimeError(
            f"Could not load the depth model {model_name!r}: {exc}. "
            "The first run downloads it, so this machine needs to reach "
            "huggingface.co once; afterwards it is cached."
        ) from exc


def resolve_device(requested: str) -> str:
    """The device to actually run on, as a string transformers accepts.

    A string rather than the device index the older API took: an index cannot
    express "cuda:1", so asking for the second GPU quietly ran on the first.

    Falling back to the CPU when CUDA was asked for and is not there is
    deliberate. This is a one-shot estimate over six images, so a slow answer
    beats no answer -- unlike training, where the same fallback would turn a
    two-minute run into an hour without saying why.
    """
    wanted = str(requested or "cpu").strip().lower()
    if not wanted.startswith("cuda"):
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return wanted
    except ImportError:                                          # pragma: no cover
        pass
    return "cpu"


def estimate(estimator, image) -> np.ndarray:
    """Predict relative inverse depth for one image.

    The value is disparity: large for near things, near zero for the sky. It
    carries no units and no shared scale between images, which is why the
    faces have to be aligned against each other afterwards.
    """
    from PIL import Image

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    result = estimator(image)
    # Recent versions return a tensor under "predicted_depth" and a PIL image
    # under "depth"; older ones only the image. Prefer the tensor: the image
    # has been quantised to eight bits, which visibly bands a large scene.
    predicted = result.get("predicted_depth")
    if predicted is not None:
        array = predicted.squeeze().detach().cpu().numpy().astype(np.float64)
    else:
        array = np.asarray(result["depth"], dtype=np.float64)

    if array.shape != (image.height, image.width):
        # Some checkpoints return depth at their own working resolution.
        from PIL import Image as PILImage
        scaled = PILImage.fromarray(array.astype(np.float32), mode="F")
        array = np.asarray(
            scaled.resize((image.width, image.height), PILImage.BILINEAR),
            dtype=np.float64)
    return array
