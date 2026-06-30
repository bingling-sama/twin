"""CLIP model singleton — loaded once, shared across all requests."""

import logging
import os
from typing import Any

import torch
from PIL import Image

# Default OpenMP threads for Faiss CPU ops (k-means training, PQ).
# Override via env: OMP_NUM_THREADS=N.  Torch intra-op threads are
# controlled separately — CLIP runs on GPU so 1 is fine for CPU fallback.
os.environ.setdefault("OMP_NUM_THREADS", "4")
torch.set_num_threads(1)

logger = logging.getLogger(__name__)

_model: Any = None
_preprocess: Any = None
_device: str = ""


def _get_device() -> str:
    # Use CUDA if available (NVIDIA GPU)
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon GPU
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(
    device: str | None = None,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
) -> None:
    """Load CLIP model into memory. Called once at app startup.

    Args:
        device: Override device selection ('cuda', 'mps', 'cpu').
                If None, auto-detects: CUDA > MPS > CPU.
        model_name: CLIP variant to load (default: ViT-B-32).
        pretrained: Pretrained weights tag (default: openai).
    """
    global _model, _preprocess, _device, _model_name, _pretrained

    if _model is not None:
        logger.info("CLIP model already loaded, skipping")
        return

    import open_clip

    _model_name = model_name
    _pretrained = pretrained
    _device = device or _get_device()
    logger.info("Loading CLIP %s (%s) on %s ...", model_name, pretrained, _device.upper())
    _model, _, _preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    _model = _model.to(_device)
    _model.eval()
    logger.info("CLIP %s loaded successfully on %s", model_name, _device.upper())


_model_name: str = ""
_pretrained: str = ""


def is_loaded() -> bool:
    return _model is not None


def get_device() -> str:
    """Return the device CLIP is running on: 'cuda', 'mps', or 'cpu'."""
    return _device


def get_model_name() -> str:
    """Return the CLIP variant name (e.g. 'ViT-B-32')."""
    return _model_name


def get_gpu_name() -> str:
    """Return the GPU device name if using CUDA, empty string otherwise."""
    if _device == "cuda" and torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            pass
    return ""


def encode_image(image: Image.Image) -> "torch.Tensor":
    """Extract CLIP image embedding. Returns (1, 512) tensor on the correct device."""
    if _model is None:
        raise RuntimeError("CLIP model not loaded. Call load() first.")

    image_tensor = _preprocess(image).unsqueeze(0).to(_device)

    with torch.inference_mode():
        features = _model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    return features


def encode_images(images: list[Image.Image]) -> "torch.Tensor":
    """Batch-encode multiple images. Returns (N, 512) tensor on the correct device.

    Much faster than calling encode_image() N times because the CLIP forward
    pass processes the whole batch at once.
    """
    if _model is None:
        raise RuntimeError("CLIP model not loaded. Call load() first.")
    if not images:
        raise ValueError("Empty image list")

    tensors = torch.stack([_preprocess(img) for img in images]).to(_device)

    with torch.inference_mode():
        features = _model.encode_image(tensors)
        features = features / features.norm(dim=-1, keepdim=True)

    return features
