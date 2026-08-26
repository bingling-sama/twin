"""DINOv2 model singleton — loaded on demand when TWIN_MODEL_TYPE=dinov2."""

import logging
from typing import Any

import torch
from PIL import Image

logger = logging.getLogger(__name__)

_model: Any = None
_preprocess: Any = None
_device: str = ""
_model_name: str = ""
_dim: int = 384


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(
    device: str | None = None,
    model_name: str = "vit_small_patch14_dinov2",
) -> None:
    """Load DINOv2 model into memory.

    Args:
        device: Override device selection ('cuda', 'mps', 'cpu').
        model_name: DINOv2 model variant from timm
            (e.g. 'vit_small_patch14_dinov2', 'vit_base_patch14_dinov2').
    """
    global _model, _preprocess, _device, _model_name, _dim

    if _model is not None:
        logger.info("DINOv2 model already loaded, skipping")
        return

    import timm

    _model_name = model_name
    _device = device or _get_device()
    logger.info("Loading DINOv2 %s on %s ...", model_name, _device.upper())

    _model = timm.create_model(model_name, pretrained=True, num_classes=0)
    _model = _model.to(_device)
    _model.eval()

    data_config = timm.data.resolve_model_data_config(_model)
    _preprocess = timm.data.create_transform(**data_config, is_training=False)

    _dim = getattr(_model, "num_features", 384)
    logger.info("DINOv2 %s (dim=%d) loaded successfully on %s", model_name, _dim, _device.upper())


def is_loaded() -> bool:
    return _model is not None


def get_device() -> str:
    return _device


def get_model_name() -> str:
    return _model_name


def get_embedding_dim() -> int:
    return _dim


def get_gpu_name() -> str:
    """Return the GPU device name if using CUDA, empty string otherwise."""
    if _device == "cuda" and torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            pass
    return ""


def encode_image(image: Image.Image) -> "torch.Tensor":
    """Extract DINOv2 image embedding. Returns (1, dim) normalized tensor on device."""
    if _model is None:
        raise RuntimeError("DINOv2 model not loaded. Call load() first.")

    image_tensor = _preprocess(image).unsqueeze(0).to(_device)
    with torch.inference_mode():
        features = _model(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features


def encode_images(images: list[Image.Image]) -> "torch.Tensor":
    """Batch-encode multiple images with DINOv2. Returns (N, dim) normalized tensor on device."""
    if _model is None:
        raise RuntimeError("DINOv2 model not loaded. Call load() first.")
    if not images:
        raise ValueError("Empty image list")

    tensors = torch.stack([_preprocess(img) for img in images]).to(_device)
    with torch.inference_mode():
        features = _model(tensors)
        features = features / features.norm(dim=-1, keepdim=True)
    return features
