"""Embedding service — unified wrapper around CLIP and DINOv2 models."""

import logging
import time

import numpy as np
from PIL import Image

from twin.core.config import settings
from twin.models import clip_model, dinov2_model

logger = logging.getLogger(__name__)


def load_model(
    device: str | None = None,
    model_name: str | None = None,
    pretrained: str | None = None,
) -> None:
    """Unified model loader.

    Automatically routes to DINOv2 if model_name contains 'dinov2' or
    settings.model_type == 'dinov2', otherwise loads OpenCLIP.
    Dynamically sets settings.embedding_dim and settings.model_type.
    """
    target_name = model_name or settings.model_name
    target_device = device or (settings.device or None)
    target_pretrained = pretrained or settings.pretrained

    if "dinov2" in target_name.lower() or settings.model_type == "dinov2":
        logger.info("Routing to DINOv2 model: %s", target_name)
        dinov2_model.load(device=target_device, model_name=target_name)
        settings.model_type = "dinov2"
        settings.embedding_dim = dinov2_model.get_embedding_dim()
        logger.info(
            "Active model set to DINOv2 (%s, dim=%d)",
            target_name, settings.embedding_dim,
        )
    else:
        logger.info("Routing to OpenCLIP model: %s (%s)", target_name, target_pretrained)
        clip_model.load(device=target_device, model_name=target_name, pretrained=target_pretrained)
        settings.model_type = "clip"
        settings.embedding_dim = clip_model.get_embedding_dim()
        logger.info(
            "Active model set to CLIP (%s, dim=%d)",
            target_name, settings.embedding_dim,
        )


def is_loaded() -> bool:
    """Check if the active model singleton is loaded."""
    if settings.model_type == "dinov2":
        return dinov2_model.is_loaded()
    return clip_model.is_loaded()


def get_device() -> str:
    """Return device string ('cuda', 'mps', 'cpu') for active model."""
    if settings.model_type == "dinov2":
        return dinov2_model.get_device()
    return clip_model.get_device()


def get_model_name() -> str:
    """Return active model variant name."""
    if settings.model_type == "dinov2":
        return dinov2_model.get_model_name()
    return clip_model.get_model_name()


def get_gpu_name() -> str:
    """Return active model GPU name if on CUDA."""
    if settings.model_type == "dinov2":
        return dinov2_model.get_gpu_name()
    return clip_model.get_gpu_name()


def get_embedding_dim() -> int:
    """Return active embedding dimension."""
    if settings.model_type == "dinov2":
        return dinov2_model.get_embedding_dim()
    return clip_model.get_embedding_dim()


def is_text_supported() -> bool:
    """Return True if active model supports natural language text search."""
    return settings.model_type != "dinov2"


def compute_embedding(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a normalized embedding vector (dynamic dim).

    The image should already be RGB.
    """
    start = time.perf_counter()
    if settings.model_type == "dinov2":
        features = dinov2_model.encode_image(image)
    else:
        features = clip_model.encode_image(image)
    vector = features.squeeze(0).cpu().numpy().astype(np.float32)
    elapsed = time.perf_counter() - start
    logger.debug("Embedding computed in %.0fms", elapsed * 1000)
    return vector


def compute_embeddings(images: list[Image.Image]) -> np.ndarray:
    """
    Batch-convert multiple PIL Images to normalized embedding vectors.

    Much faster than calling compute_embedding() in a loop due to model batch inference.
    """
    if not images:
        dim = get_embedding_dim()
        return np.empty((0, dim), dtype=np.float32)

    start = time.perf_counter()
    if settings.model_type == "dinov2":
        features = dinov2_model.encode_images(images)
    else:
        features = clip_model.encode_images(images)
    vectors = features.cpu().numpy().astype(np.float32)
    elapsed = time.perf_counter() - start
    logger.debug(
        "Batch embedding: %d images in %.0fms (%.0fms/img)",
        len(images), elapsed * 1000, elapsed * 1000 / len(images),
    )
    return vectors


def compute_text_embedding(text: str) -> np.ndarray:
    """
    Convert a text prompt to a normalized embedding vector via CLIP.

    Raises ValueError if active model does not support text search (e.g. DINOv2).
    """
    if not is_text_supported():
        raise ValueError(
            f"Current model '{get_model_name() or settings.model_name}' (DINOv2) is vision-only "
            "and does not support natural language text search. Use CLIP (e.g. ViT-B-32) for text search."
        )

    start = time.perf_counter()
    features = clip_model.encode_text(text)
    vector = features.squeeze(0).cpu().numpy().astype(np.float32)
    elapsed = time.perf_counter() - start
    logger.debug("Text embedding computed in %.0fms", elapsed * 1000)
    return vector


def compute_text_embeddings(texts: list[str]) -> np.ndarray:
    """
    Batch-convert multiple text prompts to normalized embedding vectors via CLIP.

    Raises ValueError if active model does not support text search (e.g. DINOv2).
    """
    if not is_text_supported():
        raise ValueError(
            f"Current model '{get_model_name() or settings.model_name}' (DINOv2) is vision-only "
            "and does not support natural language text search. Use CLIP (e.g. ViT-B-32) for text search."
        )

    dim = get_embedding_dim()
    if not texts:
        return np.empty((0, dim), dtype=np.float32)

    start = time.perf_counter()
    features = clip_model.encode_texts(texts)
    vectors = features.cpu().numpy().astype(np.float32)
    elapsed = time.perf_counter() - start
    logger.debug(
        "Batch text embedding: %d texts in %.0fms",
        len(texts), elapsed * 1000,
    )
    return vectors
