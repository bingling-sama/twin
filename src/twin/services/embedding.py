"""Embedding service — thin wrapper around CLIP model for vector extraction."""

import logging
import time

import numpy as np
from PIL import Image

from twin.core.config import settings
from twin.models import clip_model, dinov2_model

logger = logging.getLogger(__name__)


def compute_embedding(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a normalized embedding vector (512-dim for CLIP, or dim of DINOv2).

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
        dim = settings.embedding_dim
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
    Convert a text prompt to a normalized 512-dim embedding vector.

    Returns float32 ndarray of shape (512,).
    """
    start = time.perf_counter()
    features = clip_model.encode_text(text)
    vector = features.squeeze(0).cpu().numpy().astype(np.float32)
    elapsed = time.perf_counter() - start
    logger.debug("Text embedding computed in %.0fms", elapsed * 1000)
    return vector


def compute_text_embeddings(texts: list[str]) -> np.ndarray:
    """
    Batch-convert multiple text prompts to normalized 512-dim embedding vectors.

    Returns float32 ndarray of shape (N, 512).
    """
    if not texts:
        return np.empty((0, 512), dtype=np.float32)

    start = time.perf_counter()
    features = clip_model.encode_texts(texts)
    vectors = features.cpu().numpy().astype(np.float32)
    elapsed = time.perf_counter() - start
    logger.debug(
        "Batch text embedding: %d texts in %.0fms",
        len(texts), elapsed * 1000,
    )
    return vectors
