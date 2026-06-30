"""Embedding service — thin wrapper around CLIP model for vector extraction."""

import logging
import time

import numpy as np
from PIL import Image

from twin.models.clip_model import encode_image, encode_images

logger = logging.getLogger(__name__)


def compute_embedding(image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to a normalized 512-dim embedding vector.

    The image should already be RGB. Returns float32 ndarray of shape (512,).
    """
    start = time.perf_counter()
    features = encode_image(image)
    vector = features.squeeze(0).cpu().numpy().astype(np.float32)
    elapsed = time.perf_counter() - start
    logger.debug("Embedding computed in %.0fms", elapsed * 1000)
    return vector


def compute_embeddings(images: list[Image.Image]) -> np.ndarray:
    """
    Batch-convert multiple PIL Images to normalized 512-dim embedding vectors.

    Much faster than calling compute_embedding() in a loop due to CLIP batch inference.
    Returns float32 ndarray of shape (N, 512).
    """
    if not images:
        return np.empty((0, 512), dtype=np.float32)

    start = time.perf_counter()
    features = encode_images(images)
    vectors = features.cpu().numpy().astype(np.float32)
    elapsed = time.perf_counter() - start
    logger.debug(
        "Batch embedding: %d images in %.0fms (%.0fms/img)",
        len(images), elapsed * 1000, elapsed * 1000 / len(images),
    )
    return vectors
