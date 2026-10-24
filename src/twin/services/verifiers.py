"""Pluggable Image Verifiers for Minecraft Textures (Phase 3).

Provides:
- BaseVerifier: Abstract contract for verification stages.
- DiffMaskVerifier: Ultra-fast (0.12ms) union-mask and RGB tolerance verifier.
- SSIMVerifier: Structural similarity verifier for backward-compatibility.
- get_verifier: Factory resolver based on runtime config.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from twin.core.config import settings
from twin.services.classifier import inspect_texture
from twin.services.converter import single2double
from twin.services.hasher import compute_ssim
from twin.utils.image import load_image


@dataclass
class VerifyResult:
    """Unified outcome returned by any Verifier implementation."""

    passed: bool
    score: float  # Normalized similarity score [0.0, 1.0], higher = more similar
    diff_ratio: float = 0.0  # Fraction of differing pixels [0.0, 1.0]
    details: dict = field(default_factory=dict)


class BaseVerifier(ABC):
    """Abstract base class defining the texture verification interface."""

    @abstractmethod
    def verify(self, query_img: Image.Image, cand_img: Image.Image) -> VerifyResult:
        """Verify similarity between two in-memory PIL images."""
        ...

    def verify_path(self, query_img: Image.Image, cand_path: str) -> VerifyResult:
        """Helper to load candidate image from disk and verify against query."""
        if not cand_path or not Path(cand_path).exists():
            return VerifyResult(passed=False, score=0.0, diff_ratio=1.0)
        cand_img = load_image(cand_path)
        if cand_img is None:
            return VerifyResult(passed=False, score=0.0, diff_ratio=1.0)
        return self.verify(query_img, cand_img)


class DiffMaskVerifier(BaseVerifier):
    """Adaptive Union-Mask Differencing Verifier.

    Adapted from Eidolon's verifier.rs.
    Computes an alpha union-mask ((Alpha_A > 10) | (Alpha_B > 10)),
    evaluates per-pixel RGB delta against a tolerance threshold,
    and calculates diff_ratio = differing_pixels / valid_mask_pixels.

    Executes in ~0.10ms on 64x64 textures via vectorized NumPy operations.
    """

    def __init__(
        self,
        tolerance: int | None = None,
        threshold: float | None = None,
        alpha_threshold: int = 10,
    ) -> None:
        self.tolerance = (
            tolerance if tolerance is not None else settings.diffmask_tolerance
        )
        self.threshold = (
            threshold if threshold is not None else settings.diffmask_threshold
        )
        self.alpha_threshold = alpha_threshold

    def verify(self, query_img: Image.Image, cand_img: Image.Image) -> VerifyResult:
        # Pre-process texture upgrade if needed
        q_rep = inspect_texture(query_img)
        if q_rep.can_auto_upgrade:
            query_img = single2double(query_img)

        c_rep = inspect_texture(cand_img)
        if c_rep.can_auto_upgrade:
            cand_img = single2double(cand_img)

        # Ensure RGBA and matching dimensions
        q_rgba = query_img.convert("RGBA")
        c_rgba = cand_img.convert("RGBA")
        if q_rgba.size != c_rgba.size:
            # Dimension mismatch (e.g., cape vs skin without proper isolation)
            return VerifyResult(
                passed=False,
                score=0.0,
                diff_ratio=1.0,
                details={"reason": "size_mismatch"},
            )

        arr1 = np.asarray(q_rgba, dtype=np.int16)
        arr2 = np.asarray(c_rgba, dtype=np.int16)

        mask1 = arr1[:, :, 3] > self.alpha_threshold
        mask2 = arr2[:, :, 3] > self.alpha_threshold
        union_mask = mask1 | mask2

        total_valid = int(np.count_nonzero(union_mask))
        if total_valid == 0:
            # Completely transparent image pair
            return VerifyResult(passed=True, score=1.0, diff_ratio=0.0)

        diff_r = np.abs(arr1[:, :, 0] - arr2[:, :, 0])
        diff_g = np.abs(arr1[:, :, 1] - arr2[:, :, 1])
        diff_b = np.abs(arr1[:, :, 2] - arr2[:, :, 2])

        # A pixel differs if any color channel exceeds tolerance OR transparency status differs
        pixel_diff = (
            (diff_r > self.tolerance)
            | (diff_g > self.tolerance)
            | (diff_b > self.tolerance)
            | (mask1 != mask2)
        )

        diff_pixels = int(np.count_nonzero(pixel_diff & union_mask))
        diff_ratio = diff_pixels / total_valid
        passed = diff_ratio <= self.threshold
        score = round(max(0.0, 1.0 - diff_ratio), 4)

        return VerifyResult(
            passed=passed,
            score=score,
            diff_ratio=round(diff_ratio, 4),
            details={
                "diff_pixels": diff_pixels,
                "total_valid": total_valid,
                "tolerance": self.tolerance,
            },
        )


class SSIMVerifier(BaseVerifier):
    """Structural Similarity Verifier for legacy baseline comparison."""

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = threshold if threshold is not None else settings.ssim_threshold

    def verify(self, query_img: Image.Image, cand_img: Image.Image) -> VerifyResult:
        q_rep = inspect_texture(query_img)
        if q_rep.can_auto_upgrade:
            query_img = single2double(query_img)

        c_rep = inspect_texture(cand_img)
        if c_rep.can_auto_upgrade:
            cand_img = single2double(cand_img)

        score = compute_ssim(query_img, cand_img)
        score_norm = round(float(score), 4)
        passed = score_norm >= self.threshold
        return VerifyResult(
            passed=passed,
            score=score_norm,
            diff_ratio=round(max(0.0, 1.0 - score_norm), 4),
            details={"ssim": score_norm},
        )


def get_verifier(
    verifier_type: Literal["diffmask", "ssim"] | str | None = None,
) -> BaseVerifier:
    """Factory creating the configured Verifier instance."""
    vtype = verifier_type or settings.verifier_type
    if vtype == "diffmask":
        return DiffMaskVerifier()
    elif vtype == "ssim":
        return SSIMVerifier()
    raise ValueError(f"Unknown verifier_type: '{vtype}'. Expected 'diffmask' or 'ssim'.")
