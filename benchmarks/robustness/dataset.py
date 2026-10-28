"""Dataset sampler and evaluation query builder for Minecraft texture robustness benchmarks.

Streams real Minecraft textures from data/images.zip without extracting to disk.
Provides seamless fallback to test fixtures and synthetic textures when images.zip is absent.
"""

from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from benchmarks.robustness.perturbations import ALL_PERTURBATIONS
from twin.services.classifier import TextureKind, inspect_texture

logger = logging.getLogger(__name__)

ZIP_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "images.zip"
FIXTURES_PATH = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"


@dataclass(frozen=True)
class AnchorSample:
    """Ground truth texture sample in the reference index."""

    filename: str
    image: Image.Image
    datasource: str
    texture_type: TextureKind


@dataclass(frozen=True)
class PerturbedQuery:
    """Degraded query image mapped to its ground truth anchor."""

    query_id: str
    target_filename: str
    perturbation_type: str
    image: Image.Image
    datasource: str


def _create_synthetic_anchor(idx: int, seed: int = 42) -> AnchorSample:
    """Fallback generator for standard 64x64 double-layer skin."""
    import numpy as np

    rng = np.random.RandomState(seed + idx)
    arr = rng.randint(40, 220, (64, 64, 4), dtype=np.uint8)
    arr[:, :, 3] = 255  # fully opaque base
    img = Image.fromarray(arr, mode="RGBA")
    fname = f"synth_{idx:04d}.png"
    return AnchorSample(
        filename=fname,
        image=img,
        datasource="skin",
        texture_type=TextureKind.SKIN_DOUBLE,
    )


def sample_anchor_dataset(
    sample_size: int = 100,
    seed: int = 42,
    zip_path: Path | str = ZIP_PATH,
) -> list[AnchorSample]:
    """Sample valid Minecraft textures as anchor dataset.

    Streams images directly from ZIP without disk extraction.
    Falls back to fixtures / synthetic generation if ZIP does not exist.
    """
    import random

    anchors: list[AnchorSample] = []
    zip_file = Path(zip_path)

    if zip_file.is_file():
        try:
            with zipfile.ZipFile(zip_file, "r") as z:
                # Find all png files under images/ or root
                png_names = [
                    n
                    for n in z.namelist()
                    if n.lower().endswith(".png") and not n.startswith("__MACOSX")
                ]
                png_names.sort()

                # Deterministic random sample
                rng = random.Random(seed)
                shuffled = list(png_names)
                rng.shuffle(shuffled)

                for name in shuffled:
                    if len(anchors) >= sample_size:
                        break
                    try:
                        raw = z.read(name)
                        img = Image.open(io.BytesIO(raw))
                        report = inspect_texture(img)
                        if report.texture_type == TextureKind.UNKNOWN:
                            continue

                        ds = "cape" if report.texture_type == TextureKind.CAPE else "skin"
                        base_name = Path(name).name
                        anchors.append(
                            AnchorSample(
                                filename=base_name,
                                image=img.copy(),
                                datasource=ds,
                                texture_type=report.texture_type,
                            )
                        )
                    except Exception:
                        continue
        except Exception as e:
            logger.warning("Could not sample from %s: %s; falling back", zip_file, e)

    # Fallback to local fixtures and synthetic skins if needed
    if len(anchors) < sample_size:
        if FIXTURES_PATH.is_dir():
            for p in sorted(FIXTURES_PATH.glob("*.png")):
                if len(anchors) >= sample_size:
                    break
                try:
                    img = Image.open(p)
                    rep = inspect_texture(img)
                    if rep.texture_type != TextureKind.UNKNOWN:
                        anchors.append(
                            AnchorSample(
                                filename=p.name,
                                image=img.copy(),
                                datasource="skin",
                                texture_type=rep.texture_type,
                            )
                        )
                except Exception:
                    pass

    # Pad with synthetic textures to guarantee exact requested sample_size
    pad_idx = 0
    while len(anchors) < sample_size:
        anchors.append(_create_synthetic_anchor(pad_idx, seed=seed))
        pad_idx += 1

    return anchors


def build_evaluation_queries(
    anchors: list[AnchorSample],
    perturbations: dict[str, Callable[[Image.Image], Image.Image]] | None = None,
) -> list[PerturbedQuery]:
    """Generate degraded test queries with paired ground truth filenames."""
    ops = perturbations or ALL_PERTURBATIONS
    queries: list[PerturbedQuery] = []

    for anchor in anchors:
        for op_name, op_func in ops.items():
            try:
                perturbed_img = op_func(anchor.image.copy())
                query_id = f"{anchor.filename}_{op_name}"
                queries.append(
                    PerturbedQuery(
                        query_id=query_id,
                        target_filename=anchor.filename,
                        perturbation_type=op_name,
                        image=perturbed_img,
                        datasource=anchor.datasource,
                    )
                )
            except Exception as e:
                logger.warning(
                    "Failed to apply %s on %s: %s", op_name, anchor.filename, e, exc_info=True
                )

    return queries
