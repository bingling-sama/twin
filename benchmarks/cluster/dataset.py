"""Benchmark dataset generators and scenario builders for seedless texture discovery.

Provides ground-truth labeled clusters of Minecraft skins across:
1. Exact Duplicate Swarms (tests adaptive recall expansion and Top-K penetration).
2. Multi-cluster Derivative Variants (tests recolors, edits, and DiffMask verification).
3. Sparse Real-World Haystack (tests discovery amidst real community skins from images.zip).
"""

from __future__ import annotations

import io
import logging
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from twin.services.classifier import TextureKind, inspect_texture

logger = logging.getLogger(__name__)

ZIP_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "images.zip"


@dataclass
class BenchmarkItem:
    """A ground-truth labeled texture item for clustering benchmarks."""

    filename: str
    image: Image.Image
    cluster_id: str | None  # None indicates an isolated distractor/noise item
    datasource: str = "skin"
    texture_type: TextureKind = TextureKind.SKIN_DOUBLE


def create_base_skin(
    seed: int = 42, color_bias: tuple[int, int, int] = (180, 120, 70)
) -> Image.Image:
    """Generate a valid Minecraft 64x64 RGBA skin with distinct body regions."""
    rng = np.random.RandomState(seed)
    arr = np.zeros((64, 64, 4), dtype=np.uint8)

    # Base background (transparent outside UV bounds)
    r, g, b = color_bias

    # Head (8:16, 8:24)
    head_color = [
        np.clip(r + rng.randint(-20, 20), 0, 255),
        np.clip(g + rng.randint(-20, 20), 0, 255),
        np.clip(b + rng.randint(-20, 20), 0, 255),
        255,
    ]
    arr[8:16, 8:24] = head_color

    # Torso (20:32, 20:40)
    torso_color = [
        np.clip(r + rng.randint(-40, 40), 0, 255),
        np.clip(g + rng.randint(-40, 40), 0, 255),
        np.clip(b + rng.randint(-40, 40), 0, 255),
        255,
    ]
    arr[20:32, 20:40] = torso_color

    # Limbs (20:32, 44:56)
    arr[20:32, 44:56] = head_color

    # Lower body (32:48, 0:64)
    lower_color = [
        np.clip(r // 2 + rng.randint(0, 30), 0, 255),
        np.clip(g // 2 + rng.randint(0, 30), 0, 255),
        np.clip(b // 2 + rng.randint(0, 30), 0, 255),
        255,
    ]
    arr[32:48, 0:64] = lower_color

    # Set alpha channel strictly
    arr[0:48, 0:64, 3] = 255
    return Image.fromarray(arr, mode="RGBA")


def create_variant_skin(base_img: Image.Image, variant_idx: int) -> Image.Image:
    """Derive a controlled variant with diff_ratio within [0.08, 0.30] and Hamming <= 12."""
    arr = np.array(base_img.convert("RGBA")).copy()

    if variant_idx % 3 == 0:
        # Partial torso recolor (shirt recolor ~ 15% diff)
        arr[20:32, 20:36, :3] = (arr[20:32, 20:36, :3].astype(int) + 60) % 255
    elif variant_idx % 3 == 1:
        # Lower body hue shift (~ 22% diff)
        arr[32:44, :, :3] = (arr[32:44, :, :3].astype(int) + 120) % 255
    else:
        # Limb & helmet edit (~ 12% diff)
        arr[8:16, 16:24, :3] = 255 - arr[8:16, 16:24, :3]
        arr[20:32, 48:56, :3] = 255 - arr[20:32, 48:56, :3]

    return Image.fromarray(arr, mode="RGBA")


def build_duplicate_swarms_dataset(
    swarm_sizes: list[int] = [30, 20, 15],
    distractor_count: int = 40,
    seed: int = 100,
) -> tuple[list[BenchmarkItem], dict[str, int]]:
    """Build dataset dominated by large duplicate swarms.

    Stress tests Top-K recall starvation and adaptive recall expansion.
    """
    items: list[BenchmarkItem] = []
    ground_truth_counts: dict[str, int] = {}

    rng = random.Random(seed)

    # Generate each duplicate swarm
    for c_idx, swarm_size in enumerate(swarm_sizes):
        cluster_id = f"swarm_cluster_{c_idx + 1}"
        ground_truth_counts[cluster_id] = swarm_size
        base = create_base_skin(seed=seed + c_idx * 17)

        for item_idx in range(swarm_size):
            # Identical pixels with slight metadata variation
            img = base.copy()
            filename = f"c{c_idx + 1}_item_{item_idx:03d}.png"
            items.append(
                BenchmarkItem(
                    filename=filename,
                    image=img,
                    cluster_id=cluster_id,
                )
            )

    # Generate isolated distractors (random distinct skins)
    for d_idx in range(distractor_count):
        d_img = create_base_skin(seed=seed + 1000 + d_idx * 31)
        items.append(
            BenchmarkItem(
                filename=f"distractor_{d_idx:03d}.png",
                image=d_img,
                cluster_id=None,
            )
        )

    rng.shuffle(items)
    return items, ground_truth_counts


def build_variant_clusters_dataset(
    num_clusters: int = 5,
    items_per_cluster: int = 6,
    distractor_count: int = 40,
    seed: int = 200,
) -> tuple[list[BenchmarkItem], dict[str, int]]:
    """Build dataset of multiple clusters of derivative variants.

    Tests multi-cluster discrimination, diff_ratio bounding, and variant discovery.
    """
    items: list[BenchmarkItem] = []
    ground_truth_counts: dict[str, int] = {}
    rng = random.Random(seed)

    for c_idx in range(num_clusters):
        cluster_id = f"var_cluster_{c_idx + 1}"
        ground_truth_counts[cluster_id] = items_per_cluster
        base = create_base_skin(seed=seed + c_idx * 53)

        # First item is the base anchor
        items.append(
            BenchmarkItem(
                filename=f"v{c_idx + 1}_base.png",
                image=base.copy(),
                cluster_id=cluster_id,
            )
        )

        # Remaining items are controlled variants
        for v_idx in range(1, items_per_cluster):
            var_img = create_variant_skin(base, variant_idx=v_idx)
            items.append(
                BenchmarkItem(
                    filename=f"v{c_idx + 1}_var_{v_idx}.png",
                    image=var_img,
                    cluster_id=cluster_id,
                )
            )

    # Add distractors
    for d_idx in range(distractor_count):
        d_img = create_base_skin(seed=seed + 5000 + d_idx * 47)
        items.append(
            BenchmarkItem(
                filename=f"var_distractor_{d_idx:03d}.png",
                image=d_img,
                cluster_id=None,
            )
        )

    rng.shuffle(items)
    return items, ground_truth_counts


def build_sparse_real_haystack(
    background_size: int = 250,
    num_duplicate_pairs: int = 8,
    num_variant_clusters: int = 4,
    use_zip: bool = True,
    seed: int = 300,
) -> tuple[list[BenchmarkItem], dict[str, int]]:
    """Build dataset sampling real skins from images.zip with injected ground truth.

    Simulates real production background with sparse duplicates and variants.
    """
    items: list[BenchmarkItem] = []
    ground_truth_counts: dict[str, int] = {}
    rng = random.Random(seed)

    # 1. Sample background skins (from zip or synthetic fallback)
    sampled_skins: list[Image.Image] = []
    if use_zip and ZIP_PATH.is_file():
        try:
            with zipfile.ZipFile(ZIP_PATH, "r") as z:
                pngs = [
                    n
                    for n in z.namelist()
                    if n.lower().endswith(".png") and not n.startswith("__MACOSX")
                ]
                rng.shuffle(pngs)
                for name in pngs:
                    if (
                        len(sampled_skins)
                        >= background_size + num_duplicate_pairs + num_variant_clusters
                    ):
                        break
                    try:
                        raw = z.read(name)
                        img = Image.open(io.BytesIO(raw)).convert("RGBA")
                        if inspect_texture(img).texture_type == TextureKind.SKIN_DOUBLE:
                            sampled_skins.append(img)
                    except Exception:
                        continue
        except Exception as e:
            logger.warning("Could not read from %s: %s; using synthetic", ZIP_PATH, e)

    # Fill remainder with synthetic if zip was insufficient
    while len(sampled_skins) < background_size + num_duplicate_pairs + num_variant_clusters:
        idx = len(sampled_skins)
        sampled_skins.append(create_base_skin(seed=seed + idx * 79))

    # Split: background distractors
    bg_images = sampled_skins[:background_size]
    for b_idx, img in enumerate(bg_images):
        items.append(
            BenchmarkItem(
                filename=f"real_bg_{b_idx:04d}.png",
                image=img,
                cluster_id=None,
            )
        )

    cursor = background_size

    # Injected exact duplicate pairs (size 2~3)
    for p_idx in range(num_duplicate_pairs):
        cluster_id = f"real_dup_pair_{p_idx + 1}"
        pair_size = 2 if p_idx % 2 == 0 else 3
        ground_truth_counts[cluster_id] = pair_size
        anchor_img = sampled_skins[cursor]
        cursor += 1

        for m_idx in range(pair_size):
            items.append(
                BenchmarkItem(
                    filename=f"dup_{p_idx + 1}_m_{m_idx}.png",
                    image=anchor_img.copy(),
                    cluster_id=cluster_id,
                )
            )

    # Injected variant clusters (size 3~4)
    for v_idx in range(num_variant_clusters):
        cluster_id = f"real_var_cluster_{v_idx + 1}"
        v_size = 4
        ground_truth_counts[cluster_id] = v_size
        anchor_img = sampled_skins[cursor]
        cursor += 1

        items.append(
            BenchmarkItem(
                filename=f"varc_{v_idx + 1}_anchor.png",
                image=anchor_img.copy(),
                cluster_id=cluster_id,
            )
        )
        for sub_idx in range(1, v_size):
            v_img = create_variant_skin(anchor_img, variant_idx=sub_idx)
            items.append(
                BenchmarkItem(
                    filename=f"varc_{v_idx + 1}_member_{sub_idx}.png",
                    image=v_img,
                    cluster_id=cluster_id,
                )
            )

    rng.shuffle(items)
    return items, ground_truth_counts
