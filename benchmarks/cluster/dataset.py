"""Benchmark dataset generators and scenario builders for seedless texture discovery.

Provides ground-truth labeled clusters of Minecraft skins across:
1. Exact Duplicate Swarms (tests adaptive recall expansion and Top-K penetration).
2. Multi-cluster Derivative Variants (tests recolors, edits, and DiffMask verification).
3. Sparse Real-World Haystack (tests discovery amidst real community skins from images.zip).
4. Threshold Boundary Scenarios (tests strict boundary gating).
5. Transitivity Chain Scenarios (tests anti-transitivity collapse via greedy leader elimination).
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
from twin.services.hasher import compute_phash, hamming_distance
from twin.services.verifiers import DiffMaskVerifier

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
    """Derive a controlled, unique variant with diff_ratio in [0.08, 0.32] and Hamming <= 12."""
    arr = np.array(base_img.convert("RGBA")).copy()
    valid_coords = np.argwhere(arr[:, :, 3] > 10)
    num_valid = len(valid_coords)
    if num_valid == 0:
        return base_img.copy()

    # Target diff ratio in [0.12, 0.26]
    target_ratio = 0.12 + ((variant_idx * 3) % 15) * 0.01
    count = int(num_valid * target_ratio)

    stride = max(1, num_valid // count)
    start_offset = (variant_idx * 137) % num_valid
    indices = [(start_offset + i * stride) % num_valid for i in range(count)]
    selected_coords = valid_coords[indices]

    shift = 35 + ((variant_idx * 29) % 40)
    ch = variant_idx % 3

    for y, x in selected_coords:
        val = int(arr[y, x, ch]) + shift
        arr[y, x, ch] = val % 256

    return Image.fromarray(arr, mode="RGBA")


def verify_variant_contracts(
    base_img: Image.Image,
    variants: list[Image.Image],
    min_diff: float = 0.08,
    max_diff: float = 0.32,
    max_hamming: int = 12,
    min_pairwise_diff: float = 0.01,
) -> None:
    """Verify physical contracts of generated variants using verifier and pHash.

    Raises:
        AssertionError: If any variant violates DiffMask bounds, Hamming distance,
                        or pair-wise uniqueness.
    """
    verifier = DiffMaskVerifier()
    base_phash = compute_phash(base_img)

    for idx, var in enumerate(variants):
        diff = verifier.verify(base_img, var).diff_ratio
        h_dist = hamming_distance(base_phash, compute_phash(var))

        if not (min_diff <= diff <= max_diff):
            raise AssertionError(
                f"Variant {idx} violated diff_ratio contract: "
                f"{diff:.4f} not in [{min_diff}, {max_diff}]"
            )
        if h_dist > max_hamming:
            raise AssertionError(
                f"Variant {idx} violated Hamming contract: {h_dist} > {max_hamming}"
            )

    # Check pair-wise uniqueness
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            pair_diff = verifier.verify(variants[i], variants[j]).diff_ratio
            if pair_diff < min_pairwise_diff:
                raise AssertionError(
                    f"Variants {i} and {j} are identical or too close: diff_ratio={pair_diff:.4f}"
                )


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
        cluster_variants: list[Image.Image] = []
        for v_idx in range(1, items_per_cluster):
            var_img = create_variant_skin(base, variant_idx=v_idx)
            cluster_variants.append(var_img)
            items.append(
                BenchmarkItem(
                    filename=f"v{c_idx + 1}_var_{v_idx}.png",
                    image=var_img,
                    cluster_id=cluster_id,
                )
            )

        # Enforce physical dataset contracts at generation time
        verify_variant_contracts(base, cluster_variants)

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
        cluster_variants = []
        for sub_idx in range(1, v_size):
            v_img = create_variant_skin(anchor_img, variant_idx=sub_idx)
            cluster_variants.append(v_img)
            items.append(
                BenchmarkItem(
                    filename=f"varc_{v_idx + 1}_member_{sub_idx}.png",
                    image=v_img,
                    cluster_id=cluster_id,
                )
            )
        verify_variant_contracts(anchor_img, cluster_variants)

    rng.shuffle(items)
    return items, ground_truth_counts


def build_boundary_dataset(
    seed: int = 400,
) -> tuple[list[BenchmarkItem], dict[str, int]]:
    """Build dataset with near-boundary and beyond-boundary samples.

    Tests strict rejection when DiffMask ratio or Hamming distance exceeds thresholds.
    """
    items: list[BenchmarkItem] = []
    ground_truth_counts: dict[str, int] = {}
    base = create_base_skin(seed=seed)
    valid_coords = np.argwhere(np.array(base)[:, :, 3] > 10)
    num_valid = len(valid_coords)

    # Cluster 1: Confirmed duplicates boundary (diff ~ 0.045 < 0.05)
    cluster_conf = "boundary_confirmed_cluster"
    ground_truth_counts[cluster_conf] = 2
    items.append(
        BenchmarkItem(filename="conf_base.png", image=base.copy(), cluster_id=cluster_conf)
    )

    arr_conf_near = np.array(base).copy()
    count_near = int(num_valid * 0.045)
    for y, x in valid_coords[:count_near]:
        arr_conf_near[y, x, 0] = (int(arr_conf_near[y, x, 0]) + 40) % 256
    items.append(
        BenchmarkItem(
            filename="conf_member_near.png",
            image=Image.fromarray(arr_conf_near, mode="RGBA"),
            cluster_id=cluster_conf,
        )
    )

    # Distractor: Just beyond confirmed limit (diff ~ 0.065 > 0.05)
    # Uses disjoint coordinates [0.10 : 0.165] so it does not collapse with conf_member_near
    arr_conf_beyond = np.array(base).copy()
    start_beyond = int(num_valid * 0.10)
    end_beyond = int(num_valid * 0.165)
    for y, x in valid_coords[start_beyond:end_beyond]:
        arr_conf_beyond[y, x, 0] = (int(arr_conf_beyond[y, x, 0]) + 40) % 256
    items.append(
        BenchmarkItem(
            filename="conf_distractor_beyond.png",
            image=Image.fromarray(arr_conf_beyond, mode="RGBA"),
            cluster_id=None,
        )
    )

    # Distractor: Just beyond variants limit (diff ~ 0.380 > 0.35)
    arr_var_beyond = np.array(base).copy()
    start_var = int(num_valid * 0.30)
    end_var = int(num_valid * 0.68)
    for y, x in valid_coords[start_var:end_var]:
        arr_var_beyond[y, x, 1] = (int(arr_var_beyond[y, x, 1]) + 60) % 256
    items.append(
        BenchmarkItem(
            filename="var_distractor_beyond.png",
            image=Image.fromarray(arr_var_beyond, mode="RGBA"),
            cluster_id=None,
        )
    )

    return items, ground_truth_counts


def build_transitivity_chain_dataset(
    seed: int = 500,
) -> tuple[list[BenchmarkItem], dict[str, int]]:
    """Build transitivity chain dataset: A ~ B (0.18), B ~ C (0.20), A !~ C (0.50).

    Tests that Greedy Leader Dynamic Elimination clusters A with B and eliminates B,
    preventing non-transitive collapse where C gets improperly merged into A.
    """
    items: list[BenchmarkItem] = []
    ground_truth_counts: dict[str, int] = {}
    base_a = create_base_skin(seed=seed)
    valid_coords = np.argwhere(np.array(base_a)[:, :, 3] > 10)
    num_valid = len(valid_coords)

    # A and B belong to the same cluster
    cluster_id = "transitivity_cluster_ab"
    ground_truth_counts[cluster_id] = 2

    # Item A
    items.append(BenchmarkItem(filename="chain_A.png", image=base_a.copy(), cluster_id=cluster_id))

    # Item B: diff(A, B) ~ 0.18
    arr_b = np.array(base_a).copy()
    for y, x in valid_coords[: int(num_valid * 0.18)]:
        arr_b[y, x, 0] = (int(arr_b[y, x, 0]) + 60) % 256
    img_b = Image.fromarray(arr_b, mode="RGBA")
    items.append(BenchmarkItem(filename="chain_B.png", image=img_b, cluster_id=cluster_id))

    # Item C: diff(B, C) ~ 0.20, diff(A, C) ~ 0.48
    arr_c = np.array(arr_b).copy()
    for y, x in valid_coords[int(num_valid * 0.18) : int(num_valid * 0.48)]:
        arr_c[y, x, 1] = (int(arr_c[y, x, 1]) + 90) % 256
    img_c = Image.fromarray(arr_c, mode="RGBA")
    # C is an isolated item (does not belong to AB cluster)
    items.append(BenchmarkItem(filename="chain_C.png", image=img_c, cluster_id=None))

    return items, ground_truth_counts
