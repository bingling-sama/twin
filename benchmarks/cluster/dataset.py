"""Benchmark dataset generators and scenario builders for seedless texture discovery.

Provides ground-truth labeled clusters of Minecraft skins across:
1. Exact Duplicate Swarms (tests adaptive recall expansion and Top-K penetration).
2. Multi-cluster Derivative Variants (tests recolors, edits, and DiffMask verification).
3. Sparse Real-World Haystack (tests discovery amidst real community skins from images.zip).
4. Threshold Boundary Scenarios (tests strict boundary gating).
5. Transitivity Chain Scenarios (tests anti-transitivity collapse via greedy leader elimination).
"""

from __future__ import annotations

import hashlib
import io
import logging
import random
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from twin.core.types import DiscoveryMode
from twin.services.classifier import TextureKind, inspect_texture
from twin.services.converter import single2double
from twin.services.hasher import compute_phash, hamming_distance
from twin.services.verifiers import DiffMaskVerifier

logger = logging.getLogger(__name__)

ZIP_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "images.zip"
_ZIP_FINGERPRINT_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}


def get_zip_fingerprint(zip_path: Path = ZIP_PATH) -> dict[str, Any]:
    """Return cached size, PNG member count, and SHA-256 fingerprint for images.zip."""
    resolved = zip_path.resolve()
    if not resolved.is_file():
        return {
            "exists": False,
            "path": str(resolved),
            "size_bytes": 0,
            "png_count": 0,
            "sha256": None,
        }

    stat = resolved.stat()
    cache_key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    if cache_key in _ZIP_FINGERPRINT_CACHE:
        return dict(_ZIP_FINGERPRINT_CACHE[cache_key])

    digest = hashlib.sha256()
    with resolved.open("rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            digest.update(chunk)

    with zipfile.ZipFile(resolved, "r") as z:
        png_count = sum(
            1
            for n in z.namelist()
            if n.lower().endswith(".png") and not n.startswith("__MACOSX")
        )

    info = {
        "exists": True,
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "png_count": png_count,
        "sha256": digest.hexdigest(),
    }
    _ZIP_FINGERPRINT_CACHE[cache_key] = info
    return dict(info)


@dataclass
class BenchmarkItem:
    """A ground-truth labeled texture item for clustering benchmarks."""

    filename: str
    image: Image.Image
    cluster_id: str | None  # None indicates an isolated distractor/noise item
    datasource: str = "skin"
    texture_type: TextureKind = TextureKind.SKIN_DOUBLE
    skip_disk_write: bool = False
    image_format: str = "PNG"


def create_base_skin(
    seed: int = 42,
    color_bias: tuple[int, int, int] | None = None,
    valid_pixel_count: int = 3072,
) -> Image.Image:
    """Generate a valid Minecraft 64x64 RGBA skin with seed-distinct spatial patterns."""
    if not 1 <= valid_pixel_count <= 4096:
        raise ValueError(f"valid_pixel_count must be in [1, 4096], got {valid_pixel_count}")

    rng = np.random.RandomState(seed)
    arr = np.zeros((64, 64, 4), dtype=np.uint8)

    if color_bias is None:
        r, g, b = (int(c) for c in rng.randint(35, 225, size=3))
    else:
        r, g, b = color_bias

    # Seed-distinct backdrop inside upper 48 rows so different seeds separate in pHash/DiffMask
    bg_color = [
        int(rng.randint(0, 255)),
        int(rng.randint(0, 255)),
        int(rng.randint(0, 255)),
        255,
    ]
    arr[0:48, 0:64] = bg_color

    # Head (8:16, 8:24)
    head_color = [
        int(np.clip(r + rng.randint(-20, 20), 0, 255)),
        int(np.clip(g + rng.randint(-20, 20), 0, 255)),
        int(np.clip(b + rng.randint(-20, 20), 0, 255)),
        255,
    ]
    arr[8:16, 8:24] = head_color

    # Torso (20:32, 20:40)
    torso_color = [
        int(np.clip(r + rng.randint(-50, 50), 0, 255)),
        int(np.clip(g + rng.randint(-50, 50), 0, 255)),
        int(np.clip(b + rng.randint(-50, 50), 0, 255)),
        255,
    ]
    arr[20:32, 20:40] = torso_color

    # Limbs (20:32, 44:56)
    arr[20:32, 44:56] = head_color

    # Lower body (32:48, 0:64)
    lower_color = [
        int(np.clip(r // 2 + rng.randint(0, 80), 0, 255)),
        int(np.clip(g // 2 + rng.randint(0, 80), 0, 255)),
        int(np.clip(b // 2 + rng.randint(0, 80), 0, 255)),
        255,
    ]
    arr[32:48, 0:64] = lower_color

    # Seeded asymmetric clothing / stripe / accessory motifs for structural DCT diversity
    for _ in range(8):
        y0 = int(rng.randint(2, 40))
        x0 = int(rng.randint(0, 52))
        bh = int(rng.randint(4, 10))
        bw = int(rng.randint(6, 14))
        motif = [int(c) for c in rng.randint(0, 256, size=3)] + [255]
        arr[y0 : min(48, y0 + bh), x0 : min(64, x0 + bw)] = motif

    # Default opaque upper 48 rows -> 3072 valid pixels
    arr[0:48, 0:64, 3] = 255

    if valid_pixel_count < 3072:
        to_clear = 3072 - valid_pixel_count
        flat_alpha = arr[0:48, 0:64, 3].reshape(-1)
        flat_alpha[:to_clear] = 0
    elif valid_pixel_count > 3072:
        to_add = valid_pixel_count - 3072
        flat_alpha = arr[48:64, 0:64, 3].reshape(-1)
        flat_alpha[:to_add] = 255

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
    swarm_sizes: Sequence[int] = (30, 20, 15),
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


def _shift_blue_pixels(
    arr_in: np.ndarray,
    pixel_coords: np.ndarray,
    delta: int = 30,
) -> Image.Image:
    """Shift Blue channel by +/-delta (>15 diff_thresh) with minimal luminance drift."""
    out = arr_in.copy()
    for y, x in pixel_coords:
        b = int(out[y, x, 2])
        out[y, x, 2] = (b + delta) if b <= (255 - delta) else (b - delta)
    return Image.fromarray(out, mode="RGBA")


def _find_exact_hamming_variant(
    base_img: Image.Image,
    target_hamming: int,
    max_diff_ratio: float,
    seed: int,
    verifier: DiffMaskVerifier,
    must_differ_from: list[tuple[Image.Image, int]] | None = None,
) -> Image.Image:
    """Find a small-patch variant hitting an exact pHash Hamming distance within diff_ratio."""
    base_arr = np.array(base_img.convert("RGBA"))
    base_ph = compute_phash(base_img)
    rng = np.random.default_rng(seed)

    high_budget = max_diff_ratio > 0.10
    patch_count_range = (4, 16) if high_budget else (1, 8)
    patch_size_range = (2, 8) if high_budget else (1, 6)

    for _ in range(25000):
        cand = base_arr.copy()
        num_patches = int(rng.integers(*patch_count_range))
        for _ in range(num_patches):
            y, x = int(rng.integers(2, 42)), int(rng.integers(2, 56))
            h = int(rng.integers(*patch_size_range))
            w = int(rng.integers(*patch_size_range))
            cand[y : y + h, x : x + w] = [
                int(rng.integers(0, 256)),
                int(rng.integers(0, 256)),
                int(rng.integers(0, 256)),
                255,
            ]
        img = Image.fromarray(cand, mode="RGBA")
        cand_ph = compute_phash(img)
        if hamming_distance(base_ph, cand_ph) != target_hamming:
            continue
        if verifier.verify(base_img, img).diff_ratio > max_diff_ratio:
            continue
        if must_differ_from and any(
            hamming_distance(compute_phash(ref_img), cand_ph) <= min_dist
            for ref_img, min_dist in must_differ_from
        ):
            continue
        return img

    raise RuntimeError(
        f"Failed to generate exact Hamming={target_hamming} (diff_ratio<={max_diff_ratio}) variant"
    )


def _build_hamming_boundary_items(
    *,
    base_ham: Image.Image,
    cluster_id: str,
    prefix: str,
    target_exact: int,
    target_beyond: int,
    max_diff_ratio: float,
    seed: int,
    verifier: DiffMaskVerifier,
) -> tuple[list[BenchmarkItem], dict[str, int]]:
    """Build exact in-threshold and just-beyond pHash Hamming boundary items."""
    ham_exact = _find_exact_hamming_variant(
        base_ham,
        target_hamming=target_exact,
        max_diff_ratio=max_diff_ratio,
        seed=seed + 1,
        verifier=verifier,
    )
    ham_beyond = _find_exact_hamming_variant(
        base_ham,
        target_hamming=target_beyond,
        max_diff_ratio=max_diff_ratio,
        seed=seed + 2,
        verifier=verifier,
        must_differ_from=[(ham_exact, target_exact)],
    )
    assert hamming_distance(compute_phash(base_ham), compute_phash(ham_exact)) == target_exact
    assert hamming_distance(compute_phash(base_ham), compute_phash(ham_beyond)) == target_beyond
    assert verifier.verify(base_ham, ham_beyond).diff_ratio <= max_diff_ratio

    built = [
        BenchmarkItem(filename=f"{prefix}_base.png", image=base_ham, cluster_id=cluster_id),
        BenchmarkItem(
            filename=f"{prefix}_member_exact.png", image=ham_exact, cluster_id=cluster_id
        ),
        BenchmarkItem(
            filename=f"{prefix}_distractor_beyond.png", image=ham_beyond, cluster_id=None
        ),
    ]
    return built, {cluster_id: 2}


def build_boundary_dataset(
    seed: int = 400,
    mode: DiscoveryMode = "confirmed",
) -> tuple[list[BenchmarkItem], dict[str, int]]:
    """Build boundary dataset covering exact DiffMask and Hamming thresholds with assertions.

    In "confirmed" mode: tests exact 0.05 vs 0.06 diff_ratio and exact Hamming 4 vs 6 (plus >0.35).
    In "variants" mode: tests exact 0.35 vs 0.37 diff_ratio and exact Hamming 12 vs 14.
    """
    verifier = DiffMaskVerifier()
    items: list[BenchmarkItem] = []
    ground_truth_counts: dict[str, int] = {}

    base_diff = create_base_skin(seed=seed, valid_pixel_count=3000)
    arr_diff = np.array(base_diff)
    valid_coords = np.argwhere(arr_diff[:, :, 3] > 10)
    if len(valid_coords) != 3000:
        raise AssertionError(f"Expected 3000 valid pixels, got {len(valid_coords)}")

    base_ham = create_base_skin(seed=seed + 11, color_bias=(60, 180, 210), valid_pixel_count=3000)

    if mode == "confirmed":
        # Mini-cluster A: exact DiffMask confirmed boundaries (0.04 and exact 0.05 vs 0.06 & 0.37)
        c_diff = "boundary_confirmed_diff"
        ground_truth_counts[c_diff] = 3
        member_near = _shift_blue_pixels(arr_diff, valid_coords[:120], delta=30)
        member_exact = _shift_blue_pixels(arr_diff, valid_coords[:150], delta=30)
        dist_conf = _shift_blue_pixels(arr_diff, valid_coords[1500:1680], delta=40)
        dist_var = _shift_blue_pixels(arr_diff, valid_coords[1680:2790], delta=60)

        res_near = verifier.verify(base_diff, member_near)
        res_exact = verifier.verify(base_diff, member_exact)
        res_pair = verifier.verify(member_near, member_exact)
        res_out_conf = verifier.verify(base_diff, dist_conf)
        res_out_var = verifier.verify(base_diff, dist_var)
        assert res_near.diff_ratio == 0.04, res_near
        assert res_exact.diff_ratio == 0.05, res_exact
        assert res_pair.diff_ratio == 0.01, res_pair
        assert hamming_distance(compute_phash(base_diff), compute_phash(member_exact)) <= 4
        assert hamming_distance(compute_phash(member_near), compute_phash(member_exact)) <= 4
        assert res_out_conf.diff_ratio == 0.06, res_out_conf
        assert hamming_distance(compute_phash(base_diff), compute_phash(dist_conf)) <= 4
        assert res_out_var.diff_ratio == 0.37, res_out_var

        items.extend(
            [
                BenchmarkItem(filename="conf_diff_base.png", image=base_diff, cluster_id=c_diff),
                BenchmarkItem(
                    filename="conf_diff_member_near.png", image=member_near, cluster_id=c_diff
                ),
                BenchmarkItem(
                    filename="conf_diff_member_exact.png", image=member_exact, cluster_id=c_diff
                ),
                BenchmarkItem(
                    filename="conf_diff_distractor_beyond.png", image=dist_conf, cluster_id=None
                ),
                BenchmarkItem(
                    filename="var_diff_distractor_beyond.png", image=dist_var, cluster_id=None
                ),
            ]
        )

        # Mini-cluster B: exact Hamming confirmed boundary (4 vs 6; pHash weights are even)
        ham_items, ham_counts = _build_hamming_boundary_items(
            base_ham=base_ham,
            cluster_id="boundary_confirmed_hamming",
            prefix="conf_ham",
            target_exact=4,
            target_beyond=6,
            max_diff_ratio=0.045,
            seed=seed + 100,
            verifier=verifier,
        )
        items.extend(ham_items)
        ground_truth_counts.update(ham_counts)
    else:
        # Mini-cluster A: exact DiffMask variants boundary (exact 0.35 vs 0.37)
        c_diff = "boundary_variants_diff"
        ground_truth_counts[c_diff] = 2
        member_exact = _shift_blue_pixels(arr_diff, valid_coords[:1050], delta=30)
        dist_var = _shift_blue_pixels(arr_diff, valid_coords[1500:2610], delta=40)

        res_exact = verifier.verify(base_diff, member_exact)
        res_out = verifier.verify(base_diff, dist_var)
        res_cross = verifier.verify(member_exact, dist_var)
        assert res_exact.diff_ratio == 0.35, res_exact
        assert hamming_distance(compute_phash(base_diff), compute_phash(member_exact)) <= 12
        assert res_out.diff_ratio == 0.37, res_out
        assert res_cross.diff_ratio > 0.35, res_cross

        items.extend(
            [
                BenchmarkItem(filename="var_diff_base.png", image=base_diff, cluster_id=c_diff),
                BenchmarkItem(
                    filename="var_diff_member_exact.png", image=member_exact, cluster_id=c_diff
                ),
                BenchmarkItem(
                    filename="var_diff_distractor_beyond.png", image=dist_var, cluster_id=None
                ),
            ]
        )

        # Mini-cluster B: exact Hamming variants boundary (12 vs 14; pHash weights are even)
        ham_items, ham_counts = _build_hamming_boundary_items(
            base_ham=base_ham,
            cluster_id="boundary_variants_hamming",
            prefix="var_ham",
            target_exact=12,
            target_beyond=14,
            max_diff_ratio=0.30,
            seed=seed + 200,
            verifier=verifier,
        )
        items.extend(ham_items)
        ground_truth_counts.update(ham_counts)

    return items, ground_truth_counts


def build_transitivity_chain_dataset(
    seed: int = 500,
) -> tuple[list[BenchmarkItem], dict[str, int]]:
    """Build transitivity chain dataset: A ~ B (0.18), B ~ C (0.30), A !~ C (0.48).

    Tests that Greedy Leader Dynamic Elimination clusters A with B and eliminates B,
    preventing non-transitive collapse where C gets improperly merged into A.
    """
    verifier = DiffMaskVerifier()
    items: list[BenchmarkItem] = []
    ground_truth_counts: dict[str, int] = {}
    base_a = create_base_skin(seed=seed, valid_pixel_count=3000)
    arr_a = np.array(base_a)
    valid_coords = np.argwhere(arr_a[:, :, 3] > 10)
    num_valid = len(valid_coords)

    # A and B belong to the same cluster
    cluster_id = "transitivity_cluster_ab"
    ground_truth_counts[cluster_id] = 2

    # Item A
    items.append(BenchmarkItem(filename="chain_A.png", image=base_a.copy(), cluster_id=cluster_id))

    # Item B: Blue shift on first 18% of valid pixels => diff(A, B) == 0.18, low pHash drift
    img_b = _shift_blue_pixels(arr_a, valid_coords[: int(num_valid * 0.18)], delta=30)
    items.append(BenchmarkItem(filename="chain_B.png", image=img_b, cluster_id=cluster_id))

    # Item C: disjoint Blue shift on next 30% => diff(B, C) == 0.30, diff(A, C) == 0.48
    img_c = _shift_blue_pixels(
        np.array(img_b), valid_coords[int(num_valid * 0.18) : int(num_valid * 0.48)], delta=30
    )
    # C is an isolated item (does not belong to AB cluster)
    items.append(BenchmarkItem(filename="chain_C.png", image=img_c, cluster_id=None))

    diff_ab = verifier.verify(base_a, img_b).diff_ratio
    diff_bc = verifier.verify(img_b, img_c).diff_ratio
    diff_ac = verifier.verify(base_a, img_c).diff_ratio
    assert diff_ab <= 0.35, f"Expected A~B <= 0.35, got {diff_ab}"
    assert diff_bc <= 0.35, f"Expected B~C <= 0.35, got {diff_bc}"
    assert diff_ac > 0.35, f"Expected A!~C > 0.35, got {diff_ac}"
    assert hamming_distance(compute_phash(base_a), compute_phash(img_b)) <= 12
    assert hamming_distance(compute_phash(img_b), compute_phash(img_c)) <= 12

    return items, ground_truth_counts


def build_adversarial_dataset(
    seed: int = 600,
) -> tuple[list[BenchmarkItem], dict[str, int]]:
    """Build adversarial dataset covering Spec Section 5 torture cases.

    Positive GT clusters:
    1. adv_hd_cluster: 64x64 base + 128x128 HD NEAREST copy (ratio-preserving scaling).
    2. adv_format_cluster: 64x64 RGBA PNG + 64x64 RGB WebP copy (format/channel variance).
    3. adv_legacy_cluster: two 64x32 SKIN_SINGLE legacy skins triggering single2double upgrade.

    Adversarial traps (cluster_id=None, must reject cleanly without crashing or false merging):
    - False pHash neighbor: low Hamming (<= 4) yet global Blue shift diff_ratio >= 0.55.
    - Alpha tampering: identical RGB (Hamming == 0) with stripped lower-body alpha (diff >= 0.40).
    - Missing disk image: skip_disk_write=True sibling matching indexed pHash without disk file.
    - Aspect ratio mismatch: 128x64 non-square texture paired alongside square skins (diff=1.0).
    """
    verifier = DiffMaskVerifier()
    items: list[BenchmarkItem] = []
    ground_truth_counts: dict[str, int] = {}

    # 1. HD 128x128 positive cluster + aspect mismatch trap (128x64)
    hd_base = create_base_skin(seed=seed, color_bias=(190, 90, 60), valid_pixel_count=3072)
    hd_128 = hd_base.resize((128, 128), resample=Image.Resampling.NEAREST)
    assert verifier.verify(hd_base, hd_128).diff_ratio == 0.0
    assert hamming_distance(compute_phash(hd_base), compute_phash(hd_128)) <= 4
    ground_truth_counts["adv_hd_cluster"] = 2
    items.extend(
        [
            BenchmarkItem(filename="adv_hd_64.png", image=hd_base, cluster_id="adv_hd_cluster"),
            BenchmarkItem(filename="adv_hd_128.png", image=hd_128, cluster_id="adv_hd_cluster"),
        ]
    )

    aspect_bad = hd_base.resize((96, 64), resample=Image.Resampling.NEAREST)
    assert verifier.verify(hd_base, aspect_bad).details.get("reason") == "aspect_ratio_mismatch"
    assert verifier.verify(hd_base, aspect_bad).diff_ratio == 1.0
    items.append(
        BenchmarkItem(
            filename="adv_aspect_mismatch.png",
            image=aspect_bad,
            cluster_id=None,
            texture_type=TextureKind.SKIN_DOUBLE,
        )
    )

    # 2. Format & RGB/RGBA diversity (64x64 opaque PNG vs RGB BMP/RGB PNG)
    fmt_rgba = create_base_skin(seed=seed + 10, color_bias=(70, 170, 140), valid_pixel_count=4096)
    fmt_rgb = fmt_rgba.convert("RGB")
    assert verifier.verify(fmt_rgba, fmt_rgb).diff_ratio == 0.0
    assert hamming_distance(compute_phash(fmt_rgba), compute_phash(fmt_rgb)) == 0
    ground_truth_counts["adv_format_cluster"] = 2
    items.extend(
        [
            BenchmarkItem(
                filename="adv_fmt_rgba.png", image=fmt_rgba, cluster_id="adv_format_cluster"
            ),
            BenchmarkItem(
                filename="adv_fmt_rgb.webp",
                image=fmt_rgb,
                cluster_id="adv_format_cluster",
                image_format="WEBP",
            ),
        ]
    )

    # 3. 64x32 legacy skin auto-upgrade cluster (both SKIN_SINGLE, verified single2double)
    leg_arr = np.zeros((32, 64, 4), dtype=np.uint8)
    leg_arr[8:16, 8:24] = [180, 120, 70, 255]
    leg_arr[20:32, 20:32] = [90, 160, 210, 255]
    leg_arr[20:32, 4:16] = [110, 80, 190, 255]
    leg_arr[20:32, 44:56] = [200, 140, 60, 255]
    leg_arr[:, :, 3] = 255
    leg_a = Image.fromarray(leg_arr, mode="RGBA")
    leg_arr_b = leg_arr.copy()
    leg_arr_b[8:12, 8:14] = [195, 130, 80, 255]
    leg_b = Image.fromarray(leg_arr_b, mode="RGBA")

    assert inspect_texture(leg_a).can_auto_upgrade
    assert inspect_texture(leg_b).can_auto_upgrade
    assert verifier.verify(leg_a, single2double(leg_a)).diff_ratio == 0.0
    assert verifier.verify(leg_a, leg_b).diff_ratio <= 0.05
    assert hamming_distance(compute_phash(leg_a), compute_phash(leg_b)) <= 4
    ground_truth_counts["adv_legacy_cluster"] = 2
    items.extend(
        [
            BenchmarkItem(
                filename="adv_legacy_64x32_a.png",
                image=leg_a,
                cluster_id="adv_legacy_cluster",
                texture_type=TextureKind.SKIN_SINGLE,
            ),
            BenchmarkItem(
                filename="adv_legacy_64x32_b.png",
                image=leg_b,
                cluster_id="adv_legacy_cluster",
                texture_type=TextureKind.SKIN_SINGLE,
            ),
        ]
    )

    # 4. False pHash neighbor (Hamming <= 4, yet diff_ratio > 0.55)
    fph_base = create_base_skin(seed=seed + 20, color_bias=(150, 130, 200), valid_pixel_count=3000)
    fph_arr = np.array(fph_base)
    fph_coords = np.argwhere(fph_arr[:, :, 3] > 10)
    fph_neighbor = _shift_blue_pixels(fph_arr, fph_coords[:1800], delta=30)
    assert hamming_distance(compute_phash(fph_base), compute_phash(fph_neighbor)) <= 4
    assert verifier.verify(fph_base, fph_neighbor).diff_ratio >= 0.55
    items.extend(
        [
            BenchmarkItem(filename="adv_false_phash_a.png", image=fph_base, cluster_id=None),
            BenchmarkItem(filename="adv_false_phash_b.png", image=fph_neighbor, cluster_id=None),
        ]
    )

    # 5. Alpha channel tampering (identical RGB => Hamming == 0, diff_ratio > 0.40)
    alpha_base = create_base_skin(
        seed=seed + 30, color_bias=(210, 160, 90), valid_pixel_count=4096
    )
    alpha_arr = np.array(alpha_base).copy()
    alpha_arr[32:64, :, 3] = 0
    alpha_corrupted = Image.fromarray(alpha_arr, mode="RGBA")
    assert hamming_distance(compute_phash(alpha_base), compute_phash(alpha_corrupted)) == 0
    assert verifier.verify(alpha_base, alpha_corrupted).diff_ratio > 0.35
    items.extend(
        [
            BenchmarkItem(filename="adv_alpha_base.png", image=alpha_base, cluster_id=None),
            BenchmarkItem(
                filename="adv_alpha_corrupted.png", image=alpha_corrupted, cluster_id=None
            ),
        ]
    )

    # 6. Missing disk file trap (identical image, second item never written to disk)
    missing_base = create_base_skin(
        seed=seed + 40, color_bias=(50, 110, 210), valid_pixel_count=3072
    )
    items.extend(
        [
            BenchmarkItem(filename="adv_missing_anchor.png", image=missing_base, cluster_id=None),
            BenchmarkItem(
                filename="adv_missing_ghost.png",
                image=missing_base.copy(),
                cluster_id=None,
                skip_disk_write=True,
            ),
        ]
    )

    return items, ground_truth_counts
