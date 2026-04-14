#!/usr/bin/env python3
"""
Production-ready COCO Instance Segmentation Dataset Preparation Pipeline.
Target model: Mask2Former (and any COCO-format segmentation model).

Pipeline stages:
  1. Discovery   — walk input_dir, collect (image, mask, label) triples
  2. Validation  — parallel workers: check files, mask area, polygon validity
  3. Augmentation — albumentations with EXACT polygon keypoint transforms
  4. COCO Build  — stratified 80/20 split, parallel image copy + COCO JSON
  5. Report      — pipeline_report.json + pipeline.log

Output structure:
    <output_dir>/
      images/
        train/<category>/
        val/<category>/
      annotations/
        instances_train.json
        instances_val.json
      pipeline_report.json
      pipeline.log

Usage:
    # Full pipeline (25% aug, 80/20 split, 4 workers)
    python prepare_dataset.py --input /mnt/data/output_data --output /mnt/large_volume/training_data

    # Skip augmentation
    python prepare_dataset.py --input /mnt/data/output_data --output /mnt/large_volume/training_data --no-aug

    # Custom settings
    python prepare_dataset.py \\
        --input /mnt/data/output_data \\
        --output /mnt/large_volume/training_data \\
        --aug-ratio 0.20 \\
        --val-split 0.20 \\
        --workers 4
"""

import os
import sys
import json
import shutil
import logging
import argparse
import random
import time
import inspect
import gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set

import cv2
import numpy as np
from tqdm import tqdm

# Keep OpenCV from spawning large thread pools in each worker process.
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

# ── Fast JSON writer (3-5× faster than stdlib for large COCO files) ──────────
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False

# ── albumentations (required for correct polygon-aware augmentation) ─────────
try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

# ─────────────────────────────── CONFIG ─────────────────────────────────────

@dataclass
class Config:
    input_dir: str = "output_data"
    output_dir: str = "training_data"

    aug_ratio: float = 0.12          # fraction per category to augment (10-15% sweet spot)
    val_split: float = 0.20          # validation fraction

    workers: int = 4                 # parallel workers
    aug_workers: int = 2             # separate workers for augmentation (memory heavy)
    coco_workers: int = 1            # COCO annotation workers (streamed writer still CPU-heavy)
    min_mask_area_ratio: float = 0.002
    min_mask_pixels: int = 500
    min_polygon_points: int = 3

    seed: int = 42


# ─────────────────────────── DATA STRUCTURES ────────────────────────────────

@dataclass
class Sample:
    image_path: str
    mask_path: str
    label_path: str
    category: str
    base_name: str
    width: int = 0
    height: int = 0
    polygons: Optional[List[List[float]]] = None   # absolute pixel coords (set for augmented samples)


@dataclass
class PipelineReport:
    total_found: int = 0
    valid: int = 0
    removed_missing: int = 0
    removed_corrupted: int = 0
    removed_small_mask: int = 0
    removed_invalid_label: int = 0
    augmented: int = 0
    train_count: int = 0
    val_count: int = 0
    removed_log: List[str] = field(default_factory=list)
    by_category: Dict[str, int] = field(default_factory=dict)

    @property
    def total_removed(self):
        return (self.removed_missing + self.removed_corrupted
                + self.removed_small_mask + self.removed_invalid_label)

    def save(self, path: str):
        data = {
            "total_found": self.total_found,
            "valid": self.valid,
            "removed": {
                "total": self.total_removed,
                "missing_files": self.removed_missing,
                "corrupted": self.removed_corrupted,
                "small_or_empty_mask": self.removed_small_mask,
                "invalid_label": self.removed_invalid_label,
            },
            "augmented": self.augmented,
            "final_dataset": {
                "train": self.train_count,
                "val": self.val_count,
                "total": self.train_count + self.val_count,
            },
            "by_category": self.by_category,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ─────────────────────────── LOGGING ────────────────────────────────────────

def setup_logging(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "pipeline.log")

    logger = logging.getLogger("COCOPipeline")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _load_augmented_manifest(manifest_path: str, logger: logging.Logger) -> List[Sample]:
    """Load augmented samples saved from a previous run."""
    samples: List[Sample] = []
    if not os.path.exists(manifest_path):
        return samples

    try:
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                img_path = rec.get("image_path", "")
                mask_path = rec.get("mask_path", "")
                if not (img_path and mask_path and os.path.exists(img_path) and os.path.exists(mask_path)):
                    continue
                samples.append(Sample(
                    image_path=img_path,
                    mask_path=mask_path,
                    label_path=rec.get("label_path", ""),
                    category=rec["category"],
                    base_name=rec["base_name"],
                    width=int(rec.get("width", 0)),
                    height=int(rec.get("height", 0)),
                    polygons=rec.get("polygons"),
                ))
    except Exception as e:
        logger.warning(f"Could not load augmentation manifest {manifest_path}: {e}")
        return []

    logger.info(f"Loaded {len(samples):,} augmented samples from manifest")
    return samples


# ─────────────────────────── STEP 1: DISCOVER ───────────────────────────────

def discover_samples(input_dir: str, logger: logging.Logger) -> List[Sample]:
    """Walk input_dir, collect (image, mask, label) triples per category folder."""
    samples: List[Sample] = []
    input_path = Path(input_dir)

    category_dirs = [p for p in sorted(input_path.iterdir()) if p.is_dir() and not p.name.startswith("_")]
    for category_dir in tqdm(category_dirs, desc="Discovering", unit="cat"):
        category = category_dir.name
        for img_file in sorted(category_dir.iterdir()):
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".webp"):
                continue
            if img_file.stem.endswith("_overlay"):   # explicitly skip overlays
                continue

            base = img_file.stem
            mask_path  = str(category_dir / f"{base}.png")
            label_path = str(category_dir / f"{base}.txt")

            samples.append(Sample(
                image_path=str(img_file),
                mask_path=mask_path,
                label_path=label_path,
                category=category,
                base_name=base,
            ))

    n_cats = len(set(s.category for s in samples))
    logger.info(f"Discovered {len(samples):,} candidate samples across {n_cats} categories")
    return samples


# ─────────────────────────── STEP 2: VALIDATE ───────────────────────────────

def _get_image_dimensions(img_path: str) -> Tuple[int, int]:
    """Get (W, H) by decoding only the JPEG/WebP header — avoids full pixel decode."""
    # IMREAD_REDUCED* flags don't help for just dims; fastest is read header bytes
    # For JPEG/WebP the file header is tiny.  cv2.imread is still fast for dims.
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return 0, 0
    H, W = img.shape[:2]
    return W, H


def _validate_worker(args) -> Tuple[bool, str, str, int, int]:
    """Validate one sample. Returns (is_valid, reason, image_path, width, height)."""
    img_path, mask_path, label_path, min_area_ratio, min_pixels = args

    # Fast existence checks first (no IO beyond stat)
    if not os.path.exists(mask_path):
        return False, "missing_mask", img_path, 0, 0
    if not os.path.exists(label_path):
        return False, "missing_label", img_path, 0, 0
    if os.path.getsize(label_path) == 0:
        return False, "empty_label", img_path, 0, 0

    # Validate label BEFORE loading images (cheapest IO first)
    try:
        with open(label_path) as f:
            first_line = f.readline().strip()
        parts = first_line.split()
        if len(parts) < 7:   # class_id + at least 3 (x,y) pairs
            return False, "invalid_label_points", img_path, 0, 0
        [float(p) for p in parts[1:]]
    except Exception:
        return False, "invalid_label_parse", img_path, 0, 0

    # Image dims
    img = cv2.imread(img_path)
    if img is None:
        return False, "corrupted_image", img_path, 0, 0
    H, W = img.shape[:2]
    if H < 32 or W < 32:
        return False, "too_small_image", img_path, 0, 0

    # Mask quality
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False, "corrupted_mask", img_path, 0, 0
    nonzero = int(np.count_nonzero(mask))
    if nonzero < min_pixels:
        return False, "empty_mask", img_path, 0, 0
    if nonzero / (H * W) < min_area_ratio:
        return False, "small_mask", img_path, 0, 0

    return True, "ok", img_path, W, H


def validate_samples(
    samples: List[Sample],
    cfg: Config,
    logger: logging.Logger,
    report: PipelineReport,
) -> List[Sample]:
    """Parallel validation. Returns only valid samples."""
    logger.info(f"Validating {len(samples):,} samples ({cfg.workers} workers) ...")

    worker_args = [
        (s.image_path, s.mask_path, s.label_path,
         cfg.min_mask_area_ratio, cfg.min_mask_pixels)
        for s in samples
    ]
    valid_paths: Dict[str, Tuple[int, int]] = {}   # img_path → (W, H)
    reason_counts: Dict[str, int] = defaultdict(int)

    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        futures = {ex.submit(_validate_worker, a): s for a, s in zip(worker_args, samples)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Validating", unit="img"):
            is_valid, reason, img_path, w, h = fut.result()
            if is_valid:
                valid_paths[img_path] = (w, h)
            else:
                reason_counts[reason] += 1
                report.removed_log.append(f"{reason}: {img_path}")

    valid_samples = []
    for s in samples:
        if s.image_path in valid_paths:
            s.width, s.height = valid_paths[s.image_path]
            valid_samples.append(s)
    report.total_found        = len(samples)
    report.valid              = len(valid_samples)
    report.removed_missing    = reason_counts["missing_mask"] + reason_counts["missing_label"]
    report.removed_corrupted  = (reason_counts["corrupted_image"] + reason_counts["corrupted_mask"]
                                 + reason_counts["too_small_image"])
    report.removed_small_mask = reason_counts["empty_mask"] + reason_counts["small_mask"]
    report.removed_invalid_label = (reason_counts["empty_label"]
                                    + reason_counts["invalid_label_points"]
                                    + reason_counts["invalid_label_parse"])

    logger.info(f"Validation: {len(valid_samples):,} valid | "
                f"{len(samples) - len(valid_samples):,} removed")
    for reason, count in sorted(reason_counts.items()):
        logger.info(f"  ↳ {reason}: {count:,}")
    return valid_samples


# ─────────────────────────── POLYGON UTILS ──────────────────────────────────

def load_annotations(label_path: str, W: int, H: int) -> List[List[float]]:
    """
    Load all polygon annotations from a YOLO .txt file.
    Returns list of polygon coords in absolute pixel space: [[x0,y0,x1,y1,...], ...]
    Supports multiple objects per image (multi-line .txt).
    """
    polygons: List[List[float]] = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:          # class_id + 3 pairs minimum
                continue
            coords = [float(v) for v in parts[1:]]   # skip class_id
            # Auto-detect: if all coords <= 1.0 they are normalized; else absolute
            is_normalized = all(0.0 <= v <= 1.0 for v in coords)
            abs_coords = []
            for i in range(0, len(coords) - 1, 2):
                x = coords[i] * W if is_normalized else coords[i]
                y = coords[i + 1] * H if is_normalized else coords[i + 1]
                abs_coords.append(x)
                abs_coords.append(y)
            if len(abs_coords) >= 6:    # at least 3 points
                polygons.append(abs_coords)
    return polygons


def polygon_to_coco(
    abs_coords: List[float],
    W: int,
    H: int,
) -> Tuple[List[float], List[float], float]:
    """
    Convert an absolute-pixel polygon to COCO fields.
    Returns (segmentation_flat, bbox [x,y,w,h], area).
    """
    seg = [round(float(v), 2) for v in abs_coords]

    xs = abs_coords[0::2]
    ys = abs_coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bbox_w = round(x_max - x_min, 2)
    bbox_h = round(y_max - y_min, 2)
    bbox   = [round(x_min, 2), round(y_min, 2), bbox_w, bbox_h]

    # Shoelace area
    n = len(xs)
    area = abs(
        sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n)) / 2
    )
    return seg, bbox, round(area, 2)


# ─────────────────────────── STEP 3: AUGMENTATION ───────────────────────────

def _supports_kwarg(transform_cls, kwarg: str) -> bool:
    """Return True if a transform constructor supports the given kwarg."""
    try:
        return kwarg in inspect.signature(transform_cls.__init__).parameters
    except (TypeError, ValueError):
        return False


def _make_perspective():
    kwargs = {
        "scale": (0.02, 0.035),
        "keep_size": True,
        "p": 0.25,
    }
    if _supports_kwarg(A.Perspective, "border_mode"):
        kwargs["border_mode"] = cv2.BORDER_REFLECT_101
    elif _supports_kwarg(A.Perspective, "pad_mode"):
        kwargs["pad_mode"] = cv2.BORDER_REFLECT_101
    return A.Perspective(**kwargs)


def _make_image_compression():
    if _supports_kwarg(A.ImageCompression, "quality_range"):
        return A.ImageCompression(quality_range=(70, 100), p=0.20)
    return A.ImageCompression(quality_lower=70, quality_upper=100, p=0.20)


def _make_small_affine():
    kwargs = {
        "scale": (0.90, 1.10),
        "translate_percent": (-0.06, 0.06),
        "rotate": (0.0, 0.0),
        "p": 0.30,
    }
    if _supports_kwarg(A.Affine, "border_mode"):
        kwargs["border_mode"] = cv2.BORDER_REFLECT_101
    elif _supports_kwarg(A.Affine, "mode"):
        kwargs["mode"] = cv2.BORDER_REFLECT_101
    return A.Affine(**kwargs)


def _make_gauss_noise():
    # Albumentations v2 uses std_range/mean_range; v1 uses var_limit.
    if _supports_kwarg(A.GaussNoise, "std_range"):
        return A.GaussNoise(std_range=(0.04, 0.12), mean_range=(0.0, 0.0), p=0.20)
    return A.GaussNoise(var_limit=(10.0, 40.0), p=0.20)


def _make_coarse_dropout():
    # Albumentations v2 switched to *_range + fill.
    if _supports_kwarg(A.CoarseDropout, "num_holes_range"):
        return A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(0.02, 0.06),
            hole_width_range=(0.02, 0.06),
            fill=128,
            p=0.12,
        )
    return A.CoarseDropout(
        max_holes=6,
        max_height=32,
        max_width=32,
        min_holes=1,
        min_height=12,
        min_width=12,
        fill_value=128,
        p=0.12,
    )

def _build_transform(W: int, H: int):
    """
    Domain-generalization augmentation pipeline for Myntra catalog → real-world.

    Uses the albumentations keypoints API so polygon coordinates are
    transformed exactly alongside the image + mask.

    Design rationale (studio → real-world domain gap):
    ─────────────────────────────────────────────────────
    Myntra images are studio-shot: white background, perfect lighting,
    centered subject, sharp focus.  Real-world images have varied lighting,
    cluttered backgrounds, phone-camera noise, and slight perspective shifts.
    Each transform below targets a specific gap dimension.
    """
    return A.Compose(
        [
            # ── 1. GEOMETRIC (spatial — applied to image + mask + keypoints) ──

            # Mirror: clothing appears from both sides; doubles effective poses.
            # Simulates: selfie-mirror / left-right variation.
            A.HorizontalFlip(p=0.5),

            # Mild perspective warp: simulates phone held at slight angles.
            # limit=0.03 keeps garment shape intact (< 3% foreshortening).
            _make_perspective(),

            # Small shift + scale: subject not always centered / same distance.
            # Implemented via Affine to avoid ShiftScaleRotate deprecation warnings.
            _make_small_affine(),

            # Gentle rotation (±12°): phone tilt, slightly angled shots.
            # Anything >20° is unrealistic for fashion photos.
            A.Rotate(
                limit=12,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.25,
            ),

            # ── 2. LIGHTING (pixel-only — does NOT touch mask/keypoints) ───

            # Brightness + contrast swing: overexposed windows, dim rooms.
            # Studio images are perfectly lit; real scenes vary ±25%.
            A.RandomBrightnessContrast(
                brightness_limit=0.30,
                contrast_limit=0.25,
                p=0.50,
            ),

            # Hue / saturation / value: different white-balance settings,
            # warm tungsten vs cool fluorescent lighting.
            A.HueSaturationValue(
                hue_shift_limit=12,
                sat_shift_limit=25,
                val_shift_limit=15,
                p=0.40,
            ),

            # Random gamma: simulates non-linear display / camera curves.
            A.RandomGamma(gamma_limit=(80, 120), p=0.20),

            # CLAHE: local contrast enhancement — mimics variable indoor
            # lighting where parts of the garment are in shadow.
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.15),

            # Subtle per-channel shift: simulates white-balance drift across
            # different cameras / lighting.  Unlike ChannelShuffle (which swaps
            # entire channels → red dress becomes blue), RGBShift applies small
            # additive offsets so colors stay recognizable while reducing the
            # model's reliance on exact studio color calibration.
            A.RGBShift(
                r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.10
            ),

            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.15,
                hue=0.05,
                p=0.15,
            ),
            # ── 3. CAMERA IMPERFECTIONS (pixel-only) ──────────────────────

            # Gaussian blur: slightly out-of-focus phone cameras.
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),

            # Motion blur: hand shake during capture (kernel 3-5 is subtle).
            A.MotionBlur(blur_limit=(3, 5), p=0.10),

            # JPEG compression: social-media re-compression artifacts.
            # quality_lower=70 matches WhatsApp / Instagram compression.
            _make_image_compression(),

            # ── 4. SENSOR NOISE (pixel-only) ──────────────────────────────

            # Gaussian noise: ISO noise from phone sensors in low light.
            # var_limit 10-40 matches real phone ISO 400-1600 graininess.
            _make_gauss_noise(),

            # ── 5. BACKGROUND DISRUPTION (pixel-only) ─────────────────────
            # Studio white backgrounds are the #1 overfitting risk.
            # These transforms make the model less reliant on background.

            # Random shadow overlay: simulates cast shadows from furniture,
            # other people, etc.  Doesn't affect mask/keypoints.
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                p=0.15,
            ),

            # Random tone curve: simulates the washed-out, low-contrast look
            # of hazy / overcast environments and cheap phone auto-tone.
            # More stable than RandomFog (whose alpha_coef API changed across
            # albumentations versions) and cheaper to compute.
            A.RandomToneCurve(scale=0.15, p=0.10),

            # Coarse dropout: simulates partial occlusion (hand, bag, etc).
            # Small holes only — never more than 5% of image area.
            # fill_value=128 (neutral gray) avoids stark black-on-white artifacts
            # that the model could overfit to as a shortcut signal.
            _make_coarse_dropout(),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=False,   # keep point count stable; we clip below
            label_fields=["kp_labels"],
        ),
    )


def _polygons_to_keypoints(
    polygons: List[List[float]],
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """Flatten polygons into a flat keypoint list + per-keypoint polygon index."""
    kps: List[Tuple[float, float]] = []
    labels: List[int] = []
    for poly_idx, coords in enumerate(polygons):
        for i in range(0, len(coords) - 1, 2):
            kps.append((coords[i], coords[i + 1]))
            labels.append(poly_idx)
    return kps, labels


def _keypoints_to_polygons(
    kps: List[Tuple[float, float]],
    labels: List[int],
    n_polys: int,
) -> List[List[float]]:
    """Reassemble transformed keypoints back into per-polygon coord lists."""
    poly_map: Dict[int, List[float]] = defaultdict(list)
    for (x, y), poly_idx in zip(kps, labels):
        poly_map[poly_idx].extend([x, y])
    return [poly_map[i] for i in range(n_polys) if len(poly_map[i]) >= 6]


# Module-level transform cache — avoids rebuilding per image (~0.5ms saved per call).
# Since augmentation workers are forked processes, each gets its own cache.
_transform_cache: Dict[Tuple[int, int], object] = {}


def _get_transform(W: int, H: int):
    """Return a cached transform for given (W, H). Build only on first call."""
    key = (W, H)
    if key not in _transform_cache:
        _transform_cache[key] = _build_transform(W, H)
    return _transform_cache[key]


def apply_augmentation(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    polygons: List[List[float]],
    W: int,
    H: int,
) -> Tuple[np.ndarray, np.ndarray, List[List[float]]]:
    """
    Apply albumentations transforms to image, mask, AND polygons in sync.
    Polygons are fed as keypoints so they are transformed exactly.
    Returns (aug_image_rgb, aug_mask, aug_polygons_abs_pixel).
    """
    transform = _get_transform(W, H)
    kps, kp_labels = _polygons_to_keypoints(polygons)

    result = transform(
        image=img_rgb,
        mask=mask,
        keypoints=kps,
        kp_labels=kp_labels,
    )

    aug_img  = result["image"]
    aug_mask = result["mask"]
    aug_kps  = result["keypoints"]
    aug_lbls = result["kp_labels"]

    # Clip any out-of-bounds keypoints after transform
    new_H, new_W = aug_img.shape[:2]
    clipped_kps = [
        (max(0.0, min(float(x), new_W - 1)), max(0.0, min(float(y), new_H - 1)))
        for (x, y) in aug_kps
    ]

    aug_polygons = _keypoints_to_polygons(clipped_kps, aug_lbls, len(polygons))

    # ── Post-augmentation polygon validation ──────────────────────────────
    # Drop degenerate polygons (zero bbox dimension or tiny area)
    valid_polygons = []
    for poly in aug_polygons:
        xs = poly[0::2]
        ys = poly[1::2]
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
        if bw < 1.0 or bh < 1.0:
            continue   # collapsed to a line
        # Shoelace quick area check
        n = len(xs)
        area = abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i]
                       for i in range(n)) / 2)
        if area < 1.0:
            continue   # degenerate
        valid_polygons.append(poly)

    return aug_img, aug_mask, valid_polygons


def _augment_worker(args) -> Tuple[bool, str, Optional[dict]]:
    """
    Per-sample augmentation worker.
    Returns (success, img_path, result_dict | None).
    result_dict: {aug_img_path, aug_mask_path, polygons, W, H}
    """
    sample_dict, out_root, min_pixels = args

    if not HAS_ALBUMENTATIONS:
        return False, sample_dict["image_path"], None

    img_path   = sample_dict["image_path"]
    mask_path  = sample_dict["mask_path"]
    label_path = sample_dict["label_path"]
    category   = sample_dict["category"]
    base_name  = sample_dict["base_name"]
    aug_seed   = sample_dict.get("aug_seed", 0)

    # Per-sample deterministic seed → reproducible augmentation.
    # Combines the global seed with a hash of the filename so every
    # image gets a unique-but-repeatable random state.
    if aug_seed:
        random.seed(aug_seed)
        np.random.seed(aug_seed % (2**31))

    try:
        img = cv2.imread(img_path)
        if img is None:
            return False, img_path, None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W    = img.shape[:2]

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False, img_path, None
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        polygons = load_annotations(label_path, W, H)
        if not polygons:
            return False, img_path, None

        aug_img_rgb, aug_mask, aug_polygons = apply_augmentation(
            img_rgb, mask_bin, polygons, W, H
        )
        aug_H, aug_W = aug_img_rgb.shape[:2]

        if not aug_polygons or np.count_nonzero(aug_mask) < min_pixels:
            return False, img_path, None

        aug_base = f"aug_{base_name}"
        ext      = Path(img_path).suffix
        out_dir  = os.path.join(out_root, category)
        os.makedirs(out_dir, exist_ok=True)

        aug_img_path  = os.path.join(out_dir, f"{aug_base}{ext}")
        aug_mask_path = os.path.join(out_dir, f"{aug_base}.png")

        aug_img_bgr = cv2.cvtColor(aug_img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(aug_img_path,  aug_img_bgr)
        cv2.imwrite(aug_mask_path, aug_mask)

        return True, img_path, {
            "aug_img_path":  aug_img_path,
            "aug_mask_path": aug_mask_path,
            "category":      category,
            "base_name":     aug_base,
            "polygons":      aug_polygons,
            "W": aug_W,
            "H": aug_H,
        }
    except Exception:
        return False, img_path, None


def augment_samples(
    valid_samples: List[Sample],
    cfg: Config,
    out_root: str,
    logger: logging.Logger,
    report: PipelineReport,
    resume: bool = False,
) -> List[Sample]:
    """Stratified selection + parallel augmentation. Returns new Sample list."""
    if not HAS_ALBUMENTATIONS:
        logger.warning("albumentations not installed. Skipping augmentation.")
        return []

    random.seed(cfg.seed)
    by_cat: Dict[str, List[Sample]] = defaultdict(list)
    for s in valid_samples:
        by_cat[s.category].append(s)

    manifest_path = os.path.join(os.path.dirname(out_root), "_augmented_manifest.jsonl")
    existing_aug_samples: List[Sample] = []
    done_source_keys: Set[Tuple[str, str]] = set()

    if resume and os.path.exists(manifest_path):
        existing_aug_samples = _load_augmented_manifest(manifest_path, logger)
        # Manifest uses aug_ prefixed base_name; map back to original source base.
        for s in existing_aug_samples:
            src_base = s.base_name[4:] if s.base_name.startswith("aug_") else s.base_name
            done_source_keys.add((s.category, src_base))
        if done_source_keys:
            logger.info(f"Resume mode: {len(done_source_keys):,} augmentation items already completed")

    to_augment: List[Sample] = []
    for cat, cat_samples in by_cat.items():
        n = max(1, int(len(cat_samples) * cfg.aug_ratio))
        selected = random.sample(cat_samples, min(n, len(cat_samples)))
        if done_source_keys:
            selected = [s for s in selected if (s.category, s.base_name) not in done_source_keys]
        to_augment.extend(selected)

    logger.info(f"Augmenting {len(to_augment):,} samples "
                f"({cfg.aug_ratio * 100:.0f}% per category) using albumentations keypoint API ...")

    total_tasks = len(to_augment)

    def _worker_args_iter():
        for s in to_augment:
            yield (
                {
                    "image_path": s.image_path,
                    "mask_path": s.mask_path,
                    "label_path": s.label_path,
                    "category": s.category,
                    "base_name": s.base_name,
                    "aug_seed": cfg.seed + hash(s.base_name) % (2**31),
                },
                out_root,
                cfg.min_mask_pixels,
            )

    aug_samples: List[Sample] = existing_aug_samples[:]
    ok_count = 0

    aug_workers = max(1, min(cfg.aug_workers, cfg.workers))
    logger.info(f"Augmentation workers: {aug_workers}")
    with open(manifest_path, "a") as manifest_f:
        with ProcessPoolExecutor(max_workers=aug_workers) as ex:
            for ok, _, result in tqdm(
                ex.map(_augment_worker, _worker_args_iter(), chunksize=16),
                total=total_tasks, desc="Augmenting",
            ):
                if ok and result:
                    ok_count += 1
                    sample = Sample(
                        image_path=result["aug_img_path"],
                        mask_path=result["aug_mask_path"],
                        label_path="",
                        category=result["category"],
                        base_name=result["base_name"],
                        width=result["W"],
                        height=result["H"],
                        polygons=result["polygons"],   # carry augmented polygons forward
                    )
                    aug_samples.append(sample)

                    rec = {
                        "image_path": sample.image_path,
                        "mask_path": sample.mask_path,
                        "label_path": sample.label_path,
                        "category": sample.category,
                        "base_name": sample.base_name,
                        "width": sample.width,
                        "height": sample.height,
                        "polygons": sample.polygons,
                    }
                    manifest_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    manifest_f.flush()

    report.augmented = len(aug_samples)
    logger.info(
        f"Augmentation done: +{ok_count:,} new / {len(to_augment):,} requested; "
        f"total augmented available: {len(aug_samples):,}"
    )
    return aug_samples


# ─────────────────────────── STEP 4: COCO BUILD ─────────────────────────────

def _build_coco_worker(args) -> Optional[dict]:
    """
    Per-sample worker for the COCO build step.
    Uses cached dims + polygons when available (augmented samples),
    otherwise loads from disk.
    Returns a partial-record dict or None on failure.
    """
    img_path, label_path, category, base_name, cached_w, cached_h, cached_polygons = args

    try:
        # Use cached dimensions from validation/augmentation when available
        if cached_w > 0 and cached_h > 0:
            W, H = cached_w, cached_h
        else:
            img = cv2.imread(img_path)
            if img is None:
                return None
            H, W = img.shape[:2]

        file_name = f"{category}/{os.path.basename(img_path)}"

        # Use in-memory polygons for augmented samples; load from disk otherwise
        if cached_polygons:
            polygons = cached_polygons
        else:
            if not label_path or not os.path.exists(label_path):
                return None
            polygons = load_annotations(label_path, W, H)

        if not polygons:
            return None

        annotations = []
        for poly in polygons:
            seg, bbox, area = polygon_to_coco(poly, W, H)
            if area < 1:
                continue
            annotations.append({
                "segmentation": [seg],      # COCO: list of polygons per instance
                "bbox":         bbox,
                "area":         area,
                "iscrowd":      0,
                "category_name": category,
            })

        if not annotations:
            return None

        return {
            "file_name":  file_name,
            "img_path":   img_path,
            "width":      W,
            "height":     H,
            "category":   category,
            "base_name":  base_name,
            "annotations": annotations,
        }
    except Exception:
        return None


def _make_coco_timestamp() -> str:
    import datetime
    return datetime.datetime.utcnow().strftime("%Y/%m/%d")


def _json_dumps_line(obj: dict) -> str:
    """Compact JSON line serializer (uses orjson when available)."""
    if HAS_ORJSON:
        return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")
    return json.dumps(obj, separators=(",", ":"))


def _assemble_coco_json(
    json_path: str,
    images_jsonl: str,
    anns_jsonl: str,
    categories: List[Dict],
):
    """Assemble final COCO JSON from streamed image/annotation JSONL files."""
    with open(json_path, "w") as out:
        out.write("{")
        out.write('"info":')
        out.write(_json_dumps_line({
            "description": "Fashion Instance Segmentation Dataset",
            "date_created": _make_coco_timestamp(),
            "version": "1.0",
        }))
        out.write(',"licenses":[]')
        out.write(',"categories":')
        out.write(_json_dumps_line(categories))

        out.write(',"images":[')
        first = True
        with open(images_jsonl, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if not first:
                    out.write(",")
                out.write(line)
                first = False
        out.write("]")

        out.write(',"annotations":[')
        first = True
        with open(anns_jsonl, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if not first:
                    out.write(",")
                out.write(line)
                first = False
        out.write("]")
        out.write("}\n")


def write_coco_split_streaming(
    samples: List[Sample],
    categories: List[Dict],
    cat_name_to_id: Dict[str, int],
    workers: int,
    json_path: str,
    desc: str,
) -> Tuple[int, int]:
    """
    Memory-safe COCO writer.
    Streams per-image and per-annotation records to JSONL temp files,
    then assembles final COCO JSON without keeping huge lists in RAM.
    Returns (num_images, num_annotations).
    """
    worker_args = [
        (s.image_path, s.label_path, s.category, s.base_name,
         s.width, s.height, s.polygons)
        for s in samples
    ]

    images_jsonl = f"{json_path}.images.jsonl"
    anns_jsonl = f"{json_path}.anns.jsonl"

    image_id = 1
    ann_id = 1
    num_images = 0
    num_anns = 0

    with open(images_jsonl, "w") as img_f, open(anns_jsonl, "w") as ann_f:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for result in tqdm(
                ex.map(_build_coco_worker, worker_args, chunksize=64),
                total=len(worker_args), desc=desc,
            ):
                if result is None:
                    continue

                img_rec = {
                    "id": image_id,
                    "file_name": result["file_name"],
                    "width": result["width"],
                    "height": result["height"],
                    "category": result["category"],
                }
                img_f.write(_json_dumps_line(img_rec) + "\n")
                num_images += 1

                for ann in result["annotations"]:
                    ann_rec = {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cat_name_to_id[ann["category_name"]],
                        "segmentation": ann["segmentation"],
                        "bbox": ann["bbox"],
                        "area": ann["area"],
                        "iscrowd": 0,
                    }
                    ann_f.write(_json_dumps_line(ann_rec) + "\n")
                    ann_id += 1
                    num_anns += 1

                image_id += 1

    _assemble_coco_json(json_path, images_jsonl, anns_jsonl, categories)
    os.remove(images_jsonl)
    os.remove(anns_jsonl)
    return num_images, num_anns


def _copy_image_worker(args) -> bool:
    """Copy image — try hardlink first (instant, zero IO) then fallback to copy."""
    img_src, img_dst = args
    try:
        # Dir creation is batched before this is called, but guard just in case
        dst_dir = os.path.dirname(img_dst)
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
        try:
            os.link(img_src, img_dst)      # hardlink = instant, shares inode
        except (OSError, PermissionError):
            shutil.copy2(img_src, img_dst)  # cross-device fallback
        return True
    except Exception:
        return False


def build_dataset(
    all_samples: List[Sample],
    output_dir: str,
    cfg: Config,
    logger: logging.Logger,
    report: PipelineReport,
):
    """
    Main COCO dataset builder:
      1. Build category → ID mapping (sorted, stable)
      2. Stratified 80/20 split
      3. Copy images into images/train|val/<category>/
      4. Build COCO JSON in parallel → annotations/instances_train|val.json
    """
    logger.info(f"Building COCO dataset from {len(all_samples):,} samples ...")
    random.seed(cfg.seed)

    # ── 1. Category → ID mapping (1-indexed, sorted for stability) ───────
    all_cats = sorted(set(s.category for s in all_samples))
    cat_name_to_id = {name: idx + 1 for idx, name in enumerate(all_cats)}
    categories = [{"id": cat_name_to_id[name], "name": name, "supercategory": "fashion"}
                  for name in all_cats]
    logger.info(f"Categories: {len(all_cats)}")

    # ── 2. Stratified split ───────────────────────────────────────────────
    by_cat: Dict[str, List[Sample]] = defaultdict(list)
    for s in all_samples:
        by_cat[s.category].append(s)

    train_samples: List[Sample] = []
    val_samples:   List[Sample] = []
    for cat in sorted(by_cat.keys()):
        cat_list = by_cat[cat][:]
        random.shuffle(cat_list)
        n_val = max(1, int(len(cat_list) * cfg.val_split))
        val_samples.extend(cat_list[:n_val])
        train_samples.extend(cat_list[n_val:])
        report.by_category[cat] = len(cat_list)

    report.train_count = len(train_samples)
    report.val_count   = len(val_samples)
    logger.info(f"Split: {len(train_samples):,} train | {len(val_samples):,} val")

    # ── 3. Pre-create all output directories in one pass ───────────────
    dirs_needed: Set[str] = set()
    copy_tasks = []
    for split, split_samples in [("train", train_samples), ("val", val_samples)]:
        for s in split_samples:
            ext     = Path(s.image_path).suffix
            img_dst = os.path.join(output_dir, "images", split,
                                   s.category, f"{s.base_name}{ext}")
            copy_tasks.append((s.image_path, img_dst))
            dirs_needed.add(os.path.dirname(img_dst))

    for d in dirs_needed:
        os.makedirs(d, exist_ok=True)

    ok = 0
    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        for success in tqdm(
            ex.map(_copy_image_worker, copy_tasks, chunksize=128),
            total=len(copy_tasks), desc="Copying images",
        ):
            if success:
                ok += 1
    logger.info(f"Copied {ok:,}/{len(copy_tasks):,} images")

    # ── 4. Build COCO JSONs ───────────────────────────────────────────────
    ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    coco_workers = max(1, min(cfg.coco_workers, cfg.workers))
    logger.info(f"COCO workers: {coco_workers}")
    for split, split_samples in [("train", train_samples), ("val", val_samples)]:
        logger.info(f"Building COCO annotations for {split} ({len(split_samples):,} samples) ...")
        json_path = os.path.join(ann_dir, f"instances_{split}.json")
        n_imgs, n_anns = write_coco_split_streaming(
            split_samples,
            categories,
            cat_name_to_id,
            workers=coco_workers,
            json_path=json_path,
            desc=f"COCO {split}",
        )
        logger.info(f"  ↳ {json_path}  |  {n_imgs:,} images  |  {n_anns:,} annotations")


# ─────────────────────────── MAIN ───────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="COCO Instance Segmentation dataset prep (for Mask2Former)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",           default="output_data",
                   help="Input dir (hyperstack.py output_data)")
    p.add_argument("--output",          default="training_data",
                   help="Output COCO dataset dir")
    p.add_argument("--aug-ratio",       type=float, default=0.12,
                   help="Fraction per category to augment (0–1)")
    p.add_argument("--val-split",       type=float, default=0.20,
                   help="Validation fraction (0–1)")
    p.add_argument("--workers",         type=int,   default=4,
                   help="Parallel workers (match CPU core count)")
    p.add_argument("--aug-workers",     type=int,   default=2,
                   help="Workers for augmentation stage (lower saves RAM)")
    p.add_argument("--coco-workers",    type=int,   default=1,
                   help="Workers for COCO annotation build stage")
    p.add_argument("--min-mask-ratio",  type=float, default=0.002,
                   help="Min mask area / image area")
    p.add_argument("--min-mask-pixels", type=int,   default=500,
                   help="Min absolute mask pixels")
    p.add_argument("--no-aug",          action="store_true",
                   help="Skip augmentation")
    p.add_argument("--limit",           type=int,   default=0,
                   help="Limit total samples (for debugging; 0 = no limit)")
    p.add_argument("--dry-run",         action="store_true",
                   help="Run discovery + validation only, no copy/build")
    p.add_argument("--resume",          action="store_true",
                   help="Resume from existing augmentation manifest when available")
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = Config(
        input_dir=args.input,
        output_dir=args.output,
        aug_ratio=args.aug_ratio,
        val_split=args.val_split,
        workers=args.workers,
        aug_workers=args.aug_workers,
        coco_workers=args.coco_workers,
        min_mask_area_ratio=args.min_mask_ratio,
        min_mask_pixels=args.min_mask_pixels,
        seed=args.seed,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    logger = setup_logging(cfg.output_dir)
    report = PipelineReport()
    t0     = time.time()

    logger.info("=" * 62)
    logger.info("  COCO Instance Segmentation Dataset Pipeline")
    logger.info("=" * 62)
    logger.info(f"Input  : {cfg.input_dir}")
    logger.info(f"Output : {cfg.output_dir}")
    logger.info(
        f"Workers: {cfg.workers}  |  Aug workers: {cfg.aug_workers}  |  COCO workers: {cfg.coco_workers}  |  "
        f"Aug: {cfg.aug_ratio}  |  Val: {cfg.val_split}"
    )

    total_stages = 6
    completed_stages = 0

    def _stage_done(stage_name: str, t_start: float):
        nonlocal completed_stages
        completed_stages += 1
        elapsed = time.time() - t_start
        logger.info(f"[Stage {completed_stages}/{total_stages}] {stage_name} done in {elapsed:.1f}s")

    # ── 1. Discover ───────────────────────────────────────────────────────
    t_stage = time.time()
    samples = discover_samples(cfg.input_dir, logger)
    if not samples:
        logger.error("No samples found. Check --input path.")
        sys.exit(1)
    _stage_done("Discovery", t_stage)

    # ── Debug limit ───────────────────────────────────────────────────────
    if args.limit > 0:
        random.seed(cfg.seed)
        random.shuffle(samples)
        samples = samples[:args.limit]
        logger.info(f"Debug mode: limited to {len(samples):,} samples")

    # ── 2. Validate ───────────────────────────────────────────────────────
    t_stage = time.time()
    valid_samples = validate_samples(samples, cfg, logger, report)
    if not valid_samples:
        logger.error("No valid samples after validation.")
        sys.exit(1)
    _stage_done("Validation", t_stage)

    # Release discovery list before memory-heavy augmentation stage.
    del samples
    gc.collect()

    if args.dry_run:
        logger.info("Dry run complete. No files copied.")
        report_path = os.path.join(cfg.output_dir, "pipeline_report.json")
        report.save(report_path)
        logger.info(f"Report saved: {report_path}")
        return

    # ── 3. Augment ────────────────────────────────────────────────────────
    t_stage = time.time()
    aug_samples: List[Sample] = []
    if not args.no_aug:
        aug_temp = os.path.join(cfg.output_dir, "_augmented_temp")
        aug_samples = augment_samples(valid_samples, cfg, aug_temp, logger, report, resume=args.resume)
    else:
        logger.info("Augmentation skipped (--no-aug)")
    _stage_done("Augmentation", t_stage)

    # ── 4. Build COCO dataset ─────────────────────────────────────────────
    t_stage = time.time()
    all_samples = valid_samples + aug_samples
    build_dataset(all_samples, cfg.output_dir, cfg, logger, report)
    _stage_done("COCO Build", t_stage)

    # ── 5. Cleanup augmentation temp dir ──────────────────────────────────
    t_stage = time.time()
    aug_temp = os.path.join(cfg.output_dir, "_augmented_temp")
    aug_manifest = os.path.join(cfg.output_dir, "_augmented_manifest.jsonl")
    if os.path.isdir(aug_temp):
        shutil.rmtree(aug_temp, ignore_errors=True)
        logger.info("Cleaned up _augmented_temp")
    if os.path.exists(aug_manifest):
        os.remove(aug_manifest)
        logger.info("Cleaned up _augmented_manifest.jsonl")
    _stage_done("Cleanup", t_stage)

    # ── 6. Report ─────────────────────────────────────────────────────────
    t_stage = time.time()
    report_path = os.path.join(cfg.output_dir, "pipeline_report.json")
    report.save(report_path)
    _stage_done("Report", t_stage)

    elapsed = time.time() - t0
    logger.info("=" * 62)
    logger.info(f"  Pipeline complete in {elapsed / 60:.1f} min ({elapsed:.0f}s)")
    logger.info(f"  Discovered : {report.total_found:,}")
    logger.info(f"  Valid      : {report.valid:,}")
    logger.info(f"  Removed    : {report.total_removed:,}")
    logger.info(f"  Augmented  : {report.augmented:,}")
    logger.info(f"  Train      : {report.train_count:,}")
    logger.info(f"  Val        : {report.val_count:,}")
    logger.info(f"  Categories : {len(report.by_category)}")
    logger.info(f"  Report     : {report_path}")
    logger.info("=" * 62)


if __name__ == "__main__":
    main()