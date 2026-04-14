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
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set

import ctypes
import cv2
import numpy as np
from tqdm import tqdm

# Keep OpenCV from spawning large thread pools in each worker process.
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)


def _release_memory():
    """Force Python + OS to actually free unused heap pages.
    Python's allocator keeps freed arenas; malloc_trim gives them back."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass  # macOS / non-glibc — gc.collect() alone is fine


def _bounded_map(executor, fn, iterable, max_inflight=64, desc=None, total=None):
    """
    Like executor.map() but limits in-flight futures to max_inflight.

    executor.map() materializes ALL futures upfront — with 300K items that means
    300K Future objects + 300K result dicts in memory simultaneously.
    This submits only max_inflight tasks at a time, yielding results in order
    so memory stays bounded (~max_inflight results × result size).
    """
    from collections import deque
    it = iter(iterable)
    pending: deque = deque()   # FIFO of Future objects, preserves submission order

    # Prime the queue
    for _ in range(max_inflight):
        try:
            args = next(it)
        except StopIteration:
            break
        pending.append(executor.submit(fn, args))

    completed = 0
    pbar = tqdm(total=total, desc=desc, unit="img") if desc else None
    while pending:
        fut = pending.popleft()
        result = fut.result()     # block until this (oldest) future is done
        yield result
        completed += 1
        if pbar:
            pbar.update(1)

        # Refill: submit one more task for each result consumed
        try:
            args = next(it)
            pending.append(executor.submit(fn, args))
        except StopIteration:
            pass

    if pbar:
        pbar.close()


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


def _save_valid_samples_cache(samples: List[Sample], cache_path: str, report: PipelineReport, logger: logging.Logger):
    """Persist validated samples + report counters so resume can skip discovery+validation."""
    with open(cache_path, "w") as f:
        # First line: report counters
        header = {
            "_type": "report",
            "total_found": report.total_found,
            "valid": report.valid,
            "removed_missing": report.removed_missing,
            "removed_corrupted": report.removed_corrupted,
            "removed_small_mask": report.removed_small_mask,
            "removed_invalid_label": report.removed_invalid_label,
        }
        f.write(json.dumps(header, separators=(",", ":")) + "\n")
        for s in samples:
            rec = {
                "image_path": s.image_path,
                "mask_path": s.mask_path,
                "label_path": s.label_path,
                "category": s.category,
                "base_name": s.base_name,
                "width": s.width,
                "height": s.height,
            }
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    logger.info(f"Saved validation cache ({len(samples):,} samples) → {cache_path}")


def _load_valid_samples_cache(cache_path: str, logger: logging.Logger) -> Tuple[Optional[List[Sample]], Optional[dict]]:
    """Load validated samples from cache. Returns (samples, report_counters) or (None, None)."""
    if not os.path.exists(cache_path):
        return None, None
    try:
        samples: List[Sample] = []
        report_data = None
        with open(cache_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("_type") == "report":
                    report_data = rec
                    continue
                img_path = rec.get("image_path", "")
                if not (img_path and os.path.exists(img_path)):
                    continue
                samples.append(Sample(
                    image_path=img_path,
                    mask_path=rec.get("mask_path", ""),
                    label_path=rec.get("label_path", ""),
                    category=rec["category"],
                    base_name=rec["base_name"],
                    width=int(rec.get("width", 0)),
                    height=int(rec.get("height", 0)),
                ))
        if not samples:
            return None, None
        logger.info(f"Loaded validation cache: {len(samples):,} samples")
        return samples, report_data
    except Exception as e:
        logger.warning(f"Could not load validation cache {cache_path}: {e}")
        return None, None


def _load_augmented_manifest(manifest_path: str, logger: logging.Logger) -> List[Sample]:
    """Load augmented samples saved from a previous run."""
    samples: List[Sample] = []
    if not os.path.exists(manifest_path):
        return samples

    skipped = 0
    try:
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                img_path = rec.get("image_path", "")
                mask_path = rec.get("mask_path", "")
                if not (img_path and mask_path):
                    skipped += 1
                    continue
                # Check files still exist on disk (prev run may have been cleaned up)
                if not (os.path.exists(img_path) and os.path.exists(mask_path)):
                    skipped += 1
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

    logger.info(f"Loaded {len(samples):,} augmented samples from manifest"
                + (f" (skipped {skipped} missing)" if skipped else ""))
    return samples


def _stream_jsonl_samples(file_path: str):
    """
    Generator — yields (img_path, mask_path, label_path, category, base_name, w, h, polygons)
    one record at a time directly from a JSONL file (valid_cache or aug_manifest).
    Never builds a list; peak extra memory = one parsed dict per iteration.
    """
    if not os.path.exists(file_path):
        return
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("_type") == "report":   # header line in valid cache
                    continue
                img = rec.get("image_path", "")
                if not img:
                    continue
                yield (
                    img,
                    rec.get("mask_path", ""),
                    rec.get("label_path", ""),
                    rec.get("category", ""),
                    rec.get("base_name", ""),
                    int(rec.get("width", 0)),
                    int(rec.get("height", 0)),
                    rec.get("polygons"),
                )
            except Exception:
                continue


def _read_cache_header(cache_path: str, logger: logging.Logger) -> Optional[dict]:
    """Read only the first (report) line from the validation cache — O(1) memory."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r") as f:
            first_line = f.readline().strip()
        if first_line:
            rec = json.loads(first_line)
            if rec.get("_type") == "report":
                return rec
    except Exception as e:
        logger.warning(f"Could not read cache header {cache_path}: {e}")
    return None


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


def _write_polygons_as_yolo(label_path: str, polygons: List[List[float]], W: int, H: int):
    """Write absolute polygons to YOLO-seg txt (normalized coords, class_id=0)."""
    with open(label_path, "w") as f:
        for poly in polygons:
            if len(poly) < 6:
                continue
            parts = ["0"]
            for i in range(0, len(poly), 2):
                x = max(0.0, min(float(poly[i]) / max(W, 1), 1.0))
                y = max(0.0, min(float(poly[i + 1]) / max(H, 1), 1.0))
                parts.append(f"{x:.6f}")
                parts.append(f"{y:.6f}")
            f.write(" ".join(parts) + "\n")


def _augment_worker(args) -> Tuple[bool, str, Optional[dict]]:
    """
    Per-sample augmentation worker.
    Returns (success, img_path, result_dict | None).
    result_dict: {aug_img_path, aug_mask_path, aug_label_path, W, H}
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
        aug_label_path = os.path.join(out_dir, f"{aug_base}.txt")

        aug_img_bgr = cv2.cvtColor(aug_img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(aug_img_path,  aug_img_bgr)
        cv2.imwrite(aug_mask_path, aug_mask)
        _write_polygons_as_yolo(aug_label_path, aug_polygons, aug_W, aug_H)

        return True, img_path, {
            "aug_img_path":  aug_img_path,
            "aug_mask_path": aug_mask_path,
            "aug_label_path": aug_label_path,
            "category":      category,
            "base_name":     aug_base,
            "W": aug_W,
            "H": aug_H,
        }
    except Exception:
        return False, img_path, None


def augment_samples(
    valid_cache_path: str,
    cfg: Config,
    out_root: str,
    logger: logging.Logger,
    report: PipelineReport,
    resume: bool = False,
) -> None:
    """
    Stratified selection + parallel augmentation.
    Streams from valid_cache_path (never loads all samples into a list).
    Writes augmented records to _augmented_manifest.jsonl.
    Updates report.augmented in-place.
    """
    if not HAS_ALBUMENTATIONS:
        logger.warning("albumentations not installed. Skipping augmentation.")
        return

    manifest_path = os.path.join(os.path.dirname(out_root), "_augmented_manifest.jsonl")
    done_source_keys: Set[Tuple[str, str]] = set()

    if resume and os.path.exists(manifest_path):
        # Stream manifest — only need (category, base_name) pairs
        for _, _, _, cat, base, _, _, _ in _stream_jsonl_samples(manifest_path):
            src_base = base[4:] if base.startswith("aug_") else base
            done_source_keys.add((cat, src_base))
        if done_source_keys:
            logger.info(f"Resume mode: {len(done_source_keys):,} augmentation items already completed")

    # ── Pass 1: stream cache → per-category base_name lists (strings only, ~12 MB) ──
    random.seed(cfg.seed)
    by_cat_names: Dict[str, List[str]] = defaultdict(list)
    for _, _, _, cat, base, _, _, _ in _stream_jsonl_samples(valid_cache_path):
        by_cat_names[cat].append(base)

    # Stratified selection — store only (cat, base_name) pairs
    selected: Set[Tuple[str, str]] = set()
    for cat, names in by_cat_names.items():
        n = max(1, int(len(names) * cfg.aug_ratio))
        picks = random.sample(names, min(n, len(names)))
        for base in picks:
            if (cat, base) not in done_source_keys:
                selected.add((cat, base))

    total_tasks = len(selected)
    total_already_done = len(done_source_keys)
    del by_cat_names, done_source_keys
    _release_memory()

    logger.info(
        f"Augmenting {total_tasks:,} new samples "
        f"({cfg.aug_ratio * 100:.0f}% per category)"
        + (f" — {total_already_done:,} already done (resume)" if total_already_done else "")
    )

    if total_tasks == 0:
        logger.info("All augmentation items already completed.")
        report.augmented = total_already_done
        return

    # ── Pass 2: stream cache again, yield worker args only for selected samples ──
    # 'selected' is a small set (~36K tuples). The generator yields one dict at
    # a time — no large list is ever held in memory.
    def _worker_args_iter():
        for img, mask, label, cat, base, _, _, _ in _stream_jsonl_samples(valid_cache_path):
            if (cat, base) in selected:
                yield (
                    {
                        "image_path": img,
                        "mask_path":  mask,
                        "label_path": label,
                        "category":   cat,
                        "base_name":  base,
                        "aug_seed":   cfg.seed + hash(base) % (2**31),
                    },
                    out_root,
                    cfg.min_mask_pixels,
                )

    ok_count = 0
    aug_workers = max(1, min(cfg.aug_workers, cfg.workers))
    logger.info(f"Augmentation workers: {aug_workers}")
    mp_ctx = multiprocessing.get_context("forkserver")
    with open(manifest_path, "a") as manifest_f:
        with ProcessPoolExecutor(max_workers=aug_workers, mp_context=mp_ctx) as ex:
            for ok, _, result in _bounded_map(
                ex, _augment_worker, _worker_args_iter(),
                max_inflight=aug_workers * 4,
                desc="Augmenting", total=total_tasks,
            ):
                if ok and result:
                    ok_count += 1
                    rec = {
                        "image_path": result["aug_img_path"],
                        "mask_path":  result["aug_mask_path"],
                        "label_path": result["aug_label_path"],
                        "category":   result["category"],
                        "base_name":  result["base_name"],
                        "width":      result["W"],
                        "height":     result["H"],
                        "polygons":   None,
                    }
                    manifest_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                    manifest_f.flush()

    report.augmented = total_already_done + ok_count
    logger.info(
        f"Augmentation done: +{ok_count:,} new / {total_tasks:,} requested; "
        f"total: {report.augmented:,}"
    )


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
    sample_iter,                    # iterator of (img, label, cat, base, w, h, polygons)
    total: int,                     # number of items (for tqdm)
    categories: List[Dict],
    cat_name_to_id: Dict[str, int],
    workers: int,
    json_path: str,
    desc: str,
) -> Tuple[int, int]:
    """
    Memory-safe COCO writer.
    Consumes sample_iter one item at a time — no list ever held in RAM.
    Streams per-image and per-annotation records to JSONL temp files,
    then assembles final COCO JSON.
    Returns (num_images, num_annotations).
    """
    images_jsonl = f"{json_path}.images.jsonl"
    anns_jsonl   = f"{json_path}.anns.jsonl"

    image_id   = 1
    ann_id     = 1
    num_images = 0
    num_anns   = 0

    mp_ctx = multiprocessing.get_context("forkserver")
    with open(images_jsonl, "w") as img_f, open(anns_jsonl, "w") as ann_f:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
            for result in _bounded_map(
                ex, _build_coco_worker, sample_iter,
                max_inflight=workers * 8,
                desc=desc, total=total,
            ):
                if result is None:
                    continue

                img_rec = {
                    "id":       image_id,
                    "file_name": result["file_name"],
                    "width":    result["width"],
                    "height":   result["height"],
                    "category": result["category"],
                }
                img_f.write(_json_dumps_line(img_rec) + "\n")
                num_images += 1

                for ann in result["annotations"]:
                    ann_rec = {
                        "id":           ann_id,
                        "image_id":     image_id,
                        "category_id":  cat_name_to_id[ann["category_name"]],
                        "segmentation": ann["segmentation"],
                        "bbox":         ann["bbox"],
                        "area":         ann["area"],
                        "iscrowd":      0,
                    }
                    ann_f.write(_json_dumps_line(ann_rec) + "\n")
                    ann_id   += 1
                    num_anns += 1

                image_id += 1

    _assemble_coco_json(json_path, images_jsonl, anns_jsonl, categories)
    os.remove(images_jsonl)
    os.remove(anns_jsonl)
    return num_images, num_anns


def _copy_image_worker(args) -> bool:
    """Link image into output tree — symlink (zero space), hardlink, or copy."""
    img_src, img_dst = args
    try:
        if not os.path.exists(img_src):
            return False
        # If destination already exists (re-run), skip
        if os.path.exists(img_dst):
            return True
        dst_dir = os.path.dirname(img_dst)
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
        try:
            os.symlink(os.path.abspath(img_src), img_dst)  # symlink = zero disk space
        except (OSError, PermissionError):
            try:
                os.link(img_src, img_dst)                  # hardlink = shares inode
            except (OSError, PermissionError):
                shutil.copy2(img_src, img_dst)             # copy = last resort
        return True
    except Exception:
        return False


def build_dataset(
    valid_cache_path: str,
    aug_manifest_path: Optional[str],
    output_dir: str,
    cfg: Config,
    logger: logging.Logger,
    report: PipelineReport,
):
    """
    Main COCO dataset builder — fully streaming, never holds all samples in RAM.

    Data flow (all via _stream_jsonl_samples — one record at a time):
      Pass 1 : collect {category: [(base_name, is_aug)]} — ~30 MB of strings
      Compute : stratified split → val_set (set of (cat, base_name) tuples, ~7 MB)
      Pass 2 : stream → copy images
      Pass 3 : stream → COCO train annotations
      Pass 4 : stream → COCO val annotations
    """
    _release_memory()
    logger.info("Building COCO dataset (streaming from cache files) ...")
    random.seed(cfg.seed)

    has_aug = bool(aug_manifest_path and os.path.exists(aug_manifest_path))

    # ── Pass 1: lightweight string-only index for stratified split ────────
    # Store only (base_name, is_aug) per category — NOT Sample objects.
    by_cat: Dict[str, List[Tuple[str, bool]]] = defaultdict(list)
    for _, _, _, cat, base, _, _, _ in _stream_jsonl_samples(valid_cache_path):
        by_cat[cat].append((base, False))
    if has_aug:
        for _, _, _, cat, base, _, _, _ in _stream_jsonl_samples(aug_manifest_path):
            by_cat[cat].append((base, True))

    all_cats       = sorted(by_cat.keys())
    cat_name_to_id = {name: idx + 1 for idx, name in enumerate(all_cats)}
    categories     = [{"id": cat_name_to_id[n], "name": n, "supercategory": "fashion"}
                      for n in all_cats]
    logger.info(f"Categories: {len(all_cats)}")

    val_set:     Set[Tuple[str, str]] = set()
    total_count = 0
    for cat, entries in by_cat.items():
        shuffled = entries[:]
        random.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * cfg.val_split))
        for base, _ in shuffled[:n_val]:
            val_set.add((cat, base))
        report.by_category[cat] = len(entries)
        total_count += len(entries)

    del by_cat
    _release_memory()

    total_val   = len(val_set)
    total_train = total_count - total_val
    report.train_count = total_train
    report.val_count   = total_val
    logger.info(f"Split: {total_train:,} train | {total_val:,} val")

    # ── Pre-create output directories (97 cats × 2 splits = 194 dirs) ────
    for split in ("train", "val"):
        for cat in all_cats:
            os.makedirs(os.path.join(output_dir, "images", split, cat), exist_ok=True)

    # ── Helper: unified stream across both source files ───────────────────
    def _all_records():
        yield from _stream_jsonl_samples(valid_cache_path)
        if has_aug:
            yield from _stream_jsonl_samples(aug_manifest_path)

    # ── Pass 2: copy images (hardlink where possible) ─────────────────────
    def _copy_arg_iter():
        for img, _, _, cat, base, _, _, _ in _all_records():
            split = "val" if (cat, base) in val_set else "train"
            ext   = Path(img).suffix
            dst   = os.path.join(output_dir, "images", split, cat, f"{base}{ext}")
            yield (img, dst)

    mp_ctx = multiprocessing.get_context("forkserver")
    ok = 0
    with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=mp_ctx) as ex:
        for success in _bounded_map(
            ex, _copy_image_worker, _copy_arg_iter(),
            max_inflight=cfg.workers * 32,
            desc="Copying images", total=total_count,
        ):
            if success:
                ok += 1
    logger.info(f"Copied {ok:,}/{total_count:,} images")
    _release_memory()

    # ── Pass 3 + 4: COCO annotation JSONs ────────────────────────────────
    ann_dir = os.path.join(output_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    coco_workers = max(1, min(cfg.coco_workers, cfg.workers))
    logger.info(f"COCO workers: {coco_workers}")

    for split in ("train", "val"):
        is_val   = (split == "val")
        n_split  = total_val if is_val else total_train
        json_out = os.path.join(ann_dir, f"instances_{split}.json")
        logger.info(f"Building COCO annotations for {split} ({n_split:,} samples) ...")

        # Generator: yields 7-tuples matching _build_coco_worker's arg order
        def _split_iter(split_is_val=is_val):
            for img, _, label, cat, base, w, h, poly in _all_records():
                if ((cat, base) in val_set) == split_is_val:
                    yield (img, label, cat, base, w, h, poly)

        n_imgs, n_anns = write_coco_split_streaming(
            sample_iter    = _split_iter(),
            total          = n_split,
            categories     = categories,
            cat_name_to_id = cat_name_to_id,
            workers        = coco_workers,
            json_path      = json_out,
            desc           = f"COCO {split}",
        )
        logger.info(f"  ↳ {json_out}  |  {n_imgs:,} images  |  {n_anns:,} annotations")
        _release_memory()


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
    p.add_argument("--clean",           action="store_true",
                   help="Delete all caches, temp files, and previous output before running")
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

    # ── --clean: wipe everything and start fresh ──────────────────────────
    if args.clean:
        for name in (
            "_valid_samples_cache.jsonl",
            "_augmented_manifest.jsonl",
            "_augmented_temp",
            "images",
            "annotations",
            "pipeline_report.json",
        ):
            target = os.path.join(cfg.output_dir, name)
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            elif os.path.isfile(target):
                os.remove(target)
        print(f"Cleaned output directory: {cfg.output_dir}")

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

    # ── 1 & 2. Discover + Validate (or load cache on resume) ─────────────
    valid_cache_path = os.path.join(cfg.output_dir, "_valid_samples_cache.jsonl")
    using_cache = False

    if args.resume and os.path.exists(valid_cache_path):
        cached_report = _read_cache_header(valid_cache_path, logger)
        if cached_report:
            report.total_found          = cached_report.get("total_found", 0)
            report.valid                = cached_report.get("valid", 0)
            report.removed_missing      = cached_report.get("removed_missing", 0)
            report.removed_corrupted    = cached_report.get("removed_corrupted", 0)
            report.removed_small_mask   = cached_report.get("removed_small_mask", 0)
            report.removed_invalid_label = cached_report.get("removed_invalid_label", 0)
            completed_stages = 2
            logger.info(
                f"[Stage 1-2/6] Using validation cache "
                f"({report.valid:,} samples — skipped discovery+validation)"
            )
            using_cache = True

    if not using_cache:
        t_stage = time.time()
        samples = discover_samples(cfg.input_dir, logger)
        if not samples:
            logger.error("No samples found. Check --input path.")
            sys.exit(1)
        _stage_done("Discovery", t_stage)

        # ── Debug limit ───────────────────────────────────────────────────
        if args.limit > 0:
            random.seed(cfg.seed)
            random.shuffle(samples)
            samples = samples[:args.limit]
            logger.info(f"Debug mode: limited to {len(samples):,} samples")

        t_stage = time.time()
        valid_samples = validate_samples(samples, cfg, logger, report)
        if not valid_samples:
            logger.error("No valid samples after validation.")
            sys.exit(1)
        _stage_done("Validation", t_stage)

        # Save cache, then free immediately — everything downstream streams
        _save_valid_samples_cache(valid_samples, valid_cache_path, report, logger)
        del valid_samples, samples
        _release_memory()

    if args.dry_run:
        logger.info("Dry run complete. No files copied.")
        report_path = os.path.join(cfg.output_dir, "pipeline_report.json")
        report.save(report_path)
        logger.info(f"Report saved: {report_path}")
        return

    # ── 3. Augment (streams from cache file — no sample list in RAM) ──────
    t_stage = time.time()
    aug_manifest_path = os.path.join(cfg.output_dir, "_augmented_manifest.jsonl")
    if not args.no_aug:
        aug_temp = os.path.join(cfg.output_dir, "_augmented_temp")
        augment_samples(valid_cache_path, cfg, aug_temp, logger, report, resume=args.resume)
    else:
        logger.info("Augmentation skipped (--no-aug)")
    _stage_done("Augmentation", t_stage)

    # ── 4. Build COCO dataset (streams from cache + manifest — no lists) ──
    t_stage = time.time()
    _release_memory()
    build_dataset(
        valid_cache_path,
        aug_manifest_path if (not args.no_aug and os.path.exists(aug_manifest_path)) else None,
        cfg.output_dir, cfg, logger, report,
    )
    _release_memory()
    _stage_done("COCO Build", t_stage)

    # ── 5. Cleanup temp files (only after COCO build succeeded) ────────────
    t_stage = time.time()
    aug_temp = os.path.join(cfg.output_dir, "_augmented_temp")
    for tmp_path in [aug_temp, aug_manifest_path, valid_cache_path]:
        if os.path.isdir(tmp_path):
            shutil.rmtree(tmp_path, ignore_errors=True)
            logger.info(f"Cleaned up {os.path.basename(tmp_path)}")
        elif os.path.isfile(tmp_path):
            os.remove(tmp_path)
            logger.info(f"Cleaned up {os.path.basename(tmp_path)}")
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