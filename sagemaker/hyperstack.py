#!/usr/bin/env python3
import os
import json
import argparse
import subprocess
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# ------------- Defaults -------------
DEFAULT_MODEL_PATH = "./models/sam3.pt"
DEFAULT_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "prompts.json")
DEFAULT_MAX_RES = 1024
DEFAULT_LOG_EVERY = 500
DEFAULT_INPUT_ROOT = "./input"
DEFAULT_OUTPUT_ROOT = "./output"

DEFAULT_S3_MODEL_URI = "s3://kishankumarhs/sam3-models/sam3.pt"
DEFAULT_S3_INPUT_URI = "s3://kishankumarhs/node_downloads/"
DEFAULT_S3_SAMPLE_URI = "s3://kishankumarhs/sample_data/"
DEFAULT_S3_OUTPUT_URI = "s3://kishankumarhs/sam3_output/"
# -----------------------------------


def run(cmd: str):
    print(f"🔧 {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def resize_image(img, max_side=1024):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def list_images(folder: str):
    exts = (".jpg", ".jpeg", ".png", ".webp")
    for f in os.listdir(folder):
        if f.lower().endswith(exts):
            yield f


def load_prompts(prompts_path: str) -> dict:
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"❌ prompts.json not found: {prompts_path}")
    with open(prompts_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser("Hyperstack SAM3 Batch Segmentation (prompts.json)")
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--download-input", action="store_true")
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--upload-output", action="store_true")

    parser.add_argument("--input", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS_PATH)

    parser.add_argument("--max-res", type=int, default=DEFAULT_MAX_RES)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)

    parser.add_argument("--s3-model-uri", default=DEFAULT_S3_MODEL_URI)
    parser.add_argument("--s3-input-uri", default=DEFAULT_S3_INPUT_URI)
    parser.add_argument("--s3-sample-uri", default=DEFAULT_S3_SAMPLE_URI)
    parser.add_argument("--s3-output-uri", default=DEFAULT_S3_OUTPUT_URI)

    args = parser.parse_args()

    ensure_dir(args.input)
    ensure_dir(args.output)
    ensure_dir(os.path.dirname(args.model) or ".")

    # ✅ Load prompts.json
    prompts = load_prompts(args.prompts)
    print(f"✅ Loaded prompts.json keys: {len(prompts)}")

    # ---------------- Model weights ----------------
    if args.download_model and (not os.path.exists(args.model)):
        print(f"⬇️ Downloading model weights from: {args.s3_model_uri}")
        run(f"aws s3 cp {args.s3_model_uri} {args.model}")

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"❌ Model file not found at {args.model}. Use --download-model or copy sam3.pt"
        )

    # ---------------- Input dataset ----------------
    if args.download_input:
        s3_uri = args.s3_sample_uri if args.mode == "sample" else args.s3_input_uri
        print(f"⬇️ Syncing input from: {s3_uri}")
        run(f"aws s3 sync {s3_uri} {args.input} --no-progress")

    # ---------------- Load model ----------------
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Device: {device}")

    print("🚀 Loading SAM model...")
    sam = sam_model_registry["vit_h"](checkpoint=args.model)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    # Stats tracker
    class Stats:
        def __init__(self):
            self.total = 0
            self.processed = 0
            self.skipped = 0

    stats = Stats()

    def process_batch(img_list, current_input_dir, current_output_dir, prompt):
        print(f"\n📂 Category: {os.path.basename(current_output_dir)} | Images: {len(img_list)}")
        
        for img_name in img_list:
            stats.total += 1
            img_path = os.path.join(current_input_dir, img_name)

            base = os.path.splitext(img_name)[0]
            out_path = os.path.join(current_output_dir, f"{base}_mask.png")

            if os.path.exists(out_path):
                stats.skipped += 1
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue

            image = resize_image(image, args.max_res)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            predictor.set_image(image)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=None,
                    text_prompt=prompt,
                    multimask_output=False,
                )

            mask = (masks[0] * 255).astype(np.uint8)
            cv2.imwrite(out_path, mask)

            stats.processed += 1

            if stats.processed % args.log_every == 0:
                print(f"✅ Processed={stats.processed} | Skipped={stats.skipped} | Total={stats.total}")
                torch.cuda.empty_cache()

    # ---------------- Main loop ----------------
    # 1. Identify all files in input root and group them if they match known prompts
    flat_images = {}
    input_items = sorted(os.listdir(args.input))
    prompt_keys = sorted(prompts.keys(), key=len, reverse=True)

    for item in input_items:
        item_path = os.path.join(args.input, item)
        
        # If directory -> Use existing logic
        if os.path.isdir(item_path):
            category = item
            prompt = prompts.get(category)
            if not prompt:
                print(f"⚠️ No prompt for directory='{category}' -> skipping")
                continue

            cat_output = os.path.join(args.output, category)
            ensure_dir(cat_output)

            img_list = list(sorted(list_images(item_path)))
            if img_list:
                process_batch(img_list, item_path, cat_output, prompt)
           
        # If file -> Try to infer category
        elif os.path.isfile(item_path):
            if not any(item.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                continue

            # Find matching category
            matched_cat = None
            for key in prompt_keys:
                if item.startswith(key):
                    matched_cat = key
                    break
            
            if matched_cat:
                if matched_cat not in flat_images:
                    flat_images[matched_cat] = []
                flat_images[matched_cat].append(item)

    # 2. Process gathered flat images
    for category, img_list in flat_images.items():
        prompt = prompts.get(category)
        if not prompt:
            continue
            
        cat_output = os.path.join(args.output, category)
        ensure_dir(cat_output)
        
        # For flat files, input dir is just args.input
        process_batch(img_list, args.input, cat_output, prompt)

    print("\n✅ SAM batch segmentation completed")
    print(f"Total seen: {stats.total}")
    print(f"Processed: {stats.processed}")
    print(f"Skipped: {stats.skipped}")

    # ---------------- Upload output ----------------
    if args.upload_output:
        print(f"⬆️ Uploading masks to S3: {args.s3_output_uri}")
        run(f"aws s3 sync {args.output} {args.s3_output_uri} --no-progress")
        print("✅ Upload complete")


if __name__ == "__main__":
    main()
