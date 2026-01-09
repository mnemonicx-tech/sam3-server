#!/usr/bin/env python3
import os
import json
import argparse
import subprocess
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import wget
from PIL import Image

# Grounding DINO imports
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

# ------------- GroundingDINO Defaults -------------
GD_CONFIG_PATH = "GroundingDINO_SwinT_OGC.py" # Keep it simple in CWD or handle download
GD_WEIGHTS_PATH = "groundingdino_swint_ogc.pth"
GD_WEIGHTS_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
GD_CONFIG_URL = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

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


def load_grounding_dino():
    # Only download if missing
    if not os.path.exists(GD_CONFIG_PATH):
        print(f"⬇️ Downloading GroundingDINO config from {GD_CONFIG_URL}...")
        wget.download(GD_CONFIG_URL, GD_CONFIG_PATH)
    
    if not os.path.exists(GD_WEIGHTS_PATH):
        print(f"⬇️ Downloading GroundingDINO weights from {GD_WEIGHTS_URL}...")
        wget.download(GD_WEIGHTS_URL, GD_WEIGHTS_PATH)
        
    print(f"🔹 Loading GroundingDINO...")
    args = SLConfig.fromfile(GD_CONFIG_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(GD_WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model.to(device)


def transform_image_for_gd(image_pil):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image


def get_boxes_from_text(model, image, caption, box_threshold, text_threshold, device):
    with torch.no_grad():
        outputs = model(image[None].to(device), captions=[caption])
    
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # Filter by threshold
    filt_mask = logits.max(dim=1)[0] > box_threshold
    boxes_filt = boxes[filt_mask]
    
    return boxes_filt


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

    # ---------------- Load models ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Device: {device}")

    print("🚀 Loading GroundingDINO model...")
    gd_model = load_grounding_dino()

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

            # Load image
            image_cv = cv2.imread(img_path)
            if image_cv is None:
                continue
            
            # Prepare image for SAM (RGB)
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            
            # Prepare image for GroundingDINO (PIL -> Tensor)
            image_pil = Image.fromarray(image_rgb)
            image_gd_tensor = transform_image_for_gd(image_pil)
            
            # 1. GroundingDINO: Text -> Boxes
            boxes_filt = get_boxes_from_text(gd_model, image_gd_tensor, prompt, BOX_THRESHOLD, TEXT_THRESHOLD, device)
            
            if len(boxes_filt) == 0:
                # No object found matching prompt
                continue

            # Convert boxes (cx, cy, w, h) norm -> (x1, y1, x2, y2) pixel
            H, W = image_rgb.shape[:2]
            boxes_xyxy = boxes_filt * torch.Tensor([W, H, W, H])
            boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
            boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]

            # 2. SAM: Image + Boxes -> Mask
            # We resize for SAM after getting boxes? 
            # Actually efficient way: Resize first? 
            # Note: The original script resized first. But DINO needs original quality usually.
            # Let's keep original resolution for DINO detection, then pass to SAM.
            # SAM handles resizing internally usually via predictor.set_image.
            # If we want to use the resized logic from before, we need to resize boxes too.
            # For simplicity and accuracy, let's use original resolution (SAM is heavy but fits on A4000).
            # If OOM, we can resize. A4000 has 16GB. 1024x1024 is fine.
            
            predictor.set_image(image_rgb)
            
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy.to(device), image_rgb.shape[:2])

            with torch.cuda.amp.autocast(dtype=torch.float16):
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes, # Use all boxes
                    multimask_output=False,
                )
            
            # Combine masks if multiple boxes found (e.g. multiple cargo pants pockets?)
            # Or usually we want one mask. 
            # Let's take the union of all masks.
            if masks.shape[0] > 0:
                final_mask = torch.any(masks, dim=0).squeeze().cpu().numpy().astype(np.uint8) * 255
                
                # Resize only for saving if needed? No, save original resolution mask.
            else:
                continue

            cv2.imwrite(out_path, final_mask)

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
