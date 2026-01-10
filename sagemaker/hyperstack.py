#!/usr/bin/env python3
import os
import json
import argparse
import subprocess
import shutil
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics.models.sam import SAM3SemanticPredictor
from huggingface_hub import hf_hub_download

# ------------- Defaults -------------
DEFAULT_INPUT_ROOT = "input_data"
DEFAULT_OUTPUT_ROOT = "output_data"
DEFAULT_MODEL_PATH = "sam3.pt"
DEFAULT_MAX_RES = 1024
DEFAULT_LOG_EVERY = 10

# S3 Defaults (Placeholder)
DEFAULT_S3_MODEL_URI = "s3://my-bucket/models/" # Not really used for manual HF download but kept for arg compat
DEFAULT_S3_INPUT_URI = "s3://my-bucket/input/"
DEFAULT_S3_SAMPLE_URI = "s3://my-bucket/sample_data/"
DEFAULT_S3_OUTPUT_URI = "s3://my-bucket/output/"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ------------- Utils -------------
def load_prompts(prompts_path: str) -> dict:
    if not os.path.exists(prompts_path):
        # Fallback if file not found, though user validation should catch this
        print(f"⚠️ Warning: prompts.json not found at {prompts_path}")
        return {}
    with open(prompts_path, "r", encoding="utf-8") as f:
        return json.load(f)

def mask_to_yolo_polygon(mask, width, height):
    """
    Convert a binary mask (uint8) to a YOLO polygon string.
    Format: <class_id> <x1> <y1> <x2> <y2> ...
    Normalized coordinates [0, 1].
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ""

    # Find the largest contour by area
    c = max(contours, key=cv2.contourArea)
    
    # Needs a minimum number of points
    if len(c) < 3:
        return ""

    # Normalize points
    polygon = []
    for point in c:
        x, y = point[0]
        polygon.append(f"{x / width:.6f}")
        polygon.append(f"{y / height:.6f}")
    
    return " ".join(polygon)

def download_sam3_model(model_path):
    """Manually downloads sam3.pt from Hugging Face if not present."""
    if os.path.exists(model_path):
        print(f"✅ Model found at {model_path}")
        return

    print(f"⬇️ Model {model_path} not found. Downloading from Hugging Face (facebook/sam3)...")
    try:
        # We download to current dir, which is usually where model_path points (e.g. "sam3.pt")
        # If model_path has a dir component, we should respect it, but typically it's just a filename.
        local_dir = os.path.dirname(model_path) or "."
        filename = os.path.basename(model_path)
        
        hf_hub_download(repo_id="facebook/sam3", filename=filename, local_dir=local_dir)
        print("✅ Download complete.")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print("Ensure you have set HF_TOKEN env var or run 'huggingface-cli login'.")
        print("Also ensure you accepted the license at https://huggingface.co/facebook/sam3")
        raise e

# ------------- Main Processor -------------

from collections import defaultdict

class Stats:
    def __init__(self):
        self.total = 0
        self.processed = 0
        self.skipped = 0

def process_batch(args, predictor, prompts_dict, device):
    stats = Stats()
    category_metadata = defaultdict(list)
    
    # Gather all images
    # We support flat directory or category-subdirectories. 
    # Logic: Walk input dir.
    
    image_files = []
    # (path, filename, category)
    
    for root, dirs, files in os.walk(args.input):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                full_path = os.path.join(root, file)
                
                # Infer category
                # 1. Check parent folder name
                parent_name = os.path.basename(root)
                category = parent_name if parent_name in prompts_dict else None
                
                # 2. If not found, check filename matching keys
                if not category:
                    for key in prompts_dict:
                        if file.startswith(key):
                            category = key
                            break
                
                if category:
                    image_files.append((full_path, file, category))
    
    stats.total = len(image_files)
    print(f"Found {stats.total} images to process.")

    for i, (img_path, filename, category) in enumerate(image_files):
        if i % args.log_every == 0:
            print(f"[{i}/{stats.total}] Processing {filename} ({category})...")

        prompt_text = prompts_dict.get(category)
        if not prompt_text:
            print(f"Skipping {filename}: No prompt found for category '{category}'")
            stats.skipped += 1
            continue

        base_name = os.path.splitext(filename)[0]
        
        # Output structure: preserve subdirectory structure relative to input? 
        # Or simple flat output as per original script logic which seemed to use output root directly or subdirs.
        # Let's mirror the input folder structure if it was inside a category folder, 
        # otherwise put it in category folder in output.
        
        # Simplified: Output to args.output/<category>/
        cat_out_dir = os.path.join(args.output, category)
        ensure_dir(cat_out_dir)
        
        out_mask_path = os.path.join(cat_out_dir, f"{base_name}.png")
        out_label_path = os.path.join(cat_out_dir, f"{base_name}.txt")
        out_overlay_path = os.path.join(cat_out_dir, f"{base_name}_overlay.jpg")
        
        if os.path.exists(out_mask_path) and not args.upload_output: # upload-output flag reused as "force overwrite" maybe?
             # For now, just continue if exists unless forced? Script doesn't say. Let's process.
             pass

        try:
            # --- SAM 3 Prediction ---
            # 1. Set Image
            # SAM3SemanticPredictor handles loading.
            predictor.set_image(img_path)
            
            # 2. Predict with text
            # The user snippet passes list: text=['blazer']
            # We can pass multiple if we wanted, but here we have 1 category prompt.
            results = predictor(text=[prompt_text])
            
            # results is a list of Results objects (one per prompt text? or one per image?)
            # Since we set_image once and passed 1 text prompt, we likely get 1 result object or list of objects found for that prompt.
            # Ultralytics results: result[0].masks.data is torch tensor
            
            final_mask = None
            
            if results and results[0].masks is not None:
                # Combine all found instances for this prompt into one binary mask
                # masks.data is (N, H, W) where N is number of objects found
                masks_tensor = results[0].masks.data # GPU tensor usually
                
                if masks_tensor.numel() > 0:
                    # Union of all masks
                    files_mask = torch.any(masks_tensor, dim=0).squeeze().cpu().numpy().astype(np.uint8) * 255
                    final_mask = files_mask
            
            if final_mask is None:
               # No detection
               stats.skipped += 1
               continue

            # --- Save Outputs ---
            
            # 1. Mask
            cv2.imwrite(out_mask_path, final_mask)
            
            original_img = cv2.imread(img_path)
            H, W = original_img.shape[:2]
            
            # 2. YOLO Label
            poly_str = mask_to_yolo_polygon(final_mask, W, H)
            if poly_str:
                # Class ID 0 for all? Or specific id? 
                # Originally we didn't have class mapping. Let's assume 0.
                with open(out_label_path, "w") as f:
                    f.write(f"0 {poly_str}\n")
            
            # 3. Visualization
            # Draw polygon on original image
            if poly_str:
                coords = [float(x) for x in poly_str.split()]
                # Rescale to pixels
                pts = []
                for j in range(0, len(coords), 2):
                    px = int(coords[j] * W)
                    py = int(coords[j+1] * H)
                    pts.append([px, py])
                
                pts = np.array(pts, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(original_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.imwrite(out_overlay_path, original_img)

            # 4. Copy Original Image (Ensure YOLO dataset completeness)
            out_image_path = os.path.join(cat_out_dir, f"{base_name}{os.path.splitext(filename)[1]}")
            if not os.path.exists(out_image_path):
                shutil.copy2(img_path, out_image_path)

            # 5. Metadata Collection
            # Extract confidence (max of detected objects)
            conf_score = 0.0
            try:
                if results[0].boxes is not None and results[0].boxes.conf is not None:
                    if results[0].boxes.conf.numel() > 0:
                        conf_score = float(results[0].boxes.conf.max().item())
            except Exception:
                pass # Default to 0.0 if extraction fails

            # Rule: < 0.60 -> human review
            human_review = conf_score < 0.60
            
            category_metadata[category].append({
                "image": filename,
                "confidence": round(conf_score, 4),
                "human_review_needed": human_review
            })

            stats.processed += 1
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            stats.skipped += 1
            continue

    # --- Write Metadata JSONs ---
    for cat, meta_list in category_metadata.items():
        if not meta_list:
            continue
        
        meta_path = os.path.join(args.output, cat, "metadata.json")
        try:
            # If exists, maybe load and append? For now, we overwrite or it's a batch job.
            # Let's assume batch job per run.
            with open(meta_path, "w") as f:
                json.dump(meta_list, f, indent=2)
            print(f"Saved metadata to {meta_path}")
        except Exception as e:
            print(f"Failed to save metadata for {cat}: {e}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # SageMaker / General Arguments
    parser.add_argument("--mode", type=str, default="process", choices=["process", "sample"], help="Execution mode")
    parser.add_argument("--download-model", action="store_true") # Kept for compat, triggers HF download
    parser.add_argument("--download-input", action="store_true") # Kept for compat (S3 download)
    parser.add_argument("--upload-output", action="store_true")
    
    parser.add_argument("--input", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "prompts.json"))
    
    parser.add_argument("--max-res", type=int, default=DEFAULT_MAX_RES)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    
    parser.add_argument("--s3-model-uri", default=DEFAULT_S3_MODEL_URI)
    parser.add_argument("--s3-input-uri", default=DEFAULT_S3_INPUT_URI)
    parser.add_argument("--s3-sample-uri", default=DEFAULT_S3_SAMPLE_URI)
    parser.add_argument("--s3-output-uri", default=DEFAULT_S3_OUTPUT_URI)
    
    # SAM 3 Configs
    parser.add_argument("--box-threshold", type=float, default=0.4, help="Confidence threshold (maps to conf in SAM3)")

    args = parser.parse_args()

    ensure_dir(args.input)
    ensure_dir(args.output)
    
    # 1. Download Model (if requested or missing)
    download_sam3_model(args.model)

    # 2. Download Input (Placeholder S3 logic - user should handle via s3 sync or similar for now if strictly local, 
    # but keeping original flow if they use the flags)
    # 2. Download Input (Placeholder S3 logic - user should handle via s3 sync or similar for now if strictly local, 
    # but keeping original flow if they use the flags)
    if args.download_input:
        if args.mode == "sample":
            print(f"Downloading samples from {args.s3_sample_uri}...")
            subprocess.call(["aws", "s3", "sync", args.s3_sample_uri, args.input, "--quiet"])
        else:
            print(f"Downloading inputs from {args.s3_input_uri}...")
            subprocess.call(["aws", "s3", "sync", args.s3_input_uri, args.input, "--quiet"])
    
    # 3. Load Prompts
    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} category prompts.")

    # 4. Initialize SAM 3 Predictor
    print(f"Initializing SAM 3 Predictor with model={args.model}...")
    overrides = dict(
        conf=args.box_threshold, # Mapping box_threshold to conf for consistency with old args
        task="segment",
        mode="predict",
        model=args.model,
        # half=True, # Optional: Float16
        save=False,  # We handle saving manually
    )
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        predictor = SAM3SemanticPredictor(overrides=overrides)
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        exit(1)

    # 5. Process
    stats = process_batch(args, predictor, prompts, device)
    
    print(f"\n--- Batch Complete ---")
    print(f"Total: {stats.total}")
    print(f"Processed: {stats.processed}")
    print(f"Skipped/Failed: {stats.skipped}")

    # 6. Upload Output
    if args.upload_output and args.s3_output_uri != DEFAULT_S3_OUTPUT_URI:
        print(f"Uploading output to {args.s3_output_uri}...")
        subprocess.call(["aws", "s3", "sync", args.output, args.s3_output_uri, "--quiet"])
    
    print("Done!")
