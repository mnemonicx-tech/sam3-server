import os
import json
import cv2
import torch
import numpy as np
import wget
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

# Grounding DINO imports
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

# ---------------- CONFIG ----------------

# ---------------- CONFIG ----------------
SAM_CHECKPOINT = "/opt/ml/model/sam3.pt"
GD_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" 
GD_WEIGHTS_PATH = "groundingdino_swint_ogc.pth"
GD_WEIGHTS_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

INPUT_ROOT = os.getenv("INPUT_ROOT", "/opt/ml/input/data")
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "/opt/ml/output/data")

# Locate prompts.json relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_FILE = os.path.join(SCRIPT_DIR, "prompts.json")

IMAGE_SIZE = 1024
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------

def load_grounding_dino(config_path, model_path):
    print(f"🔹 Loading GroundingDINO from {model_path}...")
    if not os.path.exists(model_path):
        print(f"⬇️ Downloading weights from {GD_WEIGHTS_URL}...")
        wget.download(GD_WEIGHTS_URL, model_path)
    
    args = SLConfig.fromfile(config_path)
    args.device = DEVICE
    model = build_model(args)
    checkpoint = torch.load(model_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(f"✅ GroundingDINO Loaded: {load_res}")
    model.eval()
    return model.to(DEVICE)

def transform_image_for_gd(image_pil):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image.to(DEVICE)

def get_boxes_from_text(model, image, caption, box_threshold, text_threshold):
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # Filter by threshold
    filt_mask = logits.max(dim=1)[0] > box_threshold
    boxes_filt = boxes[filt_mask]
    
    return boxes_filt

def mask_to_yolo_polygon(mask, width, height):
    """
    Convert binary mask to YOLO polygon format:
    <class_id> <x1> <y1> <x2> <y2> ...
    Normalized coordinates.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    
    for contour in contours:
        if cv2.contourArea(contour) < 200: # Filter small noise
            continue
            
        contour = contour.flatten().astype(float)
        
        # Normalize
        contour[0::2] /= width
        contour[1::2] /= height
        
        # Limit precision
        contour = np.round(contour, 4)
        polygons.append(contour.tolist())
        
    return polygons

# ---------------- MAIN ----------------
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 1. Load Prompts
    if os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE, 'r') as f:
            PROMPTS = json.load(f)
    else:
        print("⚠️ prompts.json not found! Using default empty map.")
        PROMPTS = {}

    # 2. Load Models
    print(f"🔹 Device: {DEVICE}")
    
    # Check if we need to clone GD (if running locally or without dockerfile setup yet)
    if not os.path.exists("GroundingDINO"):
         print("⚠️ GroundingDINO repo not found in current dir. Assuming package installed via pip.")
         # Since we installed via pip git+, the config might be tricky to find. 
         # However, we can use the library's config if available.
         # For this script, we'll assume the repo structure if possible, OR 
         # we fallback to a known config location if installed.
         # A robust way is to rely on the user-provided setup or S3.
         pass
         
    # Locate Config (assuming Dockerfile `git clone` or manual placement)
    # The Dockerfile installs it as a package `pip install git+...`
    # But we need the config file `GroundingDINO_SwinT_OGC.py`.
    # It's usually inside site-packages/groundingdino/config/ if included, 
    # but often config files aren't packaged.
    # We might need to download the config too.
    
    if not os.path.exists(GD_CONFIG_PATH):
        print("⚠️ Config not found at path. Attempting to fetch standard config...")
        config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        os.makedirs(os.path.dirname(GD_CONFIG_PATH), exist_ok=True)
        wget.download(config_url, GD_CONFIG_PATH)

    gd_model = load_grounding_dino(GD_CONFIG_PATH, GD_WEIGHTS_PATH)

    print("🔹 Loading SAM Model...")
    # NOTE: Using 'vit_b' but loading 'sam3.pt'. Assuming architecture matches.
    # If sam3.pt is huge, might be vit_h or vit_l.
    # We will try vit_b first as per original script.
    
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
    sam = sam.to(DEVICE)
    predictor = SamPredictor(sam)

    total = 0
    processed = 0
    
    print("🔹 Starting pipeline...")

    for category in sorted(os.listdir(INPUT_ROOT)):
        cat_input = os.path.join(INPUT_ROOT, category)
        if not os.path.isdir(cat_input):
            continue

        cat_output = os.path.join(OUTPUT_ROOT, category)
        os.makedirs(cat_output, exist_ok=True)
        
        # Get Text Prompt
        text_prompt = PROMPTS.get(category)
        if not text_prompt:
            print(f"⚠️ No prompt for category: {category}. Skipping.")
            continue
            
        print(f"\n📁 Category: {category} | Prompt: \"{text_prompt}\"")

        for img_name in sorted(os.listdir(cat_input)):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            total += 1
            in_path = os.path.join(cat_input, img_name)
            base_name = img_name.rsplit(".", 1)[0]
            mask_out_path = os.path.join(cat_output, base_name + "_mask.png")
            label_out_path = os.path.join(cat_output, base_name + ".txt")

            if os.path.exists(mask_out_path):
                continue
            
            # Load Image
            image_cv = cv2.imread(in_path)
            if image_cv is None: continue
            original_h, original_w = image_cv.shape[:2]
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)

            # 1. Grounding DINO -> Box
            img_tensor = transform_image_for_gd(image_pil)
            boxes_filt = get_boxes_from_text(gd_model, img_tensor, text_prompt, BOX_THRESHOLD, TEXT_THRESHOLD)
            
            if len(boxes_filt) == 0:
                print(f"  ❌ No object found for {img_name}")
                continue
                
            # Convert boxes (cx, cy, w, h) norm -> (x1, y1, x2, y2) pixel
            H, W = original_h, original_w
            boxes_xyxy = boxes_filt * torch.Tensor([W, H, W, H]).to(DEVICE)
            boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
            boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]
            
            # 2. SAM -> Mask (using the first/best box)
            predictor.set_image(image_rgb)
            
            # Use the box with highest confidence (DINO sorts by conf? usually yes)
            # or use all boxes. For "item", usually one main item.
            # We'll stick to the first one for now to avoid multi-mask chaos unless requested.
            transformed_box = predictor.transform.apply_boxes_torch(boxes_xyxy[0:1], (H, W))
            
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_box,
                multimask_output=False,
            )
            
            mask = masks[0, 0].cpu().numpy().astype(np.uint8) * 255
            
            # Save Mask
            Image.fromarray(mask).save(mask_out_path)
            
            # 3. YOLO Label Generation (Polygon)
            polygons = mask_to_yolo_polygon(mask, W, H)
            if polygons:
                with open(label_out_path, 'w') as f:
                    # Class ID 0 for all for now, or map category to ID if needed.
                    # Assuming 1 class per folder implies single class dataset or we need a map.
                    # Start with 0.
                    for poly in polygons:
                        line = "0 " + " ".join(map(str, poly))
                        f.write(line + "\n")
            
            processed += 1
            if processed % 10 == 0:
                print(f"✅ Processed: {processed}")

    print(f"\n🎉 Completed. Total: {total}, Processed: {processed}")

if __name__ == "__main__":
    main()
