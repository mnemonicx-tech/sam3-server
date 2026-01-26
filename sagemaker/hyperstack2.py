#!/usr/bin/env python3
import os, json, cv2, torch, argparse
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor

# ---------------- FALLBACK PROMPTS ----------------

FALLBACK_PROMPTS = {
    "ethnic_wear_women_sharara": [
        "wide leg pants worn by a woman",
        "long kurta worn by a woman",
        "dupatta worn by a woman",
    ],
    "ethnic_wear_women_lehenga_choli": [
        "long skirt worn by a woman",
        "blouse worn by a woman",
        "dupatta worn by a woman",
    ],
    "ethnic_wear_women_palazzo_set": [
        "palazzo pants worn by a woman",
        "kurta worn by a woman",
    ],
    "fusion_wear_women_indo-western_dress": [
        "long dress worn by a woman",
        "kurta worn by a woman",
    ],
    "fusion_wear_women_kaftan": [
        "loose dress worn by a woman",
        "long tunic worn by a woman",
    ],
    "sleepwear_men_pyjama_set": [
        "pajama pants worn by a person",
        "t-shirt worn by a person",
    ],
    "general_unisex_tracksuit": [
        "track pants worn by a person",
        "jacket worn by a person",
    ],
}

# ---------------- UTILS ----------------

def union_masks(results):
    final_mask = None
    for r in results:
        if r.masks is not None:
            m = r.masks.data.any(dim=0).cpu().numpy().astype(np.uint8) * 255
            final_mask = m if final_mask is None else np.maximum(final_mask, m)
    return final_mask

def load_prompts(path):
    with open(path, "r") as f:
        return json.load(f)

# ---------------- MAIN ----------------

def main(args):
    prompts = load_prompts(args.prompts)

    predictor = SAM3SemanticPredictor(overrides=dict(
        model="sam3.pt",
        task="segment",
        mode="predict",
        conf=args.conf,
        imgsz=1024,
        half=torch.cuda.is_available(),
        save=False,
    ))

    for category in os.listdir(args.input):
        cat_dir = os.path.join(args.input, category)
        if not os.path.isdir(cat_dir) or category not in prompts:
            continue

        out_dir = os.path.join(args.output, category)
        os.makedirs(out_dir, exist_ok=True)

        for img_name in os.listdir(cat_dir):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(cat_dir, img_name)
            predictor.set_image(img_path)

            base_prompt = prompts[category]
            final_mask = None

            # -------- PASS 1 --------
            results = predictor(text=[f"a person wearing {base_prompt}"])
            final_mask = union_masks(results)

            # -------- PASS 2 --------
            if final_mask is None and category in FALLBACK_PROMPTS:
                results = predictor(text=FALLBACK_PROMPTS[category])
                final_mask = union_masks(results)

            # -------- PASS 3 --------
            if final_mask is None:
                results = predictor(text=["a person wearing clothing"])
                final_mask = union_masks(results)

            if final_mask is None:
                continue

            # Morphology fill
            kernel = np.ones((25, 25), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

            cv2.imwrite(
                os.path.join(out_dir, img_name.replace(".jpg", "_mask.png")),
                final_mask
            )

            print(f"✅ Annotated: {category}/{img_name}")

# ---------------- RUN ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts", default="prompts.json")
    parser.add_argument("--conf", type=float, default=0.05)
    args = parser.parse_args()
    main(args)
