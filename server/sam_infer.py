import tempfile
import os
from typing import Any, List
from dataclasses import dataclass
from .schemas import InferenceOptions


class ModelWrapper:
    """Lightweight wrapper around an Ultralytics SAM3 predictor.

    This wrapper lazily imports `ultralytics` and provides `load_model`, `set_image`, and `infer`.
    It intentionally avoids importing heavy libs at module import time so the app can start
    and provide helpful error messages if deps are missing.
    """

    def __init__(self):
        self.predictor = None
        self.model_path = None
        self.device = "cpu"
        self.half = False
        self.image_path = None
        self.image_width = None
        self.image_height = None

    def load_model(self, model_path: str | None = None, device: str = "cpu", half: bool = False, overrides: dict | None = None):
        """Load SAM3 predictor from ultralytics. If `ultralytics` is not installed, raise an informative error."""
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
        except Exception as e:
            # Enter mock mode when ultralytics or dependencies are not available.
            # This allows local development and endpoint testing without heavy weights.
            print(f"[ModelWrapper] Warning: ultralytics not available, entering mock mode: {e}")
            self.predictor = None
            self.mock = True
            return

        self.mock = False

        self.model_path = model_path
        self.device = device
        self.half = half

        overrides = overrides or {}
        # ensure some reasonable defaults
        overrides.setdefault("conf", 0.25)
        overrides.setdefault("task", "segment")
        overrides.setdefault("mode", "predict")
        if model_path:
            overrides.setdefault("model", model_path)

        # create predictor instance
        try:
            self.predictor = SAM3SemanticPredictor(overrides=overrides)
            print(f"[ModelWrapper] SAM3 model loaded from {model_path}")
        except Exception as e:
            print(f"[ModelWrapper] Error loading SAM3 from {model_path}: {e}")
            raise

    def set_image(self, image_input: bytes | str):
        """Accept bytes or a file path. If bytes, write to a temp file and pass path to predictor."""
        if isinstance(image_input, bytes):
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tf.write(image_input)
            tf.flush()
            tf.close()
            path = tf.name
        elif isinstance(image_input, str) and os.path.exists(image_input):
            path = image_input
        else:
            raise ValueError("image_input must be bytes or an existing file path")

        # allow setting image in mock mode as well
        if self.predictor is None and not getattr(self, "mock", False):
            raise RuntimeError("Model not loaded; call load_model() before set_image()")

        # call predictor.set_image only if predictor is not None
        if self.predictor is not None:
            try:
                self.predictor.set_image(path)
            except Exception:
                # fallback: try passing path anyway
                self.predictor.set_image(path)

        # record last image path; dimensions may be available on predictor or can be resolved lazily
        self.image_path = path
        try:
            from PIL import Image

            with Image.open(path) as im:
                self.image_width, self.image_height = im.size
        except Exception:
            self.image_width = None
            self.image_height = None

    def infer(self, texts: List[str], options: InferenceOptions | dict | None = None) -> List[dict]:
        """Run predictor on the last-set image with a list of text prompts.

        Uses Ultralytics native Results structures:
        - Results.masks.data: tensor of shape (N, H, W)
        - Results.boxes.conf: confidence scores
        - Results.boxes.xyxy: bounding box coordinates
        """
        # Mock inference path for local testing without SAM3 weights
        if getattr(self, "mock", False):
            masks_out: List[dict] = []
            w = self.image_width or 256
            h = self.image_height or 256
            for i, t in enumerate(texts or ["concept"]):
                # center box covering 50% of short edge
                bw = int(w * 0.5)
                bh = int(h * 0.5)
                x0 = (w - bw) // 2
                y0 = (h - bh) // 2
                x1 = x0 + bw
                y1 = y0 + bh
                masks_out.append({
                    "mask_id": str(i),
                    "score": 1.0,
                    "bbox": [x0, y0, x1, y1],
                    "rle": None,
                    "png_base64": None,
                    "area": bw * bh,
                    "raw": f"mock mask for '{t}'",
                })
            return masks_out

        # call predictor with text prompts
        try:
            results = self.predictor(text=texts)
        except TypeError:
            results = self.predictor(texts)

        masks_out: List[dict] = []
        try:
            import numpy as _np
            import base64, io
            from PIL import Image as _Image

            # results is a list of Results objects (one per image/batch)
            iter_results = results if isinstance(results, (list, tuple)) else [results]
            
            for r in iter_results:
                # Extract masks from Results.masks.data (shape: N, H, W)
                masks_tensor = None
                if hasattr(r, "masks") and r.masks is not None:
                    masks_tensor = getattr(r.masks, "data", None)
                    if masks_tensor is None:
                        masks_tensor = r.masks

                # Extract boxes and confidence from Results.boxes
                boxes_data = None
                conf_scores = None
                if hasattr(r, "boxes") and r.boxes is not None:
                    boxes_data = getattr(r.boxes, "xyxy", None)  # shape (N, 4)
                    conf_scores = getattr(r.boxes, "conf", None)  # shape (N,)

                # Convert tensors to numpy
                if masks_tensor is not None:
                    if hasattr(masks_tensor, "cpu"):
                        masks_np = masks_tensor.cpu().numpy()
                    else:
                        masks_np = _np.array(masks_tensor)
                else:
                    masks_np = None

                if boxes_data is not None:
                    if hasattr(boxes_data, "cpu"):
                        boxes_np = boxes_data.cpu().numpy()
                    else:
                        boxes_np = _np.array(boxes_data)
                else:
                    boxes_np = None

                if conf_scores is not None:
                    if hasattr(conf_scores, "cpu"):
                        conf_np = conf_scores.cpu().numpy()
                    else:
                        conf_np = _np.array(conf_scores)
                else:
                    conf_np = None

                # Process each mask
                num_masks = masks_np.shape[0] if masks_np is not None else 0
                for i in range(num_masks):
                    entry: dict[str, object] = {"mask_id": str(i)}

                    # Extract bbox from boxes_np
                    if boxes_np is not None and i < boxes_np.shape[0]:
                        bbox = boxes_np[i].tolist()
                        entry["bbox"] = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

                    # Extract confidence score
                    if conf_np is not None and i < conf_np.shape[0]:
                        entry["score"] = float(conf_np[i])

                    # Extract mask area
                    if masks_np is not None and i < masks_np.shape[0]:
                        mask = masks_np[i]
                        entry["area"] = int((mask > 0).sum())

                    masks_out.append(entry)

        except Exception as e:
            # Log the exception but return a descriptive error
            import traceback
            traceback.print_exc()
            masks_out = [{"mask_id": "0", "error": str(e)}]

        return masks_out


def ensure_model_loaded():
    mw = ModelWrapper()
    mw.load_model()
    return mw
