from pydantic import BaseModel, Field
from typing import List, Optional


class InferenceOptions(BaseModel):
    mask_threshold: float = Field(0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_masks: int = Field(10, ge=1)
    return_png: bool = True
    return_rle: bool = True
    resize_short_edge: Optional[int] = None


class MaskResult(BaseModel):
    mask_id: str
    score: Optional[float] = None
    bbox: Optional[List[int]] = None
    rle: Optional[str] = None
    png_base64: Optional[str] = None
    area: Optional[int] = None
    raw: Optional[str] = None


class InferenceResponse(BaseModel):
    request_id: str
    image_size: dict
    masks: List[MaskResult]
    processing_time_ms: float = 0.0
    overall_confidence: float | None = None
