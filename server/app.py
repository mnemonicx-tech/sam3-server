from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from uuid import uuid4
import io
import base64
from PIL import Image
from .schemas import InferenceOptions, InferenceResponse
from .sam_infer import ModelWrapper

app = FastAPI(title="SAM3 Inference Server")

# global model wrapper (loaded on startup)
model_wrapper: ModelWrapper | None = None


@app.on_event("startup")
def startup_event():
    global model_wrapper
    model_wrapper = ModelWrapper()
    try:
        # prefer the local checkpoint if available
        model_wrapper.load_model(model_path="models/sam3.pt", device="cpu", half=False)
    except Exception as e:
        # store error so endpoints can return a helpful message
        app.state.model_error = str(e)


@app.post("/infer")
async def infer(
    image: UploadFile = File(None),
    image_url: str | None = Form(None),
    base64_image: str | None = Form(None),
    prompt: str | None = Form(None),
    options: str | None = Form(None),
    mock: bool = Query(False),
):
    request_id = str(uuid4())

    # Get image bytes from file upload, S3 URL, or base64
    if image:
        contents = await image.read()
    elif image_url:
        import boto3
        # Parse S3 URL: s3://bucket-name/path/to/file.png or https://bucket.s3.amazonaws.com/path
        if image_url.startswith("s3://"):
            bucket, key = image_url.replace("s3://", "").split("/", 1)
        else:
            # Handle https:// S3 URLs (both virtual-hosted and path-style)
            if ".s3.amazonaws.com" in image_url:
                bucket = image_url.split(".")[0].split("//")[1]
                key = image_url.split(".s3.amazonaws.com/")[1]
            else:
                raise HTTPException(status_code=400, detail="Invalid S3 URL format")
        
        try:
            s3 = boto3.client("s3")
            response = s3.get_object(Bucket=bucket, Key=key)
            contents = response["Body"].read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch S3 image: {e}")
    elif base64_image:
        try:
            # Handle base64 with or without data URI prefix (e.g., "data:image/png;base64,...")
            if "," in base64_image:
                contents = base64.b64decode(base64_image.split(",")[1])
            else:
                contents = base64.b64decode(base64_image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
    else:
        raise HTTPException(status_code=400, detail="Provide 'image' file, 'image_url' (S3), or 'base64_image'")

    # Mock mode for testing (e.g., with n8n)
    if mock:
        try:
            contents = await image.read()
            with Image.open(io.BytesIO(contents)) as img:
                width, height = img.size
        except Exception:
            width, height = 1290, 1274

        response = InferenceResponse(
            request_id=request_id,
            image_size={"width": width, "height": height},
            masks=[
                {
                    "mask_id": "0",
                    "score": 0.95,
                    "bbox": [int(width * 0.2), int(height * 0.2), int(width * 0.8), int(height * 0.8)],
                    "png_base64": None,
                    "area": int((width * 0.6) * (height * 0.6)),
                }
            ],
            processing_time_ms=150.0,
        )
        return JSONResponse(status_code=200, content=response.model_dump())

    # Real inference path
    if getattr(app.state, "model_error", None):
        raise HTTPException(status_code=500, detail={"model_error": app.state.model_error})

    if model_wrapper is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    contents = await image.read()
    try:
        opts = InferenceOptions.parse_raw(options) if options else InferenceOptions()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid options JSON: {e}")

    try:
        model_wrapper.set_image(contents)
        results = model_wrapper.infer(texts=[prompt] if prompt else [], options=opts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # compute overall confidence (max of per-mask scores when available)
    overall_conf = None
    try:
        scores = [m.get("score") for m in results if m.get("score") is not None]
        if scores:
            overall_conf = float(max(scores))
    except Exception:
        overall_conf = None

    response = InferenceResponse(
        request_id=request_id,
        image_size={"width": model_wrapper.image_width or 0, "height": model_wrapper.image_height or 0},
        masks=results,
        processing_time_ms=0.0,
        overall_confidence=overall_conf,
    )
    return JSONResponse(status_code=200, content=response.model_dump())


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)
