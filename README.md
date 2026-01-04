# SAM3 Inference Server (scaffold)

This repository contains a minimal FastAPI scaffold to run a SAM3-like predictor (via `ultralytics`). It is intended for local M1 development (CPU/MPS) and later deployment to AWS g4.xlarge (T4 GPU).

Quick start (macOS M1):

1. Install `uv` or create a virtual environment.
2. Install dependencies: `pip install -r requirements.txt` (on M1, install an MPS/CPU-compatible `torch` separately).
3. Run the app: `uvicorn server.app:app --reload --port 8000`.

Notes:

- The scaffold lazily imports `ultralytics`; install `ultralytics` to use the real predictor.
- Adjust `server/sam_infer.py` if the predictor API differs from your installed `ultralytics` version.
