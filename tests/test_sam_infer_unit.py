import pytest

from server.schemas import InferenceOptions


def test_inference_options_defaults():
    opts = InferenceOptions()
    assert opts.mask_threshold == 0.5
    assert opts.max_masks == 10


@pytest.mark.asyncio
async def test_health_endpoint():
    from fastapi.testclient import TestClient
    from server.app import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"
