import base64
import io
from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from loguru import logger
from PIL import Image

from screenshot_verifier.app.api.schemas import (
    HealthResponse,
    RiskFactor,
    VerifyRequest,
    VerifyResponse,
)
from screenshot_verifier.app.detection.fusion import FusionEngine

router = APIRouter()

# Instantiate detection engines (could be loaded lazily or via dependency injection)
_fusion_engine = FusionEngine()


def _decode_base64_image(data: str) -> np.ndarray:
    """Decode base64 string to BGR image (numpy array)."""
    try:
        image_bytes = base64.b64decode(data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to decode base64 image: {}", exc)
        raise HTTPException(status_code=400, detail="Invalid image data") from exc


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:  # noqa: D401
    """健康状态探针."""
    return HealthResponse(status="ok", model_version=_fusion_engine.model_version)


@router.post("/verify/screenshot", response_model=VerifyResponse)
async def verify_screenshot(request: VerifyRequest) -> VerifyResponse:  # noqa: D401
    """验证截图真伪."""
    bgr_image = _decode_base64_image(request.image)

    authentic, confidence, risk_items = _fusion_engine.predict(
        bgr_image, metadata=request.metadata.model_dump()
    )

    return VerifyResponse(
        authentic=authentic,
        confidence=confidence,
        risk_factors=[RiskFactor(**item) for item in risk_items],
    )