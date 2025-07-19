from typing import Any, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, constr

BBox = Tuple[int, int, int, int]

class ScreenshotMetadata(BaseModel):
    source: Literal["android", "ios", "unknown"] = Field(..., description="设备来源")
    app_type: constr(strip_whitespace=True) = Field(..., description="应用类型，如 payment/social/shopping 等")
    context: Optional[str] = Field(None, description="附加上下文信息")

class RiskFactor(BaseModel):
    type: str = Field(..., description="风险类型")
    severity: Literal["low", "medium", "high"]
    location: Optional[List[BBox]] = None

class VerifyRequest(BaseModel):
    image: str = Field(..., description="Base64 编码后的图片数据")
    metadata: ScreenshotMetadata

class VerifyResponse(BaseModel):
    authentic: bool
    confidence: float = Field(..., ge=0, le=1)
    risk_factors: List[RiskFactor]
    detailed_report: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_version: str
    build: Optional[str] = None