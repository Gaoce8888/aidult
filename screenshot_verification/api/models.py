"""
API数据模型
定义请求和响应的数据结构
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class AppType(str, Enum):
    """应用类型枚举"""
    PAYMENT = "payment"
    SOCIAL = "social"
    SHOPPING = "shopping"
    GAMING = "gaming"
    FINANCE = "finance"
    OTHER = "other"


class SourceType(str, Enum):
    """来源类型枚举"""
    ANDROID = "android"
    IOS = "ios"
    WEB = "web"
    DESKTOP = "desktop"
    OTHER = "other"


class SeverityLevel(str, Enum):
    """严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskLevel(str, Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VerificationRequest(BaseModel):
    """验证请求模型"""
    image: str = Field(..., description="Base64编码的图像数据")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="元数据信息")
    
    # 可选的高级参数
    source: Optional[SourceType] = Field(default=SourceType.OTHER, description="图像来源")
    app_type: Optional[AppType] = Field(default=AppType.OTHER, description="应用类型")
    context: Optional[str] = Field(default="", description="上下文信息")
    
    @validator('image')
    def validate_image(cls, v):
        """验证图像数据"""
        if not v:
            raise ValueError("图像数据不能为空")
        
        # 检查Base64格式
        try:
            import base64
            base64.b64decode(v)
        except Exception:
            raise ValueError("图像数据必须是有效的Base64格式")
        
        return v


class RiskFactor(BaseModel):
    """风险因子模型"""
    type: str = Field(..., description="风险类型")
    severity: SeverityLevel = Field(..., description="严重程度")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    location: Optional[List[int]] = Field(default=None, description="位置坐标")
    description: str = Field(default="", description="风险描述")


class DetectorResult(BaseModel):
    """检测器结果模型"""
    is_authentic: bool = Field(..., description="是否真实")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    processing_time: float = Field(..., ge=0.0, description="处理时间")
    risk_factors: List[RiskFactor] = Field(default=[], description="风险因子")


class VerificationSummary(BaseModel):
    """验证摘要模型"""
    is_authentic: bool = Field(..., description="是否真实")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    risk_level: RiskLevel = Field(..., description="风险等级")


class VerificationResponse(BaseModel):
    """验证响应模型"""
    authentic: bool = Field(..., description="是否真实")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    risk_factors: List[RiskFactor] = Field(default=[], description="风险因子")
    processing_time: float = Field(..., ge=0.0, description="总处理时间")
    detector_results: Dict[str, DetectorResult] = Field(default={}, description="各检测器结果")
    detailed_report: Optional[Dict[str, Any]] = Field(default=None, description="详细报告")
    
    class Config:
        schema_extra = {
            "example": {
                "authentic": True,
                "confidence": 0.95,
                "risk_factors": [
                    {
                        "type": "compression_artifact",
                        "severity": "low",
                        "confidence": 0.3,
                        "description": "检测到轻微压缩痕迹"
                    }
                ],
                "processing_time": 0.15,
                "detector_results": {
                    "traditional": {
                        "is_authentic": True,
                        "confidence": 0.9,
                        "processing_time": 0.05,
                        "risk_factors": []
                    },
                    "metadata": {
                        "is_authentic": True,
                        "confidence": 0.95,
                        "processing_time": 0.03,
                        "risk_factors": []
                    },
                    "ai": {
                        "is_authentic": True,
                        "confidence": 0.92,
                        "processing_time": 0.08,
                        "risk_factors": []
                    }
                }
            }
        }


class BatchVerificationRequest(BaseModel):
    """批量验证请求模型"""
    images: List[str] = Field(..., description="Base64编码的图像数据列表")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="元数据信息")
    source: Optional[SourceType] = Field(default=SourceType.OTHER, description="图像来源")
    app_type: Optional[AppType] = Field(default=AppType.OTHER, description="应用类型")
    
    @validator('images')
    def validate_images(cls, v):
        """验证图像列表"""
        if not v:
            raise ValueError("图像列表不能为空")
        
        if len(v) > 10:
            raise ValueError("批量验证最多支持10张图像")
        
        # 验证每个图像
        for i, image in enumerate(v):
            try:
                import base64
                base64.b64decode(image)
            except Exception:
                raise ValueError(f"第{i+1}张图像数据必须是有效的Base64格式")
        
        return v


class BatchVerificationResponse(BaseModel):
    """批量验证响应模型"""
    results: List[VerificationResponse] = Field(..., description="验证结果列表")
    total_processing_time: float = Field(..., ge=0.0, description="总处理时间")
    success_count: int = Field(..., ge=0, description="成功验证数量")
    failure_count: int = Field(..., ge=0, description="失败验证数量")


class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="服务版本")
    uptime: float = Field(..., description="运行时间")
    detector_status: Dict[str, Dict[str, Any]] = Field(..., description="检测器状态")


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str = Field(..., description="错误信息")
    error_code: str = Field(..., description="错误代码")
    details: Optional[Dict[str, Any]] = Field(default=None, description="错误详情")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "图像数据格式错误",
                "error_code": "INVALID_IMAGE_FORMAT",
                "details": {
                    "field": "image",
                    "message": "Base64解码失败"
                }
            }
        }


class StatisticsResponse(BaseModel):
    """统计信息响应模型"""
    total_requests: int = Field(..., description="总请求数")
    successful_requests: int = Field(..., description="成功请求数")
    failed_requests: int = Field(..., description="失败请求数")
    average_processing_time: float = Field(..., description="平均处理时间")
    authentic_count: int = Field(..., description="真实图像数量")
    fake_count: int = Field(..., description="伪造图像数量")
    detector_performance: Dict[str, Dict[str, Any]] = Field(..., description="检测器性能")


class ModelInfoResponse(BaseModel):
    """模型信息响应模型"""
    model_name: str = Field(..., description="模型名称")
    model_version: str = Field(..., description="模型版本")
    model_type: str = Field(..., description="模型类型")
    accuracy: float = Field(..., description="准确率")
    last_updated: str = Field(..., description="最后更新时间")
    parameters: Dict[str, Any] = Field(..., description="模型参数")


class FeedbackRequest(BaseModel):
    """反馈请求模型"""
    image_id: str = Field(..., description="图像ID")
    actual_label: bool = Field(..., description="实际标签")
    feedback_type: str = Field(..., description="反馈类型")
    comments: Optional[str] = Field(default="", description="反馈意见")
    
    @validator('feedback_type')
    def validate_feedback_type(cls, v):
        """验证反馈类型"""
        valid_types = ['false_positive', 'false_negative', 'improvement', 'other']
        if v not in valid_types:
            raise ValueError(f"反馈类型必须是以下之一: {valid_types}")
        return v


class FeedbackResponse(BaseModel):
    """反馈响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="响应消息")
    feedback_id: str = Field(..., description="反馈ID")