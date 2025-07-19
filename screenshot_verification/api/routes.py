"""
API路由
实现所有主要的API端点
"""
import time
import base64
import logging
from typing import List
import numpy as np
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from .models import (
    VerificationRequest, VerificationResponse, BatchVerificationRequest,
    BatchVerificationResponse, HealthCheckResponse, ErrorResponse,
    StatisticsResponse, ModelInfoResponse, FeedbackRequest, FeedbackResponse,
    RiskFactor, DetectorResult
)
from core.verification_engine import VerificationEngine
from utils.cache import CacheManager
from utils.metrics import MetricsCollector
from utils.security import verify_api_key
from config.settings import settings

# 创建路由器
router = APIRouter()

# 全局变量
verification_engine = None
cache_manager = None
metrics_collector = None
start_time = time.time()

# 初始化函数
def get_verification_engine():
    """获取验证引擎实例"""
    global verification_engine
    if verification_engine is None:
        verification_engine = VerificationEngine()
    return verification_engine

def get_cache_manager():
    """获取缓存管理器实例"""
    global cache_manager
    if cache_manager is None:
        cache_manager = CacheManager()
    return cache_manager

def get_metrics_collector():
    """获取指标收集器实例"""
    global metrics_collector
    if metrics_collector is None:
        metrics_collector = MetricsCollector()
    return metrics_collector

# 图像处理工具函数
def decode_base64_image(image_data: str) -> np.ndarray:
    """解码Base64图像数据"""
    try:
        # 解码Base64数据
        image_bytes = base64.b64decode(image_data)
        
        # 转换为numpy数组
        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("无法解码图像数据")
        
        return image
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"图像数据解码失败: {str(e)}"
        )

def convert_risk_factors(risk_factors: List[dict]) -> List[RiskFactor]:
    """转换风险因子格式"""
    converted = []
    for risk in risk_factors:
        converted.append(RiskFactor(
            type=risk.get('type', 'unknown'),
            severity=risk.get('severity', 'low'),
            confidence=risk.get('confidence', 0.0),
            location=risk.get('location'),
            description=risk.get('description', '')
        ))
    return converted

def convert_detector_results(detector_results: dict) -> dict:
    """转换检测器结果格式"""
    converted = {}
    for name, result in detector_results.items():
        converted[name] = DetectorResult(
            is_authentic=result.is_authentic,
            confidence=result.confidence,
            processing_time=result.processing_time,
            risk_factors=convert_risk_factors(result.risk_factors)
        )
    return converted

# API端点
@router.post("/verify/screenshot", response_model=VerificationResponse)
async def verify_screenshot(
    request: VerificationRequest,
    background_tasks: BackgroundTasks,
    engine: VerificationEngine = Depends(get_verification_engine),
    cache: CacheManager = Depends(get_cache_manager),
    metrics: MetricsCollector = Depends(get_metrics_collector),
    api_key: str = Depends(verify_api_key)
):
    """
    验证单个截图真伪
    
    - **image**: Base64编码的图像数据
    - **metadata**: 元数据信息
    - **source**: 图像来源 (android/ios/web/desktop/other)
    - **app_type**: 应用类型 (payment/social/shopping/gaming/finance/other)
    - **context**: 上下文信息
    """
    try:
        # 检查缓存
        cache_key = f"screenshot_verify:{hash(request.image)}"
        cached_result = cache.get(cache_key)
        if cached_result:
            metrics.record_cache_hit()
            return VerificationResponse(**cached_result)
        
        # 解码图像
        image = decode_base64_image(request.image)
        
        # 准备元数据
        metadata = {
            'source': request.source,
            'app_type': request.app_type,
            'context': request.context,
            **request.metadata
        }
        
        # 执行验证
        result = engine.verify_screenshot(image, metadata)
        
        # 记录指标
        metrics.record_verification(
            is_authentic=result.is_authentic,
            confidence=result.confidence,
            processing_time=result.processing_time
        )
        
        # 缓存结果
        background_tasks.add_task(cache.set, cache_key, {
            'authentic': result.is_authentic,
            'confidence': result.confidence,
            'risk_factors': result.risk_factors,
            'processing_time': result.processing_time,
            'detector_results': {
                name: {
                    'is_authentic': res.is_authentic,
                    'confidence': res.confidence,
                    'processing_time': res.processing_time,
                    'risk_factors': res.risk_factors
                }
                for name, res in result.detector_results.items()
            }
        }, ttl=settings.cache_ttl)
        
        # 返回结果
        return VerificationResponse(
            authentic=result.is_authentic,
            confidence=result.confidence,
            risk_factors=convert_risk_factors(result.risk_factors),
            processing_time=result.processing_time,
            detector_results=convert_detector_results(result.detector_results),
            detailed_report=result.detailed_report
        )
        
    except Exception as e:
        logging.error(f"验证失败: {e}")
        metrics.record_error()
        raise HTTPException(
            status_code=500,
            detail=f"验证过程失败: {str(e)}"
        )

@router.post("/verify/batch", response_model=BatchVerificationResponse)
async def verify_batch_screenshots(
    request: BatchVerificationRequest,
    engine: VerificationEngine = Depends(get_verification_engine),
    metrics: MetricsCollector = Depends(get_metrics_collector),
    api_key: str = Depends(verify_api_key)
):
    """
    批量验证截图真伪
    
    - **images**: Base64编码的图像数据列表（最多10张）
    - **metadata**: 元数据信息
    - **source**: 图像来源
    - **app_type**: 应用类型
    """
    try:
        start_time = time.time()
        results = []
        success_count = 0
        failure_count = 0
        
        for i, image_data in enumerate(request.images):
            try:
                # 解码图像
                image = decode_base64_image(image_data)
                
                # 准备元数据
                metadata = {
                    'source': request.source,
                    'app_type': request.app_type,
                    'batch_index': i,
                    **request.metadata
                }
                
                # 执行验证
                result = engine.verify_screenshot(image, metadata)
                
                # 记录成功
                success_count += 1
                metrics.record_verification(
                    is_authentic=result.is_authentic,
                    confidence=result.confidence,
                    processing_time=result.processing_time
                )
                
                # 添加到结果列表
                results.append(VerificationResponse(
                    authentic=result.is_authentic,
                    confidence=result.confidence,
                    risk_factors=convert_risk_factors(result.risk_factors),
                    processing_time=result.processing_time,
                    detector_results=convert_detector_results(result.detector_results),
                    detailed_report=result.detailed_report
                ))
                
            except Exception as e:
                logging.error(f"批量验证第{i+1}张图像失败: {e}")
                failure_count += 1
                
                # 添加错误结果
                results.append(VerificationResponse(
                    authentic=True,
                    confidence=0.5,
                    risk_factors=[RiskFactor(
                        type="processing_error",
                        severity="medium",
                        confidence=0.5,
                        description=f"处理失败: {str(e)}"
                    )],
                    processing_time=0.0,
                    detector_results={},
                    detailed_report={"error": str(e)}
                ))
        
        total_processing_time = time.time() - start_time
        
        return BatchVerificationResponse(
            results=results,
            total_processing_time=total_processing_time,
            success_count=success_count,
            failure_count=failure_count
        )
        
    except Exception as e:
        logging.error(f"批量验证失败: {e}")
        metrics.record_error()
        raise HTTPException(
            status_code=500,
            detail=f"批量验证失败: {str(e)}"
        )

@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    engine: VerificationEngine = Depends(get_verification_engine)
):
    """
    健康检查
    
    返回服务状态、版本信息和检测器状态
    """
    try:
        global start_time
        uptime = time.time() - start_time
        
        # 获取检测器状态
        detector_status = engine.get_detector_status()
        
        return HealthCheckResponse(
            status="healthy",
            version=settings.version,
            uptime=uptime,
            detector_status=detector_status
        )
        
    except Exception as e:
        logging.error(f"健康检查失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"健康检查失败: {str(e)}"
        )

@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(
    metrics: MetricsCollector = Depends(get_metrics_collector),
    api_key: str = Depends(verify_api_key)
):
    """
    获取统计信息
    
    返回服务使用统计和性能指标
    """
    try:
        stats = metrics.get_statistics()
        
        return StatisticsResponse(
            total_requests=stats['total_requests'],
            successful_requests=stats['successful_requests'],
            failed_requests=stats['failed_requests'],
            average_processing_time=stats['average_processing_time'],
            authentic_count=stats['authentic_count'],
            fake_count=stats['fake_count'],
            detector_performance=stats['detector_performance']
        )
        
    except Exception as e:
        logging.error(f"获取统计信息失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取统计信息失败: {str(e)}"
        )

@router.get("/models", response_model=List[ModelInfoResponse])
async def get_model_info(
    api_key: str = Depends(verify_api_key)
):
    """
    获取模型信息
    
    返回所有可用模型的详细信息
    """
    try:
        models = [
            ModelInfoResponse(
                model_name="EfficientNet-B0",
                model_version="1.0.0",
                model_type="CNN",
                accuracy=0.95,
                last_updated="2024-01-01",
                parameters={
                    "input_size": "224x224",
                    "num_classes": 2,
                    "pretrained": True
                }
            ),
            ModelInfoResponse(
                model_name="Vision Transformer",
                model_version="1.0.0",
                model_type="Transformer",
                accuracy=0.93,
                last_updated="2024-01-01",
                parameters={
                    "patch_size": 16,
                    "num_layers": 12,
                    "num_heads": 12
                }
            ),
            ModelInfoResponse(
                model_name="Dual Stream Network",
                model_version="1.0.0",
                model_type="Hybrid",
                accuracy=0.96,
                last_updated="2024-01-01",
                parameters={
                    "rgb_stream": "EfficientNet",
                    "frequency_stream": "DCT-CNN"
                }
            )
        ]
        
        return models
        
    except Exception as e:
        logging.error(f"获取模型信息失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取模型信息失败: {str(e)}"
        )

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    提交反馈
    
    - **image_id**: 图像ID
    - **actual_label**: 实际标签
    - **feedback_type**: 反馈类型
    - **comments**: 反馈意见
    """
    try:
        # 生成反馈ID
        import uuid
        feedback_id = str(uuid.uuid4())
        
        # 异步处理反馈
        background_tasks.add_task(process_feedback, request, feedback_id)
        
        return FeedbackResponse(
            success=True,
            message="反馈提交成功",
            feedback_id=feedback_id
        )
        
    except Exception as e:
        logging.error(f"提交反馈失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"提交反馈失败: {str(e)}"
        )

async def process_feedback(request: FeedbackRequest, feedback_id: str):
    """处理反馈（异步）"""
    try:
        # 这里可以实现反馈处理逻辑
        # 例如：保存到数据库、触发模型重训练等
        logging.info(f"处理反馈 {feedback_id}: {request.feedback_type}")
        
    except Exception as e:
        logging.error(f"处理反馈失败: {e}")

@router.get("/detectors", response_model=dict)
async def get_detector_info(
    engine: VerificationEngine = Depends(get_verification_engine),
    api_key: str = Depends(verify_api_key)
):
    """
    获取检测器信息
    
    返回所有可用检测器的详细信息
    """
    try:
        detector_info = {
            "traditional": {
                "name": "传统图像分析检测器",
                "description": "基于JPEG压缩痕迹、噪声模式、边缘不一致性等传统图像分析方法",
                "enabled": settings.traditional_detector_enabled,
                "capabilities": [
                    "JPEG压缩痕迹检测",
                    "噪声模式分析",
                    "边缘不一致性检测",
                    "双重压缩检测"
                ]
            },
            "metadata": {
                "name": "元数据分析检测器",
                "description": "分析EXIF信息、文件属性、图像指纹等元数据",
                "enabled": settings.metadata_detector_enabled,
                "capabilities": [
                    "EXIF信息验证",
                    "文件属性检查",
                    "图像指纹对比"
                ]
            },
            "ai": {
                "name": "AI深度学习检测器",
                "description": "基于深度学习的图像真伪识别",
                "enabled": settings.ai_detector_enabled,
                "capabilities": [
                    "EfficientNet模型",
                    "Vision Transformer",
                    "双流网络"
                ]
            }
        }
        
        return detector_info
        
    except Exception as e:
        logging.error(f"获取检测器信息失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取检测器信息失败: {str(e)}"
        )

@router.delete("/cache")
async def clear_cache(
    cache: CacheManager = Depends(get_cache_manager),
    api_key: str = Depends(verify_api_key)
):
    """
    清除缓存
    
    清除所有缓存的验证结果
    """
    try:
        cache.clear()
        return {"message": "缓存清除成功"}
        
    except Exception as e:
        logging.error(f"清除缓存失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"清除缓存失败: {str(e)}"
        )

# 错误处理
@router.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            details={"path": request.url.path}
        ).dict()
    )

@router.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理器"""
    logging.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="内部服务器错误",
            error_code="INTERNAL_ERROR",
            details={"path": request.url.path}
        ).dict()
    )