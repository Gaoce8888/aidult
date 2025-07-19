"""
主应用文件
启动FastAPI服务
"""
import os
import sys
import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings
from api.routes import router
from utils.metrics import MetricsCollector
from utils.security import get_security_manager

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.log_file) if settings.log_file else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="手机截图AI识别真伪系统API",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# 全局变量
metrics_collector = None
security_manager = None


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    global metrics_collector, security_manager
    
    logger.info("正在启动手机截图验证服务...")
    
    # 初始化指标收集器
    metrics_collector = MetricsCollector()
    
    # 初始化安全管理器
    security_manager = get_security_manager()
    
    # 创建必要的目录
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.temp_dir, exist_ok=True)
    os.makedirs(settings.model_path, exist_ok=True)
    
    logger.info(f"服务启动完成 - 版本: {settings.version}")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("正在关闭手机截图验证服务...")
    
    # 清理资源
    if metrics_collector:
        # 导出最终指标
        final_metrics = metrics_collector.export_metrics()
        logger.info(f"最终指标: {final_metrics}")
    
    logger.info("服务已关闭")


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """请求中间件"""
    start_time = time.time()
    
    # 记录请求信息
    logger.debug(f"收到请求: {request.method} {request.url.path}")
    
    # 处理请求
    try:
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 添加处理时间到响应头
        response.headers["X-Process-Time"] = str(process_time)
        
        # 记录成功请求
        if metrics_collector:
            metrics_collector.record_verification(
                is_authentic=True,  # 这里只是记录请求，不是验证结果
                confidence=1.0,
                processing_time=process_time
            )
        
        logger.debug(f"请求处理完成: {request.method} {request.url.path} - 耗时: {process_time:.3f}s")
        
        return response
        
    except Exception as e:
        # 记录错误
        logger.error(f"请求处理失败: {request.method} {request.url.path} - 错误: {e}")
        
        if metrics_collector:
            metrics_collector.record_error()
        
        # 返回错误响应
        return JSONResponse(
            status_code=500,
            content={
                "error": "内部服务器错误",
                "error_code": "INTERNAL_ERROR",
                "details": {"path": request.url.path}
            }
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    
    if metrics_collector:
        metrics_collector.record_error()
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "error_code": "INTERNAL_ERROR",
            "details": {
                "path": request.url.path,
                "method": request.method
            }
        }
    )


# 注册路由
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "手机截图AI识别真伪系统",
        "version": settings.version,
        "status": "running",
        "docs": "/docs" if settings.debug else None
    }


@app.get("/ping")
async def ping():
    """健康检查"""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/info")
async def info():
    """服务信息"""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "description": "手机截图AI识别真伪系统",
        "features": [
            "传统图像分析检测",
            "元数据分析检测", 
            "AI深度学习检测",
            "多特征融合决策",
            "实时性能监控",
            "安全防护机制"
        ],
        "endpoints": {
            "verify": "/api/v1/verify/screenshot",
            "batch_verify": "/api/v1/verify/batch",
            "health": "/api/v1/health",
            "statistics": "/api/v1/statistics",
            "models": "/api/v1/models",
            "detectors": "/api/v1/detectors"
        }
    }


if __name__ == "__main__":
    # 启动服务器
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers if not settings.debug else 1,
        log_level=settings.log_level.lower(),
        access_log=True
    )