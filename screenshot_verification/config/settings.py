"""
项目配置文件
"""
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """应用配置类"""
    
    # 基础配置
    app_name: str = "Screenshot Verification API"
    version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # 服务器配置
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=4, env="WORKERS")
    
    # 数据库配置
    database_url: str = Field(default="sqlite:///./verification.db", env="DATABASE_URL")
    
    # Redis配置
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # 模型配置
    model_path: str = Field(default="./models/", env="MODEL_PATH")
    model_device: str = Field(default="auto", env="MODEL_DEVICE")  # auto, cpu, cuda
    
    # 检测配置
    confidence_threshold: float = Field(default=0.8, env="CONFIDENCE_THRESHOLD")
    max_image_size: int = Field(default=10 * 1024 * 1024, env="MAX_IMAGE_SIZE")  # 10MB
    supported_formats: List[str] = Field(default=["jpg", "jpeg", "png", "bmp"], env="SUPPORTED_FORMATS")
    
    # 缓存配置
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1小时
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    
    # 安全配置
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    rate_limit_per_minute: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")
    
    # 日志配置
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # 监控配置
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # 检测器配置
    traditional_detector_enabled: bool = Field(default=True, env="TRADITIONAL_DETECTOR_ENABLED")
    metadata_detector_enabled: bool = Field(default=True, env="METADATA_DETECTOR_ENABLED")
    ai_detector_enabled: bool = Field(default=True, env="AI_DETECTOR_ENABLED")
    
    # 特征提取配置
    text_detection_enabled: bool = Field(default=True, env="TEXT_DETECTION_ENABLED")
    ui_element_detection_enabled: bool = Field(default=True, env="UI_ELEMENT_DETECTION_ENABLED")
    content_logic_detection_enabled: bool = Field(default=True, env="CONTENT_LOGIC_DETECTION_ENABLED")
    
    # 性能配置
    max_processing_time: float = Field(default=5.0, env="MAX_PROCESSING_TIME")  # 5秒
    batch_size: int = Field(default=1, env="BATCH_SIZE")
    
    # 存储配置
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    temp_dir: str = Field(default="./temp", env="TEMP_DIR")
    
    # 模型权重配置
    efficientnet_model_path: str = Field(default="./models/efficientnet_b0.pth", env="EFFICIENTNET_MODEL_PATH")
    transformer_model_path: str = Field(default="./models/vision_transformer.pth", env="TRANSFORMER_MODEL_PATH")
    dual_stream_model_path: str = Field(default="./models/dual_stream.pth", env="DUAL_STREAM_MODEL_PATH")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# 全局配置实例
settings = Settings()