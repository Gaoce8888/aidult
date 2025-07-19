"""
Configuration settings for the Screenshot Authenticity AI system
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class ModelConfig(BaseModel):
    """Model configuration"""
    efficientnet_variant: str = "efficientnet_b0"
    vision_transformer_model: str = "vit_base_patch16_224"
    input_size: tuple = (224, 224)
    batch_size: int = 32
    confidence_threshold: float = 0.7
    use_gpu: bool = True
    model_cache_dir: str = "models/cache"


class DetectionConfig(BaseModel):
    """Detection algorithm configuration"""
    enable_traditional_methods: bool = True
    enable_metadata_analysis: bool = True
    enable_deep_learning: bool = True
    enable_ensemble: bool = True
    
    # Traditional detection parameters
    jpeg_quality_threshold: float = 0.8
    noise_variance_threshold: float = 0.1
    edge_sharpness_threshold: float = 0.5
    
    # Multi-stage detection timeouts (ms)
    stage1_timeout: int = 50
    stage2_timeout: int = 200
    stage3_timeout: int = 1000


class APIConfig(BaseModel):
    """API configuration"""
    title: str = "Screenshot Authenticity AI API"
    version: str = "1.0.0"
    description: str = "AI-powered screenshot authenticity verification system"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_image_formats: List[str] = ["jpeg", "jpg", "png", "webp"]
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour


class DatabaseConfig(BaseModel):
    """Database configuration"""
    url: str = "sqlite:///./authenticity.db"
    echo: bool = False
    pool_size: int = 20
    max_overflow: int = 30


class SecurityConfig(BaseModel):
    """Security configuration"""
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    api_key_header: str = "X-API-Key"


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = "logs/app.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class MonitoringConfig(BaseModel):
    """Monitoring and metrics configuration"""
    enable_metrics: bool = True
    metrics_port: int = 8001
    enable_health_check: bool = True
    health_check_interval: int = 30


class Settings(BaseSettings):
    """Application settings"""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default="8000", env="PORT")
    
    # Configuration sections
    model: ModelConfig = ModelConfig()
    detection: DetectionConfig = DetectionConfig()
    api: APIConfig = APIConfig()
    database: DatabaseConfig = DatabaseConfig()
    security: SecurityConfig = SecurityConfig()
    logging: LoggingConfig = LoggingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    
    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    models_dir: Path = project_root / "models"
    logs_dir: Path = project_root / "logs"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.data_dir.mkdir(exist_ok=True)
settings.models_dir.mkdir(exist_ok=True)
settings.logs_dir.mkdir(exist_ok=True)


# Application-specific configurations
SCREENSHOT_PATTERNS = {
    "android": {
        "status_bar_height": 24,
        "navigation_bar_height": 48,
        "common_resolutions": [(1080, 1920), (720, 1280), (1440, 2560)],
        "system_fonts": ["Roboto", "Noto Sans"],
    },
    "ios": {
        "status_bar_height": 20,
        "navigation_bar_height": 44,
        "common_resolutions": [(750, 1334), (1125, 2436), (828, 1792)],
        "system_fonts": ["San Francisco", "Helvetica Neue"],
    }
}

APP_UI_PATTERNS = {
    "wechat": {
        "primary_color": "#07C160",
        "message_bubble_radius": 8,
        "avatar_size": 40,
    },
    "alipay": {
        "primary_color": "#1677FF",
        "header_height": 56,
        "button_radius": 4,
    },
    "banking_apps": {
        "security_indicators": ["lock_icon", "ssl_badge"],
        "amount_formats": [r"\d+\.\d{2}", r"¥\d+", r"\$\d+"],
    }
}

RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.8,
    "critical": 0.95,
}