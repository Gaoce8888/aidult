"""
Logging configuration for the Screenshot Authenticity AI system
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import structlog
from config.config import settings


def setup_logging(log_level: Optional[str] = None):
    """
    Setup logging configuration for the application
    
    Args:
        log_level: Override log level from config
    """
    # Get log level
    level = log_level or settings.logging.level
    log_level_int = getattr(logging, level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    if settings.logging.file_path:
        log_file_path = Path(settings.logging.file_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_int)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level_int)
    
    # File handler (if configured)
    file_handler = None
    if settings.logging.file_path:
        file_handler = logging.handlers.RotatingFileHandler(
            settings.logging.file_path,
            maxBytes=settings.logging.max_file_size,
            backupCount=settings.logging.backup_count
        )
        file_handler.setLevel(log_level_int)
    
    # Create formatter
    formatter = logging.Formatter(
        settings.logging.format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Apply formatter to handlers
    console_handler.setFormatter(formatter)
    if file_handler:
        file_handler.setFormatter(formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set logging levels for specific modules
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    logging.info("Logging configuration completed")


def get_logger(name: str):
    """Get a configured logger instance"""
    return structlog.get_logger(name)