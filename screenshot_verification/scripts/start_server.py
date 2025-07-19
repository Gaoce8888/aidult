#!/usr/bin/env python3
"""
服务器启动脚本
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings


def setup_logging(log_level: str = None):
    """设置日志"""
    if log_level is None:
        log_level = settings.log_level
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('server.log') if settings.log_file else logging.NullHandler()
        ]
    )


def check_dependencies():
    """检查依赖"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'torch',
        'opencv-python',
        'numpy',
        'pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    return True


def check_environment():
    """检查环境"""
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("需要Python 3.8或更高版本")
        return False
    
    # 检查必要的目录
    required_dirs = [
        settings.upload_dir,
        settings.temp_dir,
        settings.model_path
    ]
    
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return True


def start_server(host: str = None, port: int = None, workers: int = None, debug: bool = None):
    """启动服务器"""
    # 使用参数或默认配置
    host = host or settings.host
    port = port or settings.port
    workers = workers or settings.workers
    debug = debug if debug is not None else settings.debug
    
    print(f"启动手机截图验证服务...")
    print(f"主机: {host}")
    print(f"端口: {port}")
    print(f"工作进程: {workers}")
    print(f"调试模式: {debug}")
    print(f"版本: {settings.version}")
    
    # 导入并启动应用
    import uvicorn
    from main import app
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=debug,
        workers=workers if not debug else 1,
        log_level=settings.log_level.lower(),
        access_log=True
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="手机截图验证服务启动脚本")
    parser.add_argument("--host", default=settings.host, help="服务器主机地址")
    parser.add_argument("--port", type=int, default=settings.port, help="服务器端口")
    parser.add_argument("--workers", type=int, default=settings.workers, help="工作进程数")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default=settings.log_level, help="日志级别")
    parser.add_argument("--check-only", action="store_true", help="仅检查环境，不启动服务")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    
    print("=" * 50)
    print("手机截图AI识别真伪系统")
    print("=" * 50)
    
    # 检查依赖
    logger.info("检查依赖...")
    if not check_dependencies():
        sys.exit(1)
    
    # 检查环境
    logger.info("检查环境...")
    if not check_environment():
        sys.exit(1)
    
    if args.check_only:
        print("环境检查完成，所有依赖都已满足")
        return
    
    # 启动服务器
    try:
        start_server(
            host=args.host,
            port=args.port,
            workers=args.workers,
            debug=args.debug
        )
    except KeyboardInterrupt:
        logger.info("服务器被用户中断")
    except Exception as e:
        logger.error(f"启动服务器失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()