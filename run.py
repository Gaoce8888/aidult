#!/usr/bin/env python3
"""
Screenshot Authenticity AI - Main entry point
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.main import main as api_main
from src.utils.logging_config import setup_logging


def setup_project_directories():
    """Create necessary project directories"""
    directories = [
        "logs",
        "models",
        "temp",
        "data/real",
        "data/fake",
        "data/preprocessed"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'torch', 'torchvision', 'timm',
        'opencv-python', 'pillow', 'scikit-image', 'numpy',
        'pydantic', 'structlog'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✓ All required dependencies are installed")
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Screenshot Authenticity AI System")
    parser.add_argument(
        '--mode', 
        choices=['api', 'setup', 'check'], 
        default='api',
        help='Operation mode (default: api)'
    )
    parser.add_argument(
        '--host', 
        default='0.0.0.0',
        help='Host to bind the API server (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8000,
        help='Port to bind the API server (default: 8000)'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    print("🔍 Screenshot Authenticity AI System")
    print("=" * 50)
    
    if args.mode == 'setup':
        print("🛠️  Setting up project...")
        setup_project_directories()
        
        if not check_dependencies():
            sys.exit(1)
        
        print("\n✅ Project setup completed successfully!")
        print("\nNext steps:")
        print("1. Configure settings in config/config.py")
        print("2. Start the API server: python run.py --mode api")
        
    elif args.mode == 'check':
        print("🔍 Checking system health...")
        
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)
        
        # Check configuration
        try:
            from config.config import settings
            print("✓ Configuration loaded successfully")
        except Exception as e:
            print(f"❌ Configuration error: {e}")
            sys.exit(1)
        
        # Check model availability
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️  CUDA not available, using CPU")
        except Exception as e:
            print(f"❌ PyTorch check failed: {e}")
        
        print("\n✅ System health check completed!")
        
    elif args.mode == 'api':
        print("🚀 Starting API server...")
        
        # Check dependencies first
        if not check_dependencies():
            sys.exit(1)
        
        # Setup directories
        setup_project_directories()
        
        # Set environment variables
        if args.debug:
            os.environ['DEBUG'] = 'true'
        
        os.environ['HOST'] = args.host
        os.environ['PORT'] = str(args.port)
        
        logger.info(f"Starting API server on {args.host}:{args.port}")
        logger.info(f"Debug mode: {args.debug}")
        
        try:
            # Import and run the API
            import uvicorn
            uvicorn.run(
                "src.api.main:app",
                host=args.host,
                port=args.port,
                reload=args.debug,
                access_log=True,
                log_level=args.log_level.lower()
            )
        except KeyboardInterrupt:
            print("\n\n🛑 Server stopped by user")
        except Exception as e:
            print(f"\n❌ Failed to start server: {e}")
            logger.error(f"Server startup failed: {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()