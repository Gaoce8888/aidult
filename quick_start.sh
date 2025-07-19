#!/bin/bash

# Screenshot Authenticity AI - Quick Start Script
# 手机截图真伪检测系统快速启动脚本

set -e

echo "🔍 Screenshot Authenticity AI - Quick Start"
echo "=========================================="

# 检查Python版本
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -o "Python [0-9]\+\.[0-9]\+")
echo "✓ Found: $python_version"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "❌ Python 3.8+ is required"
    exit 1
fi

# 创建虚拟环境
echo "📦 Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# 激活虚拟环境
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# 升级pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# 安装依赖
echo "📚 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ Dependencies installed from requirements.txt"
else
    echo "⚠️  requirements.txt not found, installing basic dependencies..."
    pip install fastapi uvicorn torch torchvision timm opencv-python pillow scikit-image numpy pydantic structlog python-multipart aiofiles python-jose passlib bcrypt python-dotenv
fi

# 创建必要的目录
echo "📁 Creating project directories..."
python3 run.py --mode setup

# 检查系统健康
echo "🔍 Checking system health..."
python3 run.py --mode check

# 询问是否启动服务器
echo ""
echo "🚀 Setup completed! Ready to start the API server?"
echo "   Press Enter to start the server on http://localhost:8000"
echo "   Or Ctrl+C to exit"
read -r

# 启动服务器
echo "🌟 Starting Screenshot Authenticity AI API server..."
echo "   API Documentation: http://localhost:8000/docs"
echo "   Health Check: http://localhost:8000/health"
echo "   Press Ctrl+C to stop the server"
echo ""

python3 run.py --mode api --debug

echo ""
echo "👋 Server stopped. Thank you for using Screenshot Authenticity AI!"