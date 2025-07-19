# Screenshot Authenticity AI - Docker Image
# 手机截图真伪检测系统Docker镜像

FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgeos-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制项目源代码
COPY src/ ./src/
COPY config/ ./config/
COPY run.py .
COPY .env.example .env

# 创建必要的目录
RUN mkdir -p logs models temp data/real data/fake data/preprocessed

# 设置权限
RUN chmod +x run.py

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000 8001

# 设置用户
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# 启动命令
CMD ["python", "run.py", "--mode", "api", "--host", "0.0.0.0", "--port", "8000"]