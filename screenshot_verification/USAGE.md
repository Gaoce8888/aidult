# 手机截图AI识别真伪系统 - 使用说明

## 快速开始

### 1. 环境要求

- Python 3.8+
- 8GB+ RAM
- CUDA 11.0+ (可选，用于GPU加速)

### 2. 快速启动

#### 使用快速启动脚本（推荐）

```bash
# 克隆项目后，进入项目目录
cd screenshot_verification

# 给启动脚本添加执行权限
chmod +x quick_start.sh

# 启动服务
./quick_start.sh start

# 或者使用Docker启动
./quick_start.sh docker
```

#### 手动启动

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python scripts/start_server.py --debug
```

### 3. 验证服务

服务启动后，可以通过以下方式验证：

```bash
# 检查健康状态
curl http://localhost:8000/ping

# 查看API文档
# 浏览器访问: http://localhost:8000/docs
```

## API使用

### 1. 基本认证

所有API请求都需要API密钥认证：

```bash
# 在请求头中添加
Authorization: Bearer test-api-key-123
```

### 2. 验证单个截图

```python
import requests
import base64

# 读取图像文件
with open("screenshot.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# 发送验证请求
response = requests.post("http://localhost:8000/api/v1/verify/screenshot", 
    json={
        "image": image_data,
        "metadata": {
            "source": "android",
            "app_type": "payment"
        }
    },
    headers={"Authorization": "Bearer test-api-key-123"}
)

result = response.json()
print(f"真实性: {result['authentic']}")
print(f"置信度: {result['confidence']}")
```

### 3. 批量验证

```python
# 批量验证多张截图
images = [image1_data, image2_data, image3_data]

response = requests.post("http://localhost:8000/api/v1/verify/batch",
    json={
        "images": images,
        "metadata": {"source": "android"}
    },
    headers={"Authorization": "Bearer test-api-key-123"}
)

results = response.json()
for i, result in enumerate(results['results']):
    print(f"图像 {i+1}: {'真实' if result['authentic'] else '伪造'}")
```

### 4. 获取统计信息

```python
response = requests.get("http://localhost:8000/api/v1/statistics",
    headers={"Authorization": "Bearer test-api-key-123"}
)

stats = response.json()
print(f"总请求数: {stats['total_requests']}")
print(f"成功率: {stats['success_rate']:.2%}")
```

## 示例客户端

项目包含了一个完整的示例客户端：

```bash
# 运行示例客户端
python examples/client_example.py --create-test

# 验证指定图像
python examples/client_example.py --image path/to/screenshot.png

# 查看所有选项
python examples/client_example.py --help
```

## 配置说明

### 环境变量

可以通过环境变量或`.env`文件配置系统：

```bash
# 服务器配置
HOST=0.0.0.0
PORT=8000
DEBUG=false

# 模型配置
MODEL_DEVICE=auto  # auto, cpu, cuda
CONFIDENCE_THRESHOLD=0.8

# 缓存配置
REDIS_URL=redis://localhost:6379
CACHE_TTL=3600

# 安全配置
SECRET_KEY=your-secret-key-here
RATE_LIMIT_PER_MINUTE=100
```

### 检测器配置

可以启用/禁用不同的检测器：

```bash
# 传统图像分析检测器
TRADITIONAL_DETECTOR_ENABLED=true

# 元数据分析检测器
METADATA_DETECTOR_ENABLED=true

# AI深度学习检测器
AI_DETECTOR_ENABLED=true
```

## 部署选项

### 1. 本地部署

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
```

### 2. Docker部署

```bash
# 构建镜像
docker build -t screenshot-verification .

# 运行容器
docker run -p 8000:8000 screenshot-verification
```

### 3. Docker Compose部署

```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 4. 生产环境部署

#### 使用Nginx反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

#### 使用Gunicorn

```bash
# 安装Gunicorn
pip install gunicorn

# 启动服务
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 监控和日志

### 1. 健康检查

```bash
# 检查服务状态
curl http://localhost:8000/api/v1/health
```

### 2. 性能监控

```bash
# 获取统计信息
curl http://localhost:8000/api/v1/statistics
```

### 3. 日志配置

```python
# 在config/settings.py中配置
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

## 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 检查端口占用
   lsof -i :8000
   
   # 停止占用进程
   kill -9 <PID>
   ```

2. **依赖安装失败**
   ```bash
   # 升级pip
   pip install --upgrade pip
   
   # 使用国内镜像
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

3. **CUDA相关错误**
   ```bash
   # 检查CUDA版本
   nvidia-smi
   
   # 安装对应版本的PyTorch
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **内存不足**
   ```bash
   # 减少工作进程数
   python scripts/start_server.py --workers 1
   
   # 或者使用CPU模式
   export MODEL_DEVICE=cpu
   ```

### 调试模式

```bash
# 启用调试模式
python scripts/start_server.py --debug --log-level DEBUG
```

## 开发指南

### 1. 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_api.py::TestAPI::test_verify_screenshot_with_auth -v
```

### 2. 代码格式化

```bash
# 格式化代码
black .

# 检查代码风格
flake8 .

# 类型检查
mypy .
```

### 3. 添加新的检测器

1. 在`core/detectors.py`中创建新的检测器类
2. 继承`BaseDetector`类
3. 实现`detect`方法
4. 在`core/verification_engine.py`中注册检测器

### 4. 添加新的API端点

1. 在`api/models.py`中定义请求/响应模型
2. 在`api/routes.py`中添加路由
3. 实现业务逻辑
4. 添加测试用例

## 性能优化

### 1. 缓存配置

```bash
# 启用Redis缓存
REDIS_URL=redis://localhost:6379
CACHE_ENABLED=true
CACHE_TTL=3600
```

### 2. 模型优化

```bash
# 使用GPU加速
MODEL_DEVICE=cuda

# 使用量化模型
# 在模型加载时启用量化
```

### 3. 并发配置

```bash
# 增加工作进程数
WORKERS=4

# 调整线程池大小
# 在verification_engine.py中修改ThreadPoolExecutor的max_workers
```

## 安全考虑

### 1. API密钥管理

- 使用强密码生成API密钥
- 定期轮换API密钥
- 限制API密钥的权限范围

### 2. 输入验证

- 验证图像格式和大小
- 限制请求频率
- 过滤恶意输入

### 3. 数据保护

- 加密敏感数据
- 定期清理临时文件
- 限制数据访问权限

## 支持

如果遇到问题，请：

1. 查看日志文件
2. 检查配置是否正确
3. 运行测试用例
4. 提交Issue到项目仓库

## 许可证

本项目采用MIT许可证，详见LICENSE文件。