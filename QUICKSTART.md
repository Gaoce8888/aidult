# 🚀 Screenshot Authenticity AI - 快速开始指南

这是一个完整的手机截图AI识别真伪项目，采用多层检测架构，结合传统图像分析、元数据分析和深度学习技术。

## 📁 项目结构

```
screenshot-authenticity-ai/
├── 📂 src/                          # 源代码
│   ├── 📂 api/                      # FastAPI接口
│   │   └── main.py                  # 主API应用
│   ├── 📂 core/                     # 核心检测引擎
│   │   └── authenticity_engine.py   # 真伪检测引擎
│   ├── 📂 detectors/                # 检测器模块
│   │   ├── traditional_detector.py  # 传统图像分析
│   │   └── metadata_detector.py     # 元数据分析
│   ├── 📂 models/                   # 深度学习模型
│   │   └── deep_learning_models.py  # AI模型定义
│   └── 📂 utils/                    # 工具类
│       ├── logging_config.py        # 日志配置
│       ├── rate_limiter.py          # 限流器
│       └── security.py              # 安全管理
├── 📂 config/                       # 配置文件
│   └── config.py                    # 主配置
├── 📂 data/                         # 数据目录
│   ├── real/                        # 真实截图
│   └── fake/                        # 伪造截图
├── 📂 models/                       # 模型文件
├── 📂 logs/                         # 日志文件
├── 📂 temp/                         # 临时文件
├── 📄 requirements.txt              # Python依赖
├── 📄 run.py                        # 主启动脚本
├── 📄 test_api.py                   # API测试脚本
├── 📄 quick_start.sh                # 快速启动脚本
├── 📄 Dockerfile                    # Docker镜像
├── 📄 docker-compose.yml            # Docker编排
├── 📄 .env.example                  # 环境变量示例
└── 📄 README.md                     # 详细文档
```

## 🎯 核心功能

### 🔍 多层检测架构
- **第一阶段** (< 50ms): 基础检查 - 分辨率、格式验证
- **第二阶段** (< 200ms): 深度分析 - 并行运行多种检测方法  
- **第三阶段** (< 1000ms): 高级验证 - 集成模型综合判断

### 🧠 检测技术
1. **传统图像分析**
   - 压缩伪影检测 (JPEG块效应、双重压缩)
   - 噪声模式分析 (局部不一致性、纹理分析)  
   - 边缘一致性检测 (锐度分析、混合伪影)

2. **元数据分析**
   - EXIF信息验证 (设备信息、时间戳)
   - 文件属性检查 (创建时间、格式一致性)
   - 图像指纹计算 (多种哈希算法)
   - 隐写检测 (LSB分析、频域分析)

3. **深度学习模型**
   - EfficientNet (轻量级特征提取)
   - Vision Transformer (全局特征理解)
   - 双流网络 (RGB + 频域分析)
   - 多尺度网络 (不同粒度特征)
   - 集成模型 (多模型投票决策)

## ⚡ 快速启动

### 方法一：一键启动脚本

```bash
# 下载项目
git clone <your-repo-url>
cd screenshot-authenticity-ai

# 运行快速启动脚本
chmod +x quick_start.sh
./quick_start.sh
```

### 方法二：手动安装

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 项目设置
python run.py --mode setup

# 4. 检查系统
python run.py --mode check

# 5. 启动API服务
python run.py --mode api --debug
```

### 方法三：Docker部署

```bash
# 单容器启动
docker build -t screenshot-ai .
docker run -p 8000:8000 screenshot-ai

# 完整服务栈
docker-compose up -d
```

## 🧪 测试API

启动服务后，可以使用以下方法测试：

### 1. 基础健康检查
```bash
curl http://localhost:8000/health
```

### 2. 使用测试脚本
```bash
# 创建测试图片并测试
python test_api.py --create-test-image

# 使用现有图片测试
python test_api.py --image your_screenshot.jpg

# 带认证测试
python test_api.py --api-key sa_your_api_key_here
```

### 3. 手动API调用
```bash
curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "file=@screenshot.jpg" \
  -F "context={\"source\":\"android\",\"app_type\":\"payment\"}" \
  http://localhost:8000/api/v1/verify/screenshot
```

## 📊 API响应示例

```json
{
  "authentic": true,
  "confidence": 0.92,
  "risk_assessment": {
    "overall_risk_score": 0.15,
    "risk_level": "LOW",
    "confidence": 0.88,
    "evidence": []
  },
  "risk_factors": [],
  "detection_summary": {
    "methods_used": ["traditional", "metadata", "deep_learning"],
    "stages_completed": [1, 2],
    "total_analysis_time_ms": 156.7,
    "performance_sla_met": true
  },
  "request_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "analysis_time_ms": 156.7
}
```

## 🔧 配置说明

复制 `.env.example` 为 `.env` 并根据需要修改：

```bash
cp .env.example .env
```

关键配置项：
- `DEBUG=true` - 开发模式
- `USE_GPU=true` - 启用GPU加速  
- `LOG_LEVEL=INFO` - 日志级别
- `ENABLE_DEEP_LEARNING=true` - 启用AI模型

## 📈 监控和运维

### API文档
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 监控端点
- 健康检查: http://localhost:8000/health
- 系统状态: http://localhost:8000/status
- 指标收集: http://localhost:8001/metrics

### 日志查看
```bash
# 查看实时日志
tail -f logs/app.log

# 查看错误日志
grep "ERROR" logs/app.log
```

## 🏗️ 应用场景

- **金融风控**: 贷款凭证、转账截图验证
- **电商平台**: 评价截图、订单凭证审核  
- **保险理赔**: 事故照片、医疗凭证验证
- **法律证据**: 聊天记录、转账凭证审核

## 📞 问题排查

### 常见问题

1. **依赖安装失败**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --no-cache-dir
   ```

2. **GPU不可用**  
   ```bash
   # 检查CUDA
   python -c "import torch; print(torch.cuda.is_available())"
   
   # 设置使用CPU
   export USE_GPU=false
   ```

3. **端口被占用**
   ```bash
   # 使用其他端口
   python run.py --port 8080
   ```

4. **权限问题**
   ```bash
   chmod +x run.py quick_start.sh test_api.py
   ```

### 获取帮助

```bash
# 查看帮助
python run.py --help

# 运行健康检查
python run.py --mode check

# 查看系统状态
curl http://localhost:8000/status
```

## 📈 性能指标

- **准确率**: 95%+
- **响应时间**: < 1000ms  
- **并发能力**: 100+ QPS
- **资源使用**: < 2GB RAM, < 4GB GPU

## 🤝 贡献和支持

- 🐛 问题反馈: [GitHub Issues]
- 💬 讨论交流: [GitHub Discussions]  
- 📚 详细文档: [README.md](README.md)
- 📧 邮箱支持: support@example.com

---

**🔍 让AI守护数字世界的真实性！**