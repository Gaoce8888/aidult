# Screenshot Verifier

智能手机截图真伪识别服务 (Proof‐of‐Concept)

## 项目简介
本项目提供一套多层级检测架构，用于判断输入截图是否被篡改或伪造。系统集成了传统图像分析、深度学习模型以及多特征融合决策机制，并通过 FastAPI 暴露统一 HTTP 接口，支持本地、边缘与云端部署场景。

## 目录结构
```
screenshot_verifier/
├── app/                # 服务端源代码
│   ├── __init__.py
│   ├── main.py         # FastAPI 启动入口
│   ├── api/
│   │   ├── __init__.py
│   │   ├── schemas.py  # Pydantic 请求/响应模型
│   │   └── endpoints.py
│   └── detection/
│       ├── __init__.py
│       ├── traditional.py  # 传统图像检测算法
│       ├── ai_model.py     # 深度学习推理逻辑
│       └── fusion.py       # 多模态融合与置信度计算
├── requirements.txt    # 依赖清单
└── README.md           # 项目说明
```

## 快速开始
```bash
# 创建虚拟环境并安装依赖
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 运行服务
uvicorn screenshot_verifier.app.main:app --reload --host 0.0.0.0 --port 8000
```

## API 示例
POST `/api/v1/verify/screenshot`
```json
{
  "image": "base64_encoded_image",
  "metadata": {
    "source": "android",
    "app_type": "payment",
    "context": "extra info"
  }
}
```
响应:
```json
{
  "authentic": false,
  "confidence": 0.87,
  "risk_factors": [
    {
      "type": "compression_artifact",
      "severity": "medium",
      "location": [[10, 20, 120, 240]]
    }
  ],
  "detailed_report": null
}
```

## 开发计划
- [ ] 传统图像分析模块实现
- [ ] 预训练模型集成 (EfficientNet/ViT)
- [ ] 多特征融合策略
- [ ] 流水线性能优化 (<200 ms)
- [ ] 自动化测试与 CI