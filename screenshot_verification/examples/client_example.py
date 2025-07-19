#!/usr/bin/env python3
"""
示例客户端
演示如何使用手机截图验证API
"""
import requests
import base64
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import argparse


class ScreenshotVerificationClient:
    """截图验证客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "test-api-key-123"):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def _encode_image(self, image_path: str) -> str:
        """编码图像为Base64"""
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode()
    
    def verify_screenshot(self, image_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """验证单个截图"""
        if metadata is None:
            metadata = {}
        
        # 编码图像
        image_data = self._encode_image(image_path)
        
        # 准备请求数据
        request_data = {
            "image": image_data,
            "metadata": metadata,
            "source": "android",
            "app_type": "payment"
        }
        
        # 发送请求
        response = requests.post(
            f"{self.base_url}/api/v1/verify/screenshot",
            json=request_data,
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"请求失败: {response.status_code} - {response.text}")
    
    def batch_verify_screenshots(self, image_paths: List[str], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """批量验证截图"""
        if metadata is None:
            metadata = {}
        
        # 编码所有图像
        image_data_list = [self._encode_image(path) for path in image_paths]
        
        # 准备请求数据
        request_data = {
            "images": image_data_list,
            "metadata": metadata,
            "source": "android",
            "app_type": "payment"
        }
        
        # 发送请求
        response = requests.post(
            f"{self.base_url}/api/v1/verify/batch",
            json=request_data,
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"请求失败: {response.status_code} - {response.text}")
    
    def get_health(self) -> Dict[str, Any]:
        """获取健康状态"""
        response = requests.get(f"{self.base_url}/api/v1/health")
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"请求失败: {response.status_code} - {response.text}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        response = requests.get(f"{self.base_url}/api/v1/statistics", headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"请求失败: {response.status_code} - {response.text}")
    
    def get_models(self) -> List[Dict[str, Any]]:
        """获取模型信息"""
        response = requests.get(f"{self.base_url}/api/v1/models", headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"请求失败: {response.status_code} - {response.text}")
    
    def get_detectors(self) -> Dict[str, Any]:
        """获取检测器信息"""
        response = requests.get(f"{self.base_url}/api/v1/detectors", headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"请求失败: {response.status_code} - {response.text}")
    
    def submit_feedback(self, image_id: str, actual_label: bool, feedback_type: str, comments: str = "") -> Dict[str, Any]:
        """提交反馈"""
        request_data = {
            "image_id": image_id,
            "actual_label": actual_label,
            "feedback_type": feedback_type,
            "comments": comments
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/feedback",
            json=request_data,
            headers=self.headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"请求失败: {response.status_code} - {response.text}")


def print_result(result: Dict[str, Any], title: str = "验证结果"):
    """打印结果"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    if "authentic" in result:
        # 单个验证结果
        print(f"真实性: {'真实' if result['authentic'] else '伪造'}")
        print(f"置信度: {result['confidence']:.3f}")
        print(f"处理时间: {result['processing_time']:.3f}秒")
        
        if result['risk_factors']:
            print("\n风险因子:")
            for risk in result['risk_factors']:
                print(f"  - {risk['type']}: {risk['description']} (严重程度: {risk['severity']})")
        
        if result['detector_results']:
            print("\n检测器结果:")
            for name, detector_result in result['detector_results'].items():
                print(f"  - {name}: {'真实' if detector_result['is_authentic'] else '伪造'} "
                      f"(置信度: {detector_result['confidence']:.3f}, "
                      f"耗时: {detector_result['processing_time']:.3f}s)")
    
    elif "results" in result:
        # 批量验证结果
        print(f"总处理时间: {result['total_processing_time']:.3f}秒")
        print(f"成功数量: {result['success_count']}")
        print(f"失败数量: {result['failure_count']}")
        
        print("\n详细结果:")
        for i, res in enumerate(result['results']):
            print(f"  图像 {i+1}: {'真实' if res['authentic'] else '伪造'} "
                  f"(置信度: {res['confidence']:.3f})")
    
    else:
        # 其他类型的结果
        print(json.dumps(result, indent=2, ensure_ascii=False))


def create_test_image(output_path: str = "test_screenshot.png"):
    """创建测试图像"""
    from PIL import Image, ImageDraw, ImageFont
    
    # 创建一个模拟的手机截图
    width, height = 375, 812  # iPhone尺寸
    image = Image.new('RGB', (width, height), color='#f0f0f0')
    draw = ImageDraw.Draw(image)
    
    # 添加状态栏
    draw.rectangle([0, 0, width, 44], fill='#000000')
    draw.text((width//2, 22), "9:41", fill='white', anchor="mm")
    
    # 添加应用界面
    draw.rectangle([0, 44, width, height], fill='#ffffff')
    
    # 添加标题
    draw.text((width//2, 80), "支付成功", fill='#000000', anchor="mm")
    
    # 添加金额
    draw.text((width//2, 150), "¥1,000.00", fill='#007AFF', anchor="mm")
    
    # 添加详细信息
    details = [
        "交易时间: 2024-01-15 14:30:25",
        "交易号: 20240115143025001",
        "商户名称: 测试商户",
        "支付方式: 微信支付"
    ]
    
    y = 250
    for detail in details:
        draw.text((50, y), detail, fill='#666666')
        y += 30
    
    # 添加按钮
    draw.rectangle([50, height-100, width-50, height-60], fill='#007AFF')
    draw.text((width//2, height-80), "完成", fill='white', anchor="mm")
    
    # 保存图像
    image.save(output_path)
    print(f"测试图像已保存到: {output_path}")
    
    return output_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="手机截图验证客户端示例")
    parser.add_argument("--url", default="http://localhost:8000", help="API服务器地址")
    parser.add_argument("--api-key", default="test-api-key-123", help="API密钥")
    parser.add_argument("--image", help="要验证的图像路径")
    parser.add_argument("--create-test", action="store_true", help="创建测试图像")
    parser.add_argument("--health", action="store_true", help="检查服务健康状态")
    parser.add_argument("--stats", action="store_true", help="获取统计信息")
    parser.add_argument("--models", action="store_true", help="获取模型信息")
    parser.add_argument("--detectors", action="store_true", help="获取检测器信息")
    
    args = parser.parse_args()
    
    # 创建客户端
    client = ScreenshotVerificationClient(args.url, args.api_key)
    
    try:
        # 创建测试图像
        if args.create_test:
            test_image_path = create_test_image()
            print(f"测试图像已创建: {test_image_path}")
        
        # 检查健康状态
        if args.health:
            health = client.get_health()
            print_result(health, "服务健康状态")
        
        # 获取统计信息
        if args.stats:
            stats = client.get_statistics()
            print_result(stats, "统计信息")
        
        # 获取模型信息
        if args.models:
            models = client.get_models()
            print_result({"models": models}, "模型信息")
        
        # 获取检测器信息
        if args.detectors:
            detectors = client.get_detectors()
            print_result(detectors, "检测器信息")
        
        # 验证图像
        if args.image:
            if not Path(args.image).exists():
                print(f"错误: 图像文件不存在: {args.image}")
                return
            
            print(f"正在验证图像: {args.image}")
            result = client.verify_screenshot(args.image)
            print_result(result, "验证结果")
        
        # 如果没有指定任何操作，显示帮助
        if not any([args.create_test, args.health, args.stats, args.models, args.detectors, args.image]):
            print("请指定要执行的操作。使用 --help 查看选项。")
            
            # 创建测试图像并验证
            print("\n创建测试图像并验证...")
            test_image_path = create_test_image()
            result = client.verify_screenshot(test_image_path)
            print_result(result, "测试验证结果")
    
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()