"""
API测试文件
包含基本的测试用例
"""
import pytest
import base64
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
import io

from main import app

client = TestClient(app)


def create_test_image(width=800, height=600, format='PNG'):
    """创建测试图像"""
    # 创建一个简单的测试图像
    image = Image.new('RGB', (width, height), color='white')
    
    # 添加一些内容
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(image)
    
    # 添加文字
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw.text((50, 50), "Test Screenshot", fill='black', font=font)
    draw.text((50, 100), "This is a test image for verification", fill='black', font=font)
    
    # 转换为Base64
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode()


class TestAPI:
    """API测试类"""
    
    def test_root_endpoint(self):
        """测试根端点"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
    
    def test_ping_endpoint(self):
        """测试ping端点"""
        response = client.get("/ping")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
    
    def test_info_endpoint(self):
        """测试info端点"""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "features" in data
        assert "endpoints" in data
    
    def test_health_check_without_auth(self):
        """测试健康检查（无需认证）"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime" in data
        assert "detector_status" in data
    
    def test_verify_screenshot_without_auth(self):
        """测试截图验证（无认证）"""
        test_image = create_test_image()
        
        response = client.post("/api/v1/verify/screenshot", json={
            "image": test_image,
            "metadata": {},
            "source": "android",
            "app_type": "payment"
        })
        
        # 应该返回401未授权
        assert response.status_code == 401
    
    def test_verify_screenshot_with_auth(self):
        """测试截图验证（有认证）"""
        test_image = create_test_image()
        
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        response = client.post("/api/v1/verify/screenshot", json={
            "image": test_image,
            "metadata": {},
            "source": "android",
            "app_type": "payment"
        }, headers=headers)
        
        # 应该返回200成功
        assert response.status_code == 200
        data = response.json()
        
        # 检查响应结构
        assert "authentic" in data
        assert "confidence" in data
        assert "risk_factors" in data
        assert "processing_time" in data
        assert "detector_results" in data
        
        # 检查数据类型
        assert isinstance(data["authentic"], bool)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["risk_factors"], list)
        assert isinstance(data["processing_time"], float)
        assert isinstance(data["detector_results"], dict)
        
        # 检查置信度范围
        assert 0.0 <= data["confidence"] <= 1.0
    
    def test_verify_screenshot_invalid_image(self):
        """测试无效图像"""
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        response = client.post("/api/v1/verify/screenshot", json={
            "image": "invalid-base64-data",
            "metadata": {},
            "source": "android",
            "app_type": "payment"
        }, headers=headers)
        
        # 应该返回400错误
        assert response.status_code == 400
    
    def test_verify_screenshot_empty_image(self):
        """测试空图像"""
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        response = client.post("/api/v1/verify/screenshot", json={
            "image": "",
            "metadata": {},
            "source": "android",
            "app_type": "payment"
        }, headers=headers)
        
        # 应该返回422验证错误
        assert response.status_code == 422
    
    def test_batch_verify(self):
        """测试批量验证"""
        test_images = [create_test_image() for _ in range(3)]
        
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        response = client.post("/api/v1/verify/batch", json={
            "images": test_images,
            "metadata": {},
            "source": "android",
            "app_type": "payment"
        }, headers=headers)
        
        # 应该返回200成功
        assert response.status_code == 200
        data = response.json()
        
        # 检查响应结构
        assert "results" in data
        assert "total_processing_time" in data
        assert "success_count" in data
        assert "failure_count" in data
        
        # 检查结果数量
        assert len(data["results"]) == 3
        assert data["success_count"] == 3
        assert data["failure_count"] == 0
    
    def test_batch_verify_too_many_images(self):
        """测试批量验证（图像过多）"""
        test_images = [create_test_image() for _ in range(15)]  # 超过10张限制
        
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        response = client.post("/api/v1/verify/batch", json={
            "images": test_images,
            "metadata": {},
            "source": "android",
            "app_type": "payment"
        }, headers=headers)
        
        # 应该返回422验证错误
        assert response.status_code == 422
    
    def test_get_statistics_with_auth(self):
        """测试获取统计信息（有认证）"""
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        response = client.get("/api/v1/statistics", headers=headers)
        
        # 应该返回200成功
        assert response.status_code == 200
        data = response.json()
        
        # 检查响应结构
        assert "total_requests" in data
        assert "successful_requests" in data
        assert "failed_requests" in data
        assert "average_processing_time" in data
        assert "authentic_count" in data
        assert "fake_count" in data
        assert "detector_performance" in data
    
    def test_get_models_with_auth(self):
        """测试获取模型信息（有认证）"""
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        response = client.get("/api/v1/models", headers=headers)
        
        # 应该返回200成功
        assert response.status_code == 200
        data = response.json()
        
        # 检查响应结构
        assert isinstance(data, list)
        assert len(data) > 0
        
        for model in data:
            assert "model_name" in model
            assert "model_version" in model
            assert "model_type" in model
            assert "accuracy" in model
            assert "last_updated" in model
            assert "parameters" in model
    
    def test_get_detectors_with_auth(self):
        """测试获取检测器信息（有认证）"""
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        response = client.get("/api/v1/detectors", headers=headers)
        
        # 应该返回200成功
        assert response.status_code == 200
        data = response.json()
        
        # 检查响应结构
        assert "traditional" in data
        assert "metadata" in data
        assert "ai" in data
        
        for detector in data.values():
            assert "name" in detector
            assert "description" in detector
            assert "enabled" in detector
            assert "capabilities" in detector
    
    def test_submit_feedback(self):
        """测试提交反馈"""
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        response = client.post("/api/v1/feedback", json={
            "image_id": "test-image-123",
            "actual_label": True,
            "feedback_type": "false_positive",
            "comments": "This is a test feedback"
        }, headers=headers)
        
        # 应该返回200成功
        assert response.status_code == 200
        data = response.json()
        
        # 检查响应结构
        assert "success" in data
        assert "message" in data
        assert "feedback_id" in data
        
        assert data["success"] == True
    
    def test_clear_cache_with_auth(self):
        """测试清除缓存（有认证）"""
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        response = client.delete("/api/v1/cache", headers=headers)
        
        # 应该返回200成功
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "缓存清除成功" in data["message"]


class TestErrorHandling:
    """错误处理测试类"""
    
    def test_invalid_api_key(self):
        """测试无效API密钥"""
        headers = {"Authorization": "Bearer invalid-key"}
        
        response = client.get("/api/v1/statistics", headers=headers)
        
        # 应该返回401未授权
        assert response.status_code == 401
        data = response.json()
        assert "无效的API密钥" in data["error"]
    
    def test_missing_api_key(self):
        """测试缺少API密钥"""
        response = client.get("/api/v1/statistics")
        
        # 应该返回401未授权
        assert response.status_code == 401
        data = response.json()
        assert "缺少API密钥" in data["error"]
    
    def test_invalid_feedback_type(self):
        """测试无效反馈类型"""
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        response = client.post("/api/v1/feedback", json={
            "image_id": "test-image-123",
            "actual_label": True,
            "feedback_type": "invalid_type",
            "comments": "Test feedback"
        }, headers=headers)
        
        # 应该返回422验证错误
        assert response.status_code == 422


class TestPerformance:
    """性能测试类"""
    
    def test_concurrent_requests(self):
        """测试并发请求"""
        import threading
        import time
        
        test_image = create_test_image()
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        results = []
        errors = []
        
        def make_request():
            try:
                response = client.post("/api/v1/verify/screenshot", json={
                    "image": test_image,
                    "metadata": {},
                    "source": "android",
                    "app_type": "payment"
                }, headers=headers)
                results.append(response.status_code)
            except Exception as e:
                errors.append(str(e))
        
        # 创建10个并发请求
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 检查结果
        assert len(errors) == 0, f"并发请求出现错误: {errors}"
        assert len(results) == 10
        assert all(status == 200 for status in results)
    
    def test_large_image(self):
        """测试大图像处理"""
        # 创建一个较大的测试图像
        test_image = create_test_image(width=1920, height=1080)
        
        headers = {"Authorization": "Bearer test-api-key-123"}
        
        start_time = time.time()
        
        response = client.post("/api/v1/verify/screenshot", json={
            "image": test_image,
            "metadata": {},
            "source": "android",
            "app_type": "payment"
        }, headers=headers)
        
        processing_time = time.time() - start_time
        
        # 应该返回200成功
        assert response.status_code == 200
        
        # 处理时间应该在合理范围内（小于5秒）
        assert processing_time < 5.0, f"大图像处理时间过长: {processing_time:.2f}s"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])