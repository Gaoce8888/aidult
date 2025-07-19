#!/usr/bin/env python3
"""
简单的API测试脚本
"""
import requests
import json
import time
import base64
from pathlib import Path
import argparse


def test_health_check(base_url):
    """测试健康检查端点"""
    print("🏥 Testing health check...")
    
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")


def test_root_endpoint(base_url):
    """测试根端点"""
    print("\n🏠 Testing root endpoint...")
    
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            print("✅ Root endpoint works")
            data = response.json()
            print(f"   API Version: {data.get('version', 'Unknown')}")
            print(f"   Status: {data.get('status', 'Unknown')}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")


def test_screenshot_verification(base_url, image_path, api_key):
    """测试截图验证"""
    print(f"\n📱 Testing screenshot verification with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        return
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    } if api_key else {}
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'context': json.dumps({
                    'source': 'android',
                    'app_type': 'payment',
                    'priority': 'normal'
                })
            }
            
            start_time = time.time()
            response = requests.post(
                f"{base_url}/api/v1/verify/screenshot",
                files=files,
                data=data,
                headers=headers
            )
            end_time = time.time()
            
            print(f"   Request time: {(end_time - start_time) * 1000:.1f}ms")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Screenshot verification successful")
                print(f"   Authentic: {result.get('authentic')}")
                print(f"   Confidence: {result.get('confidence', 0):.3f}")
                print(f"   Risk Level: {result.get('risk_assessment', {}).get('risk_level', 'Unknown')}")
                print(f"   Analysis Time: {result.get('analysis_time_ms', 0):.1f}ms")
                print(f"   Request ID: {result.get('request_id', 'Unknown')}")
            elif response.status_code == 401:
                print("❌ Authentication required - provide API key")
            elif response.status_code == 413:
                print("❌ File too large")
            elif response.status_code == 415:
                print("❌ Unsupported file format")
            else:
                print(f"❌ Verification failed: {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"   Error: {error_detail.get('detail', 'Unknown error')}")
                except:
                    print(f"   Raw response: {response.text}")
                    
    except Exception as e:
        print(f"❌ Verification error: {e}")


def test_system_status(base_url, api_key):
    """测试系统状态"""
    print(f"\n📊 Testing system status...")
    
    if not api_key:
        print("⚠️  Skipping system status test - API key required")
        return
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        response = requests.get(f"{base_url}/status", headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ System status retrieved")
            print(f"   Status: {result.get('status', 'Unknown')}")
            print(f"   Version: {result.get('version', 'Unknown')}")
            print(f"   Uptime: {result.get('uptime_seconds', 0):.1f}s")
            
            detectors = result.get('detectors', {})
            print("   Detectors:")
            for detector, status in detectors.items():
                status_icon = "✅" if status else "❌"
                print(f"     {status_icon} {detector}")
                
        elif response.status_code == 401:
            print("❌ Authentication failed")
        else:
            print(f"❌ System status failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ System status error: {e}")


def create_test_image():
    """创建一个简单的测试图片"""
    try:
        from PIL import Image, ImageDraw
        import numpy as np
        
        # 创建一个简单的测试图片
        img = Image.new('RGB', (400, 800), color='white')
        draw = ImageDraw.Draw(img)
        
        # 模拟手机截图界面
        # 状态栏
        draw.rectangle([0, 0, 400, 50], fill='black')
        draw.text((10, 15), "9:41", fill='white')
        draw.text((350, 15), "100%", fill='white')
        
        # 应用标题栏
        draw.rectangle([0, 50, 400, 100], fill='blue')
        draw.text((150, 65), "测试应用", fill='white')
        
        # 内容区域
        draw.rectangle([20, 120, 380, 200], fill='lightgray')
        draw.text((30, 140), "这是一个测试截图", fill='black')
        draw.text((30, 160), "用于验证API功能", fill='black')
        
        # 保存图片
        test_image_path = "test_screenshot.png"
        img.save(test_image_path)
        print(f"✅ Created test image: {test_image_path}")
        return test_image_path
        
    except ImportError:
        print("⚠️  PIL not available, cannot create test image")
        return None
    except Exception as e:
        print(f"❌ Error creating test image: {e}")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Screenshot Authenticity AI API Test")
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--api-key', help='API key for authentication')
    parser.add_argument('--image', help='Path to test image file')
    parser.add_argument('--create-test-image', action='store_true', help='Create a test image')
    
    args = parser.parse_args()
    
    print("🧪 Screenshot Authenticity AI - API Test")
    print("=" * 50)
    print(f"API URL: {args.url}")
    
    # 基础测试
    test_health_check(args.url)
    test_root_endpoint(args.url)
    
    # 如果需要创建测试图片
    if args.create_test_image:
        test_image_path = create_test_image()
        if test_image_path:
            args.image = test_image_path
    
    # 截图验证测试
    if args.image:
        test_screenshot_verification(args.url, args.image, args.api_key)
    else:
        print("\n⚠️  No test image provided. Use --image or --create-test-image")
    
    # 系统状态测试
    test_system_status(args.url, args.api_key)
    
    print("\n🏁 Test completed!")
    print("\nExample usage:")
    print("  # Basic test")
    print("  python test_api.py")
    print("  # Test with image")
    print("  python test_api.py --image screenshot.jpg")
    print("  # Test with API key")
    print("  python test_api.py --api-key sa_your_api_key_here")
    print("  # Create and test with generated image")
    print("  python test_api.py --create-test-image")


if __name__ == "__main__":
    main()