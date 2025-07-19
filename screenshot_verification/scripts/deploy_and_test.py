#!/usr/bin/env python3
"""
部署测试与生产环境脚本
包括生产环境部署、性能测试、负载测试和监控设置
"""
import os
import sys
import argparse
import logging
import json
import time
import subprocess
import requests
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import docker
import kubernetes
from kubernetes import client, config
import prometheus_client
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import psutil
import base64
from PIL import Image
import io

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings


class ProductionDeployer:
    """生产环境部署器"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        
        # 加载部署配置
        with open(self.config_path, 'r') as f:
            self.deploy_config = json.load(f)
    
    def deploy(self, environment: str = 'production'):
        """部署到生产环境"""
        self.logger.info(f"开始部署到 {environment} 环境")
        
        # 1. 环境检查
        self._check_environment()
        
        # 2. 构建镜像
        self._build_docker_image()
        
        # 3. 推送镜像
        self._push_docker_image()
        
        # 4. 部署服务
        if self.deploy_config.get('use_kubernetes', False):
            self._deploy_kubernetes()
        else:
            self._deploy_docker_compose()
        
        # 5. 健康检查
        self._health_check()
        
        self.logger.info("部署完成")
    
    def _check_environment(self):
        """检查部署环境"""
        self.logger.info("检查部署环境")
        
        # 检查Docker
        try:
            client = docker.from_env()
            client.ping()
            self.logger.info("Docker 环境正常")
        except Exception as e:
            self.logger.error(f"Docker 环境检查失败: {e}")
            raise
        
        # 检查Kubernetes（如果使用）
        if self.deploy_config.get('use_kubernetes', False):
            try:
                config.load_kube_config()
                v1 = client.CoreV1Api()
                v1.list_namespace()
                self.logger.info("Kubernetes 环境正常")
            except Exception as e:
                self.logger.error(f"Kubernetes 环境检查失败: {e}")
                raise
        
        # 检查资源
        self._check_resources()
    
    def _check_resources(self):
        """检查系统资源"""
        # CPU
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        self.logger.info(f"CPU: {cpu_count} 核心, 使用率: {cpu_percent}%")
        
        # 内存
        memory = psutil.virtual_memory()
        self.logger.info(f"内存: {memory.total / 1024**3:.1f}GB, 使用率: {memory.percent}%")
        
        # 磁盘
        disk = psutil.disk_usage('/')
        self.logger.info(f"磁盘: {disk.total / 1024**3:.1f}GB, 使用率: {disk.percent}%")
        
        # 检查资源是否足够
        if cpu_percent > 80:
            self.logger.warning("CPU 使用率过高")
        if memory.percent > 80:
            self.logger.warning("内存使用率过高")
        if disk.percent > 80:
            self.logger.warning("磁盘使用率过高")
    
    def _build_docker_image(self):
        """构建Docker镜像"""
        self.logger.info("构建Docker镜像")
        
        try:
            client = docker.from_env()
            
            # 构建镜像
            image, logs = client.images.build(
                path=str(Path(__file__).parent.parent),
                dockerfile='Dockerfile',
                tag=self.deploy_config['docker_image'],
                rm=True
            )
            
            self.logger.info(f"Docker镜像构建成功: {image.tags}")
            
        except Exception as e:
            self.logger.error(f"Docker镜像构建失败: {e}")
            raise
    
    def _push_docker_image(self):
        """推送Docker镜像"""
        if not self.deploy_config.get('push_image', False):
            return
        
        self.logger.info("推送Docker镜像")
        
        try:
            client = docker.from_env()
            
            # 登录到镜像仓库
            if 'registry_auth' in self.deploy_config:
                auth = self.deploy_config['registry_auth']
                client.login(
                    username=auth['username'],
                    password=auth['password'],
                    registry=auth['registry']
                )
            
            # 推送镜像
            for line in client.images.push(self.deploy_config['docker_image'], stream=True):
                self.logger.debug(line.decode().strip())
            
            self.logger.info("Docker镜像推送成功")
            
        except Exception as e:
            self.logger.error(f"Docker镜像推送失败: {e}")
            raise
    
    def _deploy_kubernetes(self):
        """部署到Kubernetes"""
        self.logger.info("部署到Kubernetes")
        
        try:
            config.load_kube_config()
            
            # 创建命名空间
            self._create_namespace()
            
            # 部署ConfigMap
            self._deploy_configmap()
            
            # 部署Secret
            self._deploy_secret()
            
            # 部署服务
            self._deploy_service()
            
            # 部署Deployment
            self._deploy_deployment()
            
            # 部署Ingress
            if self.deploy_config.get('use_ingress', False):
                self._deploy_ingress()
            
            self.logger.info("Kubernetes部署成功")
            
        except Exception as e:
            self.logger.error(f"Kubernetes部署失败: {e}")
            raise
    
    def _deploy_docker_compose(self):
        """使用Docker Compose部署"""
        self.logger.info("使用Docker Compose部署")
        
        try:
            # 生成docker-compose.yml
            self._generate_docker_compose()
            
            # 启动服务
            subprocess.run([
                'docker-compose', '-f', 'docker-compose.prod.yml', 'up', '-d'
            ], check=True)
            
            self.logger.info("Docker Compose部署成功")
            
        except Exception as e:
            self.logger.error(f"Docker Compose部署失败: {e}")
            raise
    
    def _health_check(self):
        """健康检查"""
        self.logger.info("执行健康检查")
        
        # 等待服务启动
        time.sleep(10)
        
        # 检查API端点
        base_url = self.deploy_config.get('base_url', 'http://localhost:8000')
        
        endpoints = [
            '/api/v1/health',
            '/api/v1/statistics',
            '/api/v1/models'
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
                if response.status_code == 200:
                    self.logger.info(f"健康检查通过: {endpoint}")
                else:
                    self.logger.error(f"健康检查失败: {endpoint}, 状态码: {response.status_code}")
            except Exception as e:
                self.logger.error(f"健康检查失败: {endpoint}, 错误: {e}")


class PerformanceTester:
    """性能测试器"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
        
        # 初始化Prometheus指标
        self.request_counter = Counter('api_requests_total', 'Total API requests', ['endpoint', 'status'])
        self.request_duration = Histogram('api_request_duration_seconds', 'API request duration', ['endpoint'])
        self.active_requests = Gauge('api_active_requests', 'Active API requests')
    
    def run_performance_test(self, test_config: Dict):
        """运行性能测试"""
        self.logger.info("开始性能测试")
        
        # 启动Prometheus服务器
        start_http_server(8001)
        
        # 1. 单请求性能测试
        self._single_request_test()
        
        # 2. 并发性能测试
        self._concurrent_test(test_config.get('concurrent_users', 10))
        
        # 3. 负载测试
        self._load_test(test_config.get('load_duration', 300))
        
        # 4. 压力测试
        self._stress_test(test_config.get('stress_users', 100))
        
        # 5. 生成测试报告
        self._generate_test_report()
        
        self.logger.info("性能测试完成")
    
    def _single_request_test(self):
        """单请求性能测试"""
        self.logger.info("执行单请求性能测试")
        
        # 准备测试图像
        test_image = self._create_test_image()
        
        # 测试单个请求
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/verify/screenshot",
                json={
                    "image": test_image,
                    "source": "android",
                    "app_type": "payment",
                    "context": "test"
                },
                headers={"X-API-Key": "test-key"},
                timeout=30
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                self.logger.info(f"单请求测试成功，耗时: {duration:.3f}s")
                self.request_duration.labels(endpoint='/verify/screenshot').observe(duration)
            else:
                self.logger.error(f"单请求测试失败，状态码: {response.status_code}")
            
        except Exception as e:
            self.logger.error(f"单请求测试异常: {e}")
    
    def _concurrent_test(self, num_users: int):
        """并发性能测试"""
        self.logger.info(f"执行并发性能测试，用户数: {num_users}")
        
        test_image = self._create_test_image()
        results = []
        
        def make_request(user_id: int):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/verify/screenshot",
                    json={
                        "image": test_image,
                        "source": "android",
                        "app_type": "payment",
                        "context": f"concurrent_test_user_{user_id}"
                    },
                    headers={"X-API-Key": "test-key"},
                    timeout=30
                )
                
                duration = time.time() - start_time
                
                return {
                    'user_id': user_id,
                    'status_code': response.status_code,
                    'duration': duration,
                    'success': response.status_code == 200
                }
                
            except Exception as e:
                duration = time.time() - start_time
                return {
                    'user_id': user_id,
                    'status_code': 0,
                    'duration': duration,
                    'success': False,
                    'error': str(e)
                }
        
        # 使用线程池执行并发请求
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_users)]
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                # 更新指标
                if result['success']:
                    self.request_counter.labels(endpoint='/verify/screenshot', status='success').inc()
                else:
                    self.request_counter.labels(endpoint='/verify/screenshot', status='error').inc()
                
                self.request_duration.labels(endpoint='/verify/screenshot').observe(result['duration'])
        
        # 分析结果
        self._analyze_concurrent_results(results)
    
    def _load_test(self, duration: int):
        """负载测试"""
        self.logger.info(f"执行负载测试，持续时间: {duration}s")
        
        test_image = self._create_test_image()
        start_time = time.time()
        request_count = 0
        success_count = 0
        total_duration = 0
        
        while time.time() - start_time < duration:
            request_start = time.time()
            
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/verify/screenshot",
                    json={
                        "image": test_image,
                        "source": "android",
                        "app_type": "payment",
                        "context": "load_test"
                    },
                    headers={"X-API-Key": "test-key"},
                    timeout=30
                )
                
                request_duration = time.time() - request_start
                total_duration += request_duration
                request_count += 1
                
                if response.status_code == 200:
                    success_count += 1
                    self.request_counter.labels(endpoint='/verify/screenshot', status='success').inc()
                else:
                    self.request_counter.labels(endpoint='/verify/screenshot', status='error').inc()
                
                self.request_duration.labels(endpoint='/verify/screenshot').observe(request_duration)
                
                # 控制请求频率
                time.sleep(0.1)
                
            except Exception as e:
                request_duration = time.time() - request_start
                total_duration += request_duration
                request_count += 1
                self.request_counter.labels(endpoint='/verify/screenshot', status='error').inc()
                self.request_duration.labels(endpoint='/verify/screenshot').observe(request_duration)
        
        # 分析负载测试结果
        self._analyze_load_test_results(request_count, success_count, total_duration, duration)
    
    def _stress_test(self, max_users: int):
        """压力测试"""
        self.logger.info(f"执行压力测试，最大用户数: {max_users}")
        
        test_image = self._create_test_image()
        results = []
        
        def stress_request(user_id: int):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/verify/screenshot",
                    json={
                        "image": test_image,
                        "source": "android",
                        "app_type": "payment",
                        "context": f"stress_test_user_{user_id}"
                    },
                    headers={"X-API-Key": "test-key"},
                    timeout=30
                )
                
                duration = time.time() - start_time
                
                return {
                    'user_id': user_id,
                    'status_code': response.status_code,
                    'duration': duration,
                    'success': response.status_code == 200
                }
                
            except Exception as e:
                duration = time.time() - start_time
                return {
                    'user_id': user_id,
                    'status_code': 0,
                    'duration': duration,
                    'success': False,
                    'error': str(e)
                }
        
        # 逐步增加并发用户数
        for user_count in [10, 20, 50, 100, max_users]:
            self.logger.info(f"测试并发用户数: {user_count}")
            
            with ThreadPoolExecutor(max_workers=user_count) as executor:
                futures = [executor.submit(stress_request, i) for i in range(user_count)]
                
                batch_results = []
                for future in as_completed(futures):
                    result = future.result()
                    batch_results.append(result)
                    
                    # 更新指标
                    if result['success']:
                        self.request_counter.labels(endpoint='/verify/screenshot', status='success').inc()
                    else:
                        self.request_counter.labels(endpoint='/verify/screenshot', status='error').inc()
                    
                    self.request_duration.labels(endpoint='/verify/screenshot').observe(result['duration'])
                
                # 分析批次结果
                success_rate = sum(1 for r in batch_results if r['success']) / len(batch_results)
                avg_duration = np.mean([r['duration'] for r in batch_results])
                
                self.logger.info(f"用户数 {user_count}: 成功率 {success_rate:.2%}, 平均耗时 {avg_duration:.3f}s")
                
                # 如果成功率低于阈值，停止测试
                if success_rate < 0.8:
                    self.logger.warning(f"成功率过低 ({success_rate:.2%})，停止压力测试")
                    break
        
        self._analyze_stress_test_results(results)
    
    def _create_test_image(self) -> str:
        """创建测试图像"""
        # 创建一个简单的测试图像
        img = Image.new('RGB', (400, 800), color='white')
        
        # 转换为base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    def _analyze_concurrent_results(self, results: List[Dict]):
        """分析并发测试结果"""
        success_count = sum(1 for r in results if r['success'])
        success_rate = success_count / len(results)
        avg_duration = np.mean([r['duration'] for r in results])
        max_duration = max([r['duration'] for r in results])
        min_duration = min([r['duration'] for r in results])
        
        self.logger.info(f"并发测试结果:")
        self.logger.info(f"  总请求数: {len(results)}")
        self.logger.info(f"  成功数: {success_count}")
        self.logger.info(f"  成功率: {success_rate:.2%}")
        self.logger.info(f"  平均耗时: {avg_duration:.3f}s")
        self.logger.info(f"  最大耗时: {max_duration:.3f}s")
        self.logger.info(f"  最小耗时: {min_duration:.3f}s")
    
    def _analyze_load_test_results(self, request_count: int, success_count: int, total_duration: float, test_duration: int):
        """分析负载测试结果"""
        success_rate = success_count / request_count if request_count > 0 else 0
        avg_duration = total_duration / request_count if request_count > 0 else 0
        rps = request_count / test_duration
        
        self.logger.info(f"负载测试结果:")
        self.logger.info(f"  总请求数: {request_count}")
        self.logger.info(f"  成功数: {success_count}")
        self.logger.info(f"  成功率: {success_rate:.2%}")
        self.logger.info(f"  平均耗时: {avg_duration:.3f}s")
        self.logger.info(f"  请求速率: {rps:.2f} RPS")
    
    def _analyze_stress_test_results(self, results: List[Dict]):
        """分析压力测试结果"""
        if not results:
            return
        
        success_count = sum(1 for r in results if r['success'])
        success_rate = success_count / len(results)
        avg_duration = np.mean([r['duration'] for r in results])
        
        self.logger.info(f"压力测试结果:")
        self.logger.info(f"  总请求数: {len(results)}")
        self.logger.info(f"  成功数: {success_count}")
        self.logger.info(f"  成功率: {success_rate:.2%}")
        self.logger.info(f"  平均耗时: {avg_duration:.3f}s")
    
    def _generate_test_report(self):
        """生成测试报告"""
        self.logger.info("生成性能测试报告")
        
        # 这里应该生成详细的测试报告
        # 包括图表、统计数据等
        pass


class MonitoringSetup:
    """监控设置"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        
        with open(self.config_path, 'r') as f:
            self.monitor_config = json.load(f)
    
    def setup_monitoring(self):
        """设置监控"""
        self.logger.info("设置监控系统")
        
        # 1. 设置Prometheus
        self._setup_prometheus()
        
        # 2. 设置Grafana
        self._setup_grafana()
        
        # 3. 设置告警
        self._setup_alerts()
        
        # 4. 设置日志聚合
        self._setup_logging()
        
        self.logger.info("监控系统设置完成")
    
    def _setup_prometheus(self):
        """设置Prometheus"""
        self.logger.info("设置Prometheus")
        
        # 创建Prometheus配置
        prometheus_config = {
            'global': {
                'scrape_interval': '15s',
                'evaluation_interval': '15s'
            },
            'scrape_configs': [
                {
                    'job_name': 'screenshot-verification-api',
                    'static_configs': [
                        {
                            'targets': [self.monitor_config.get('api_endpoint', 'localhost:8000')]
                        }
                    ]
                }
            ]
        }
        
        # 保存配置
        with open('prometheus.yml', 'w') as f:
            import yaml
            yaml.dump(prometheus_config, f)
        
        self.logger.info("Prometheus配置已生成")
    
    def _setup_grafana(self):
        """设置Grafana"""
        self.logger.info("设置Grafana")
        
        # 创建Grafana仪表板配置
        dashboard_config = {
            'dashboard': {
                'title': 'Screenshot Verification API',
                'panels': [
                    {
                        'title': 'Request Rate',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': 'rate(api_requests_total[5m])',
                                'legendFormat': '{{endpoint}}'
                            }
                        ]
                    },
                    {
                        'title': 'Response Time',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': 'histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))',
                                'legendFormat': '95th percentile'
                            }
                        ]
                    },
                    {
                        'title': 'Error Rate',
                        'type': 'graph',
                        'targets': [
                            {
                                'expr': 'rate(api_requests_total{status="error"}[5m])',
                                'legendFormat': 'Error rate'
                            }
                        ]
                    }
                ]
            }
        }
        
        # 保存配置
        with open('grafana_dashboard.json', 'w') as f:
            json.dump(dashboard_config, f, indent=2)
        
        self.logger.info("Grafana仪表板配置已生成")
    
    def _setup_alerts(self):
        """设置告警"""
        self.logger.info("设置告警规则")
        
        # 创建告警规则
        alert_rules = [
            {
                'name': 'HighErrorRate',
                'expr': 'rate(api_requests_total{status="error"}[5m]) > 0.1',
                'for': '5m',
                'labels': {
                    'severity': 'warning'
                },
                'annotations': {
                    'summary': 'High error rate detected',
                    'description': 'Error rate is above 10% for 5 minutes'
                }
            },
            {
                'name': 'HighResponseTime',
                'expr': 'histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m])) > 1',
                'for': '5m',
                'labels': {
                    'severity': 'warning'
                },
                'annotations': {
                    'summary': 'High response time detected',
                    'description': '95th percentile response time is above 1 second'
                }
            }
        ]
        
        # 保存告警规则
        with open('alert_rules.yml', 'w') as f:
            import yaml
            yaml.dump(alert_rules, f)
        
        self.logger.info("告警规则已生成")
    
    def _setup_logging(self):
        """设置日志聚合"""
        self.logger.info("设置日志聚合")
        
        # 这里应该设置ELK Stack或其他日志聚合系统
        pass


def main():
    parser = argparse.ArgumentParser(description='部署测试与生产环境工具')
    parser.add_argument('--mode', type=str, 
                       choices=['deploy', 'test', 'monitor'], 
                       required=True, help='运行模式')
    parser.add_argument('--config', type=str, required=True, 
                       help='配置文件路径')
    parser.add_argument('--base_url', type=str, default='http://localhost:8000', 
                       help='API基础URL（用于测试）')
    parser.add_argument('--environment', type=str, default='production', 
                       help='部署环境')
    
    args = parser.parse_args()
    
    if args.mode == 'deploy':
        deployer = ProductionDeployer(args.config)
        deployer.deploy(args.environment)
    
    elif args.mode == 'test':
        tester = PerformanceTester(args.base_url)
        
        # 加载测试配置
        with open(args.config, 'r') as f:
            test_config = json.load(f)
        
        tester.run_performance_test(test_config)
    
    elif args.mode == 'monitor':
        monitor = MonitoringSetup(args.config)
        monitor.setup_monitoring()


if __name__ == '__main__':
    main()