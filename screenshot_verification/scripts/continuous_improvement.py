#!/usr/bin/env python3
"""
持续改进与反馈系统
包括用户反馈收集、模型增量学习、A/B测试和性能监控
"""
import os
import sys
import argparse
import logging
import json
import time
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import schedule
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import redis
import pickle

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_models import ModelTrainer, ScreenshotDataset, get_transforms
from core.model_enhancements import create_model_ensemble
from config.settings import settings


class FeedbackCollector:
    """用户反馈收集器"""
    
    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # 初始化数据库
        self._init_database()
        
        # 初始化Redis连接
        self.redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=1,  # 使用不同的数据库
            decode_responses=True
        )
    
    def _init_database(self):
        """初始化反馈数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建反馈表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                user_id TEXT,
                image_hash TEXT,
                prediction TEXT,
                user_feedback TEXT,
                confidence REAL,
                processing_time REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # 创建错误日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT,
                error_type TEXT,
                error_message TEXT,
                stack_trace TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建性能指标表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                endpoint TEXT,
                response_time REAL,
                success_rate REAL,
                error_count INTEGER,
                request_count INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def collect_feedback(self, feedback_data: Dict[str, Any]):
        """收集用户反馈"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO feedback (
                    request_id, user_id, image_hash, prediction, 
                    user_feedback, confidence, processing_time, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback_data.get('request_id'),
                feedback_data.get('user_id'),
                feedback_data.get('image_hash'),
                feedback_data.get('prediction'),
                feedback_data.get('user_feedback'),
                feedback_data.get('confidence'),
                feedback_data.get('processing_time'),
                json.dumps(feedback_data.get('metadata', {}))
            ))
            
            conn.commit()
            conn.close()
            
            # 缓存到Redis
            self._cache_feedback(feedback_data)
            
            self.logger.info(f"反馈已收集: {feedback_data.get('request_id')}")
            
        except Exception as e:
            self.logger.error(f"收集反馈失败: {e}")
    
    def _cache_feedback(self, feedback_data: Dict[str, Any]):
        """缓存反馈到Redis"""
        try:
            key = f"feedback:{feedback_data.get('request_id')}"
            self.redis_client.setex(key, 3600, json.dumps(feedback_data))  # 1小时过期
        except Exception as e:
            self.logger.error(f"缓存反馈失败: {e}")
    
    def log_error(self, error_data: Dict[str, Any]):
        """记录错误日志"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO error_logs (
                    request_id, error_type, error_message, stack_trace
                ) VALUES (?, ?, ?, ?)
            ''', (
                error_data.get('request_id'),
                error_data.get('error_type'),
                error_data.get('error_message'),
                error_data.get('stack_trace')
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.error(f"错误已记录: {error_data.get('error_type')}")
            
        except Exception as e:
            self.logger.error(f"记录错误失败: {e}")
    
    def log_performance(self, performance_data: Dict[str, Any]):
        """记录性能指标"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics (
                    endpoint, response_time, success_rate, error_count, request_count
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                performance_data.get('endpoint'),
                performance_data.get('response_time'),
                performance_data.get('success_rate'),
                performance_data.get('error_count'),
                performance_data.get('request_count')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"记录性能指标失败: {e}")
    
    def get_feedback_summary(self, days: int = 7) -> Dict[str, Any]:
        """获取反馈摘要"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 获取时间范围内的反馈
            since_date = datetime.now() - timedelta(days=days)
            
            # 反馈统计
            feedback_df = pd.read_sql_query('''
                SELECT * FROM feedback 
                WHERE timestamp >= ?
            ''', conn, params=(since_date,))
            
            # 错误统计
            error_df = pd.read_sql_query('''
                SELECT * FROM error_logs 
                WHERE timestamp >= ?
            ''', conn, params=(since_date,))
            
            # 性能统计
            perf_df = pd.read_sql_query('''
                SELECT * FROM performance_metrics 
                WHERE timestamp >= ?
            ''', conn, params=(since_date,))
            
            conn.close()
            
            # 计算统计信息
            summary = {
                'total_feedback': len(feedback_df),
                'positive_feedback': len(feedback_df[feedback_df['user_feedback'] == 'correct']),
                'negative_feedback': len(feedback_df[feedback_df['user_feedback'] == 'incorrect']),
                'total_errors': len(error_df),
                'error_types': error_df['error_type'].value_counts().to_dict(),
                'avg_response_time': perf_df['response_time'].mean() if len(perf_df) > 0 else 0,
                'avg_success_rate': perf_df['success_rate'].mean() if len(perf_df) > 0 else 0,
                'feedback_trend': self._calculate_feedback_trend(feedback_df),
                'performance_trend': self._calculate_performance_trend(perf_df)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"获取反馈摘要失败: {e}")
            return {}


class IncrementalLearner:
    """增量学习器"""
    
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # 加载当前模型
        self.model = self._load_model()
        
        # 增量学习配置
        self.learning_config = {
            'learning_rate': 1e-5,  # 较小的学习率
            'batch_size': 16,
            'epochs_per_update': 5,
            'min_samples_for_update': 100,
            'max_samples_per_update': 1000
        }
    
    def _load_model(self) -> nn.Module:
        """加载当前模型"""
        checkpoint = torch.load(self.model_path, map_location='cpu')
        model_config = checkpoint['config']
        
        # 根据配置创建模型
        if model_config['model_type'] == 'ensemble':
            model = create_model_ensemble(model_config['ensemble_configs'])
        else:
            # 其他模型类型的加载逻辑
            pass
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def update_model(self, new_data: List[Tuple[str, int]]):
        """使用新数据更新模型"""
        if len(new_data) < self.learning_config['min_samples_for_update']:
            self.logger.info(f"新数据不足，需要至少 {self.learning_config['min_samples_for_update']} 个样本")
            return False
        
        self.logger.info(f"开始增量学习，新数据: {len(new_data)} 个样本")
        
        try:
            # 准备新数据
            new_dataset = self._prepare_incremental_data(new_data)
            new_loader = DataLoader(
                new_dataset,
                batch_size=self.learning_config['batch_size'],
                shuffle=True
            )
            
            # 增量学习
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_config['learning_rate']
            )
            
            criterion = nn.CrossEntropyLoss()
            
            # 训练几个epoch
            for epoch in range(self.learning_config['epochs_per_update']):
                self.model.train()
                total_loss = 0
                
                for batch_idx, (data, target) in enumerate(new_loader):
                    optimizer.zero_grad()
                    
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(new_loader)
                self.logger.info(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            
            # 保存更新后的模型
            self._save_updated_model()
            
            self.logger.info("增量学习完成")
            return True
            
        except Exception as e:
            self.logger.error(f"增量学习失败: {e}")
            return False
    
    def _prepare_incremental_data(self, new_data: List[Tuple[str, int]]) -> Dataset:
        """准备增量学习数据"""
        # 这里应该实现数据预处理逻辑
        # 包括图像加载、变换等
        pass
    
    def _save_updated_model(self):
        """保存更新后的模型"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'updated_at': datetime.now().isoformat(),
            'incremental_update': True
        }
        
        updated_model_path = self.output_dir / f"incremental_model_{int(time.time())}.pth"
        torch.save(checkpoint, updated_model_path)
        
        self.logger.info(f"更新后的模型已保存: {updated_model_path}")


class ABTestManager:
    """A/B测试管理器"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        
        with open(self.config_path, 'r') as f:
            self.ab_config = json.load(f)
        
        # 初始化Redis连接
        self.redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=2,  # A/B测试专用数据库
            decode_responses=True
        )
    
    def assign_variant(self, user_id: str) -> str:
        """为用户分配测试变体"""
        try:
            # 检查用户是否已有分配
            existing_variant = self.redis_client.get(f"ab_test:{user_id}")
            if existing_variant:
                return existing_variant
            
            # 根据配置分配变体
            variants = self.ab_config['variants']
            weights = [v['weight'] for v in variants]
            
            # 加权随机选择
            variant_name = np.random.choice(
                [v['name'] for v in variants],
                p=np.array(weights) / sum(weights)
            )
            
            # 保存分配结果
            self.redis_client.setex(f"ab_test:{user_id}", 86400, variant_name)  # 24小时过期
            
            self.logger.info(f"用户 {user_id} 分配到变体: {variant_name}")
            return variant_name
            
        except Exception as e:
            self.logger.error(f"分配变体失败: {e}")
            return self.ab_config['default_variant']
    
    def record_experiment(self, user_id: str, variant: str, result: Dict[str, Any]):
        """记录A/B测试结果"""
        try:
            experiment_key = f"ab_experiment:{user_id}:{variant}"
            
            # 记录结果
            self.redis_client.hset(experiment_key, mapping={
                'user_id': user_id,
                'variant': variant,
                'timestamp': datetime.now().isoformat(),
                'result': json.dumps(result)
            })
            
            # 设置过期时间
            self.redis_client.expire(experiment_key, 86400 * 7)  # 7天过期
            
        except Exception as e:
            self.logger.error(f"记录A/B测试结果失败: {e}")
    
    def analyze_results(self) -> Dict[str, Any]:
        """分析A/B测试结果"""
        try:
            results = {}
            
            for variant in self.ab_config['variants']:
                variant_name = variant['name']
                
                # 获取该变体的所有结果
                pattern = f"ab_experiment:*:{variant_name}"
                keys = self.redis_client.keys(pattern)
                
                variant_results = []
                for key in keys:
                    data = self.redis_client.hgetall(key)
                    if data:
                        result = json.loads(data.get('result', '{}'))
                        variant_results.append(result)
                
                if variant_results:
                    # 计算统计信息
                    df = pd.DataFrame(variant_results)
                    
                    results[variant_name] = {
                        'sample_size': len(variant_results),
                        'avg_accuracy': df.get('accuracy', [0]).mean(),
                        'avg_response_time': df.get('response_time', [0]).mean(),
                        'success_rate': df.get('success', [False]).mean(),
                        'user_satisfaction': df.get('satisfaction', [0]).mean()
                    }
            
            # 计算统计显著性
            if len(results) >= 2:
                results['statistical_significance'] = self._calculate_significance(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"分析A/B测试结果失败: {e}")
            return {}
    
    def _calculate_significance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算统计显著性"""
        # 这里应该实现统计显著性检验
        # 例如t检验、卡方检验等
        return {'significant': False, 'p_value': 1.0}


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        
        with open(self.config_path, 'r') as f:
            self.monitor_config = json.load(f)
        
        # 初始化指标存储
        self.metrics = {
            'response_times': [],
            'error_rates': [],
            'throughput': [],
            'resource_usage': []
        }
    
    def start_monitoring(self):
        """开始性能监控"""
        self.logger.info("开始性能监控")
        
        # 设置定时任务
        schedule.every(1).minutes.do(self._collect_metrics)
        schedule.every(5).minutes.do(self._analyze_performance)
        schedule.every(1).hours.do(self._generate_report)
        
        # 运行监控循环
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def _collect_metrics(self):
        """收集性能指标"""
        try:
            # 收集系统资源使用情况
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 收集应用指标
            app_metrics = self._collect_app_metrics()
            
            # 存储指标
            self.metrics['resource_usage'].append({
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                **app_metrics
            })
            
            # 检查告警条件
            self._check_alerts(cpu_percent, memory.percent, disk.percent)
            
        except Exception as e:
            self.logger.error(f"收集指标失败: {e}")
    
    def _collect_app_metrics(self) -> Dict[str, Any]:
        """收集应用指标"""
        # 这里应该从应用收集指标
        # 例如从Prometheus、日志等
        return {
            'response_time': 0.0,
            'error_rate': 0.0,
            'throughput': 0.0
        }
    
    def _check_alerts(self, cpu_percent: float, memory_percent: float, disk_percent: float):
        """检查告警条件"""
        alerts = []
        
        if cpu_percent > self.monitor_config.get('cpu_threshold', 80):
            alerts.append(f"CPU使用率过高: {cpu_percent}%")
        
        if memory_percent > self.monitor_config.get('memory_threshold', 80):
            alerts.append(f"内存使用率过高: {memory_percent}%")
        
        if disk_percent > self.monitor_config.get('disk_threshold', 80):
            alerts.append(f"磁盘使用率过高: {disk_percent}%")
        
        if alerts:
            self._send_alerts(alerts)
    
    def _send_alerts(self, alerts: List[str]):
        """发送告警"""
        for alert in alerts:
            self.logger.warning(f"告警: {alert}")
            # 这里应该实现告警发送逻辑
            # 例如发送邮件、Slack消息等
    
    def _analyze_performance(self):
        """分析性能趋势"""
        try:
            if not self.metrics['resource_usage']:
                return
            
            # 转换为DataFrame
            df = pd.DataFrame(self.metrics['resource_usage'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 分析趋势
            recent_data = df.tail(30)  # 最近30个数据点
            
            trends = {
                'cpu_trend': self._calculate_trend(recent_data['cpu_percent']),
                'memory_trend': self._calculate_trend(recent_data['memory_percent']),
                'disk_trend': self._calculate_trend(recent_data['disk_percent'])
            }
            
            # 检查异常
            self._detect_anomalies(recent_data)
            
        except Exception as e:
            self.logger.error(f"分析性能失败: {e}")
    
    def _calculate_trend(self, data: pd.Series) -> str:
        """计算趋势"""
        if len(data) < 2:
            return 'stable'
        
        slope = np.polyfit(range(len(data)), data, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _detect_anomalies(self, data: pd.DataFrame):
        """检测异常"""
        # 这里应该实现异常检测算法
        # 例如基于统计方法、机器学习等
        pass
    
    def _generate_report(self):
        """生成性能报告"""
        try:
            if not self.metrics['resource_usage']:
                return
            
            df = pd.DataFrame(self.metrics['resource_usage'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 生成报告
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'avg_cpu': df['cpu_percent'].mean(),
                    'avg_memory': df['memory_percent'].mean(),
                    'avg_disk': df['disk_percent'].mean(),
                    'max_cpu': df['cpu_percent'].max(),
                    'max_memory': df['memory_percent'].max(),
                    'max_disk': df['disk_percent'].max()
                },
                'trends': {
                    'cpu_trend': self._calculate_trend(df['cpu_percent']),
                    'memory_trend': self._calculate_trend(df['memory_percent']),
                    'disk_trend': self._calculate_trend(df['disk_percent'])
                }
            }
            
            # 保存报告
            report_path = Path(self.monitor_config.get('report_dir', './reports'))
            report_path.mkdir(parents=True, exist_ok=True)
            
            with open(report_path / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info("性能报告已生成")
            
        except Exception as e:
            self.logger.error(f"生成性能报告失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='持续改进与反馈系统')
    parser.add_argument('--mode', type=str, 
                       choices=['collect', 'learn', 'abtest', 'monitor'], 
                       required=True, help='运行模式')
    parser.add_argument('--config', type=str, required=True, 
                       help='配置文件路径')
    parser.add_argument('--model_path', type=str, 
                       help='模型路径（用于增量学习）')
    parser.add_argument('--output_dir', type=str, default='./improvements', 
                       help='输出目录')
    
    args = parser.parse_args()
    
    if args.mode == 'collect':
        # 启动反馈收集
        collector = FeedbackCollector()
        
        # 这里应该启动一个服务来接收反馈
        # 或者定期从数据库读取反馈
        print("反馈收集器已启动")
    
    elif args.mode == 'learn':
        if not args.model_path:
            parser.error("learn模式需要指定model_path")
        
        learner = IncrementalLearner(args.model_path, args.output_dir)
        
        # 这里应该从反馈数据库读取新数据
        # 然后进行增量学习
        print("增量学习器已启动")
    
    elif args.mode == 'abtest':
        manager = ABTestManager(args.config)
        
        # 分析A/B测试结果
        results = manager.analyze_results()
        print("A/B测试结果:", json.dumps(results, indent=2))
    
    elif args.mode == 'monitor':
        monitor = PerformanceMonitor(args.config)
        monitor.start_monitoring()


if __name__ == '__main__':
    main()