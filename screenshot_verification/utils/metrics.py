"""
指标收集器
用于收集和统计服务性能指标
"""
import time
import logging
from typing import Dict, Any, List
from collections import defaultdict, deque
from threading import Lock
import statistics

from config.settings import settings


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 计数器
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 分类统计
        self.authentic_count = 0
        self.fake_count = 0
        
        # 性能指标
        self.processing_times = deque(maxlen=1000)  # 保留最近1000次请求的处理时间
        self.detector_times = defaultdict(lambda: deque(maxlen=1000))
        
        # 检测器性能
        self.detector_performance = defaultdict(lambda: {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'average_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0
        })
        
        # 错误统计
        self.error_counts = defaultdict(int)
        
        # 线程安全
        self.lock = Lock()
        
        # 启动时间
        self.start_time = time.time()
        
        self.logger.info("指标收集器初始化完成")
    
    def record_verification(self, is_authentic: bool, confidence: float, processing_time: float):
        """记录验证指标"""
        with self.lock:
            self.total_requests += 1
            self.successful_requests += 1
            self.processing_times.append(processing_time)
            
            if is_authentic:
                self.authentic_count += 1
            else:
                self.fake_count += 1
    
    def record_error(self):
        """记录错误"""
        with self.lock:
            self.total_requests += 1
            self.failed_requests += 1
    
    def record_cache_hit(self):
        """记录缓存命中"""
        with self.lock:
            self.cache_hits += 1
    
    def record_cache_miss(self):
        """记录缓存未命中"""
        with self.lock:
            self.cache_misses += 1
    
    def record_detector_performance(self, detector_name: str, success: bool, processing_time: float):
        """记录检测器性能"""
        with self.lock:
            perf = self.detector_performance[detector_name]
            perf['total_calls'] += 1
            
            if success:
                perf['successful_calls'] += 1
            else:
                perf['failed_calls'] += 1
            
            # 更新时间统计
            perf['average_time'] = (
                (perf['average_time'] * (perf['total_calls'] - 1) + processing_time) / 
                perf['total_calls']
            )
            perf['min_time'] = min(perf['min_time'], processing_time)
            perf['max_time'] = max(perf['max_time'], processing_time)
            
            # 记录到时间队列
            self.detector_times[detector_name].append(processing_time)
    
    def record_error_type(self, error_type: str):
        """记录错误类型"""
        with self.lock:
            self.error_counts[error_type] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            stats = {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'authentic_count': self.authentic_count,
                'fake_count': self.fake_count,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'uptime': time.time() - self.start_time
            }
            
            # 计算平均处理时间
            if self.processing_times:
                stats['average_processing_time'] = statistics.mean(self.processing_times)
                stats['min_processing_time'] = min(self.processing_times)
                stats['max_processing_time'] = max(self.processing_times)
                stats['processing_time_std'] = statistics.stdev(self.processing_times) if len(self.processing_times) > 1 else 0
            else:
                stats['average_processing_time'] = 0.0
                stats['min_processing_time'] = 0.0
                stats['max_processing_time'] = 0.0
                stats['processing_time_std'] = 0.0
            
            # 计算成功率
            if self.total_requests > 0:
                stats['success_rate'] = self.successful_requests / self.total_requests
                stats['error_rate'] = self.failed_requests / self.total_requests
            else:
                stats['success_rate'] = 0.0
                stats['error_rate'] = 0.0
            
            # 计算缓存命中率
            total_cache_requests = self.cache_hits + self.cache_misses
            if total_cache_requests > 0:
                stats['cache_hit_rate'] = self.cache_hits / total_cache_requests
            else:
                stats['cache_hit_rate'] = 0.0
            
            # 检测器性能
            detector_perf = {}
            for name, perf in self.detector_performance.items():
                if perf['total_calls'] > 0:
                    detector_perf[name] = {
                        'total_calls': perf['total_calls'],
                        'successful_calls': perf['successful_calls'],
                        'failed_calls': perf['failed_calls'],
                        'success_rate': perf['successful_calls'] / perf['total_calls'],
                        'average_time': perf['average_time'],
                        'min_time': perf['min_time'] if perf['min_time'] != float('inf') else 0.0,
                        'max_time': perf['max_time']
                    }
                else:
                    detector_perf[name] = {
                        'total_calls': 0,
                        'successful_calls': 0,
                        'failed_calls': 0,
                        'success_rate': 0.0,
                        'average_time': 0.0,
                        'min_time': 0.0,
                        'max_time': 0.0
                    }
            
            stats['detector_performance'] = detector_perf
            
            # 错误统计
            stats['error_counts'] = dict(self.error_counts)
            
            # 最近性能趋势（最近100次请求）
            recent_times = list(self.processing_times)[-100:]
            if recent_times:
                stats['recent_average_time'] = statistics.mean(recent_times)
                stats['recent_min_time'] = min(recent_times)
                stats['recent_max_time'] = max(recent_times)
            else:
                stats['recent_average_time'] = 0.0
                stats['recent_min_time'] = 0.0
                stats['recent_max_time'] = 0.0
            
            return stats
    
    def get_detector_statistics(self, detector_name: str) -> Dict[str, Any]:
        """获取特定检测器的统计信息"""
        with self.lock:
            if detector_name not in self.detector_performance:
                return {}
            
            perf = self.detector_performance[detector_name]
            times = list(self.detector_times[detector_name])
            
            stats = {
                'total_calls': perf['total_calls'],
                'successful_calls': perf['successful_calls'],
                'failed_calls': perf['failed_calls'],
                'average_time': perf['average_time'],
                'min_time': perf['min_time'] if perf['min_time'] != float('inf') else 0.0,
                'max_time': perf['max_time']
            }
            
            if perf['total_calls'] > 0:
                stats['success_rate'] = perf['successful_calls'] / perf['total_calls']
                stats['failure_rate'] = perf['failed_calls'] / perf['total_calls']
            else:
                stats['success_rate'] = 0.0
                stats['failure_rate'] = 0.0
            
            if times:
                stats['time_std'] = statistics.stdev(times) if len(times) > 1 else 0
                stats['time_median'] = statistics.median(times)
                stats['time_percentiles'] = {
                    'p50': statistics.quantiles(times, n=2)[0] if len(times) > 1 else times[0],
                    'p90': statistics.quantiles(times, n=10)[8] if len(times) > 9 else max(times),
                    'p95': statistics.quantiles(times, n=20)[18] if len(times) > 19 else max(times),
                    'p99': statistics.quantiles(times, n=100)[98] if len(times) > 99 else max(times)
                }
            else:
                stats['time_std'] = 0.0
                stats['time_median'] = 0.0
                stats['time_percentiles'] = {'p50': 0.0, 'p90': 0.0, 'p95': 0.0, 'p99': 0.0}
            
            return stats
    
    def get_performance_trends(self, window_size: int = 100) -> Dict[str, Any]:
        """获取性能趋势"""
        with self.lock:
            if len(self.processing_times) < window_size:
                return {}
            
            recent_times = list(self.processing_times)[-window_size:]
            
            # 计算趋势
            if len(recent_times) > 1:
                # 简单线性回归
                x = list(range(len(recent_times)))
                y = recent_times
                
                n = len(x)
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(x[i] * y[i] for i in range(n))
                sum_x2 = sum(x[i] ** 2 for i in range(n))
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
                intercept = (sum_y - slope * sum_x) / n
                
                trend = {
                    'slope': slope,
                    'intercept': intercept,
                    'trend_direction': 'increasing' if slope > 0.001 else 'decreasing' if slope < -0.001 else 'stable',
                    'prediction_next': intercept + slope * n
                }
            else:
                trend = {
                    'slope': 0.0,
                    'intercept': recent_times[0] if recent_times else 0.0,
                    'trend_direction': 'stable',
                    'prediction_next': recent_times[0] if recent_times else 0.0
                }
            
            return {
                'window_size': window_size,
                'average_time': statistics.mean(recent_times),
                'min_time': min(recent_times),
                'max_time': max(recent_times),
                'std_time': statistics.stdev(recent_times) if len(recent_times) > 1 else 0,
                'trend': trend
            }
    
    def reset_statistics(self):
        """重置统计信息"""
        with self.lock:
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.authentic_count = 0
            self.fake_count = 0
            self.processing_times.clear()
            self.detector_times.clear()
            self.detector_performance.clear()
            self.error_counts.clear()
            self.start_time = time.time()
            
            self.logger.info("统计信息已重置")
    
    def export_metrics(self) -> Dict[str, Any]:
        """导出指标数据"""
        stats = self.get_statistics()
        
        # 添加时间戳
        stats['timestamp'] = time.time()
        stats['export_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        stats = self.get_statistics()
        
        # 检查关键指标
        health_status = {
            'status': 'healthy',
            'checks': {}
        }
        
        # 检查成功率
        if stats['total_requests'] > 0:
            success_rate = stats['success_rate']
            if success_rate < 0.9:
                health_status['status'] = 'warning'
                health_status['checks']['success_rate'] = f"成功率较低: {success_rate:.2%}"
            else:
                health_status['checks']['success_rate'] = f"成功率正常: {success_rate:.2%}"
        
        # 检查平均处理时间
        avg_time = stats['average_processing_time']
        if avg_time > 2.0:  # 超过2秒
            health_status['status'] = 'warning'
            health_status['checks']['processing_time'] = f"处理时间较长: {avg_time:.3f}s"
        else:
            health_status['checks']['processing_time'] = f"处理时间正常: {avg_time:.3f}s"
        
        # 检查错误率
        if stats['total_requests'] > 0:
            error_rate = stats['error_rate']
            if error_rate > 0.1:  # 错误率超过10%
                health_status['status'] = 'critical'
                health_status['checks']['error_rate'] = f"错误率过高: {error_rate:.2%}"
            else:
                health_status['checks']['error_rate'] = f"错误率正常: {error_rate:.2%}"
        
        return health_status