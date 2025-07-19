"""
缓存管理器
用于缓存验证结果，提高响应速度
"""
import time
import logging
from typing import Any, Optional, Dict
import json
import hashlib
from threading import Lock

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis未安装，将使用内存缓存")

from config.settings import settings


class CacheManager:
    """缓存管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        self.memory_cache = {}
        self.cache_lock = Lock()
        
        # 初始化缓存
        self._initialize_cache()
        
        self.logger.info("缓存管理器初始化完成")
    
    def _initialize_cache(self):
        """初始化缓存"""
        if settings.cache_enabled:
            if REDIS_AVAILABLE and settings.redis_url != "redis://localhost:6379":
                try:
                    self.redis_client = redis.from_url(settings.redis_url)
                    # 测试连接
                    self.redis_client.ping()
                    self.logger.info("Redis缓存初始化成功")
                except Exception as e:
                    self.logger.warning(f"Redis连接失败，使用内存缓存: {e}")
                    self.redis_client = None
            else:
                self.logger.info("使用内存缓存")
        else:
            self.logger.info("缓存已禁用")
    
    def _generate_key(self, data: Any) -> str:
        """生成缓存键"""
        if isinstance(data, str):
            key_data = data
        else:
            key_data = json.dumps(data, sort_keys=True)
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if not settings.cache_enabled:
            return None
        
        try:
            if self.redis_client:
                # 从Redis获取
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            else:
                # 从内存获取
                with self.cache_lock:
                    if key in self.memory_cache:
                        item = self.memory_cache[key]
                        if time.time() < item['expires_at']:
                            return item['value']
                        else:
                            # 过期，删除
                            del self.memory_cache[key]
            
        except Exception as e:
            self.logger.error(f"获取缓存失败: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """设置缓存值"""
        if not settings.cache_enabled:
            return False
        
        if ttl is None:
            ttl = settings.cache_ttl
        
        try:
            if self.redis_client:
                # 存储到Redis
                return self.redis_client.setex(
                    key, 
                    ttl, 
                    json.dumps(value, ensure_ascii=False)
                )
            else:
                # 存储到内存
                with self.cache_lock:
                    self.memory_cache[key] = {
                        'value': value,
                        'expires_at': time.time() + ttl
                    }
                    
                    # 清理过期缓存
                    self._cleanup_expired()
                
                return True
                
        except Exception as e:
            self.logger.error(f"设置缓存失败: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        if not settings.cache_enabled:
            return False
        
        try:
            if self.redis_client:
                # 从Redis删除
                return bool(self.redis_client.delete(key))
            else:
                # 从内存删除
                with self.cache_lock:
                    if key in self.memory_cache:
                        del self.memory_cache[key]
                        return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"删除缓存失败: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        if not settings.cache_enabled:
            return False
        
        try:
            if self.redis_client:
                # 检查Redis
                return bool(self.redis_client.exists(key))
            else:
                # 检查内存
                with self.cache_lock:
                    if key in self.memory_cache:
                        item = self.memory_cache[key]
                        if time.time() < item['expires_at']:
                            return True
                        else:
                            # 过期，删除
                            del self.memory_cache[key]
                
                return False
                
        except Exception as e:
            self.logger.error(f"检查缓存失败: {e}")
            return False
    
    def clear(self) -> bool:
        """清除所有缓存"""
        if not settings.cache_enabled:
            return False
        
        try:
            if self.redis_client:
                # 清除Redis缓存
                self.redis_client.flushdb()
            else:
                # 清除内存缓存
                with self.cache_lock:
                    self.memory_cache.clear()
            
            self.logger.info("缓存清除成功")
            return True
            
        except Exception as e:
            self.logger.error(f"清除缓存失败: {e}")
            return False
    
    def _cleanup_expired(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []
        
        for key, item in self.memory_cache.items():
            if current_time >= item['expires_at']:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        if expired_keys:
            self.logger.debug(f"清理了 {len(expired_keys)} 个过期缓存")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = {
            'enabled': settings.cache_enabled,
            'type': 'redis' if self.redis_client else 'memory',
            'total_items': 0,
            'memory_usage': 0
        }
        
        try:
            if self.redis_client:
                # Redis统计
                stats['total_items'] = self.redis_client.dbsize()
                info = self.redis_client.info('memory')
                stats['memory_usage'] = info.get('used_memory_human', '0B')
            else:
                # 内存统计
                with self.cache_lock:
                    stats['total_items'] = len(self.memory_cache)
                    # 估算内存使用
                    stats['memory_usage'] = f"{len(self.memory_cache) * 1024}B"  # 粗略估算
                
        except Exception as e:
            self.logger.error(f"获取缓存统计失败: {e}")
        
        return stats
    
    def health_check(self) -> bool:
        """健康检查"""
        if not settings.cache_enabled:
            return True
        
        try:
            if self.redis_client:
                # 检查Redis连接
                self.redis_client.ping()
                return True
            else:
                # 内存缓存总是可用
                return True
                
        except Exception as e:
            self.logger.error(f"缓存健康检查失败: {e}")
            return False