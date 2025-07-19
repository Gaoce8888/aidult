"""
安全模块
实现API密钥验证和限流功能
"""
import time
import logging
from typing import Dict, Optional
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from collections import defaultdict, deque
from threading import Lock

from config.settings import settings

# 安全认证
security = HTTPBearer(auto_error=False)

# 限流存储
rate_limit_store = defaultdict(lambda: deque(maxlen=1000))
rate_limit_lock = Lock()


def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> str:
    """验证API密钥"""
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="缺少API密钥"
        )
    
    api_key = credentials.credentials
    
    # 简单的API密钥验证（实际应用中应该使用数据库或配置文件）
    valid_api_keys = [
        "test-api-key-123",
        "demo-api-key-456",
        "production-api-key-789"
    ]
    
    # 检查API密钥是否有效
    if api_key not in valid_api_keys:
        raise HTTPException(
            status_code=401,
            detail="无效的API密钥"
        )
    
    return api_key


def check_rate_limit(request: Request, api_key: str) -> bool:
    """检查限流"""
    current_time = time.time()
    window_start = current_time - 60  # 1分钟窗口
    
    with rate_limit_lock:
        # 获取该API密钥的请求记录
        requests = rate_limit_store[api_key]
        
        # 清理过期的请求记录
        while requests and requests[0] < window_start:
            requests.popleft()
        
        # 检查是否超过限制
        if len(requests) >= settings.rate_limit_per_minute:
            return False
        
        # 记录当前请求
        requests.append(current_time)
        
        return True


def get_rate_limit_info(api_key: str) -> Dict:
    """获取限流信息"""
    current_time = time.time()
    window_start = current_time - 60
    
    with rate_limit_lock:
        requests = rate_limit_store[api_key]
        
        # 清理过期记录
        while requests and requests[0] < window_start:
            requests.popleft()
        
        return {
            'current_requests': len(requests),
            'limit': settings.rate_limit_per_minute,
            'remaining': max(0, settings.rate_limit_per_minute - len(requests)),
            'reset_time': window_start + 60
        }


class SecurityManager:
    """安全管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.failed_attempts = defaultdict(int)
        self.blocked_ips = set()
        self.blocked_apis = set()
        self.security_lock = Lock()
        
        self.logger.info("安全管理器初始化完成")
    
    def check_request_security(self, request: Request, api_key: str) -> Dict:
        """检查请求安全性"""
        security_info = {
            'ip_address': self._get_client_ip(request),
            'user_agent': request.headers.get('user-agent', ''),
            'api_key': api_key,
            'timestamp': time.time(),
            'is_blocked': False,
            'risk_score': 0
        }
        
        # 检查IP是否被阻止
        if security_info['ip_address'] in self.blocked_ips:
            security_info['is_blocked'] = True
            security_info['risk_score'] = 100
            return security_info
        
        # 检查API密钥是否被阻止
        if api_key in self.blocked_apis:
            security_info['is_blocked'] = True
            security_info['risk_score'] = 100
            return security_info
        
        # 检查限流
        if not check_rate_limit(request, api_key):
            security_info['risk_score'] += 50
        
        # 检查失败次数
        key = f"{security_info['ip_address']}:{api_key}"
        failed_count = self.failed_attempts.get(key, 0)
        if failed_count > 10:
            security_info['risk_score'] += 30
        
        # 检查User-Agent
        if not security_info['user_agent'] or 'bot' in security_info['user_agent'].lower():
            security_info['risk_score'] += 10
        
        return security_info
    
    def record_failed_attempt(self, ip_address: str, api_key: str):
        """记录失败尝试"""
        with self.security_lock:
            key = f"{ip_address}:{api_key}"
            self.failed_attempts[key] += 1
            
            # 如果失败次数过多，阻止该组合
            if self.failed_attempts[key] > 20:
                self.blocked_ips.add(ip_address)
                self.blocked_apis.add(api_key)
                self.logger.warning(f"阻止IP {ip_address} 和API密钥 {api_key}")
    
    def record_successful_attempt(self, ip_address: str, api_key: str):
        """记录成功尝试"""
        with self.security_lock:
            key = f"{ip_address}:{api_key}"
            if key in self.failed_attempts:
                self.failed_attempts[key] = max(0, self.failed_attempts[key] - 1)
    
    def unblock_ip(self, ip_address: str):
        """解除IP阻止"""
        with self.security_lock:
            self.blocked_ips.discard(ip_address)
            self.logger.info(f"解除IP {ip_address} 的阻止")
    
    def unblock_api_key(self, api_key: str):
        """解除API密钥阻止"""
        with self.security_lock:
            self.blocked_apis.discard(api_key)
            self.logger.info(f"解除API密钥 {api_key} 的阻止")
    
    def get_security_stats(self) -> Dict:
        """获取安全统计信息"""
        with self.security_lock:
            return {
                'blocked_ips': len(self.blocked_ips),
                'blocked_apis': len(self.blocked_apis),
                'failed_attempts': dict(self.failed_attempts),
                'total_failed_attempts': sum(self.failed_attempts.values())
            }
    
    def _get_client_ip(self, request: Request) -> str:
        """获取客户端IP地址"""
        # 检查代理头
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # 获取直接IP
        return request.client.host if request.client else 'unknown'
    
    def cleanup_old_records(self, max_age: int = 3600):
        """清理旧记录"""
        current_time = time.time()
        
        with self.security_lock:
            # 这里可以实现更复杂的清理逻辑
            # 例如：清理超过1小时的失败记录
            pass


# 全局安全管理器实例
security_manager = SecurityManager()


def get_security_manager() -> SecurityManager:
    """获取安全管理器实例"""
    return security_manager


def validate_request_security(request: Request, api_key: str):
    """验证请求安全性"""
    security_info = security_manager.check_request_security(request, api_key)
    
    if security_info['is_blocked']:
        raise HTTPException(
            status_code=403,
            detail="请求被阻止"
        )
    
    if security_info['risk_score'] > 80:
        raise HTTPException(
            status_code=429,
            detail="请求频率过高"
        )
    
    return security_info


def sanitize_input(data: str) -> str:
    """清理输入数据"""
    import re
    
    # 移除潜在的恶意字符
    sanitized = re.sub(r'[<>"\']', '', data)
    
    # 限制长度
    if len(sanitized) > 10000:
        sanitized = sanitized[:10000]
    
    return sanitized


def validate_image_data(image_data: str) -> bool:
    """验证图像数据"""
    import base64
    
    try:
        # 检查Base64格式
        decoded = base64.b64decode(image_data)
        
        # 检查文件大小
        if len(decoded) > settings.max_image_size:
            return False
        
        # 检查文件头
        if len(decoded) < 8:
            return False
        
        # 检查常见图像格式的文件头
        headers = {
            b'\xff\xd8\xff': 'JPEG',
            b'\x89PNG\r\n\x1a\n': 'PNG',
            b'GIF87a': 'GIF',
            b'GIF89a': 'GIF',
            b'BM': 'BMP',
            b'RIFF': 'WEBP'
        }
        
        for header, format_name in headers.items():
            if decoded.startswith(header):
                return True
        
        return False
        
    except Exception:
        return False


def encrypt_sensitive_data(data: str) -> str:
    """加密敏感数据"""
    from cryptography.fernet import Fernet
    
    try:
        # 使用配置中的密钥
        key = settings.secret_key.encode()
        if len(key) != 32:
            # 如果密钥长度不对，生成一个合适的密钥
            import hashlib
            key = hashlib.sha256(key).digest()
        
        f = Fernet(base64.urlsafe_b64encode(key))
        encrypted = f.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()
        
    except Exception as e:
        logging.error(f"加密失败: {e}")
        return data


def decrypt_sensitive_data(encrypted_data: str) -> str:
    """解密敏感数据"""
    from cryptography.fernet import Fernet
    
    try:
        # 使用配置中的密钥
        key = settings.secret_key.encode()
        if len(key) != 32:
            # 如果密钥长度不对，生成一个合适的密钥
            import hashlib
            key = hashlib.sha256(key).digest()
        
        f = Fernet(base64.urlsafe_b64encode(key))
        decoded = base64.b64decode(encrypted_data)
        decrypted = f.decrypt(decoded)
        return decrypted.decode()
        
    except Exception as e:
        logging.error(f"解密失败: {e}")
        return encrypted_data