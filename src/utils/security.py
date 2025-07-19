"""
Security utilities for the Screenshot Authenticity AI API
"""
import hashlib
import hmac
import jwt
import time
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from passlib.context import CryptContext
from passlib.hash import bcrypt
import secrets
import logging

logger = logging.getLogger(__name__)


class SecurityManager:
    """
    Security manager for authentication and authorization
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Initialize security manager
        
        Args:
            secret_key: Secret key for JWT tokens
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = "HS256"
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # API key store (in production, this would be a database)
        self.api_keys: Dict[str, Dict] = {}
        
        # Session store (in production, this would be Redis or similar)
        self.sessions: Dict[str, Dict] = {}
    
    def generate_api_key(self, user_id: str, permissions: Optional[list] = None) -> str:
        """
        Generate API key for a user
        
        Args:
            user_id: User identifier
            permissions: List of permissions for this key
            
        Returns:
            Generated API key
        """
        api_key = f"sa_{''.join(secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(32))}"
        
        self.api_keys[api_key] = {
            'user_id': user_id,
            'permissions': permissions or ['verify:screenshot'],
            'created_at': time.time(),
            'last_used': None,
            'active': True
        }
        
        logger.info(f"Generated API key for user {user_id}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """
        Validate API key
        
        Args:
            api_key: API key to validate
            
        Returns:
            User information if valid, None otherwise
        """
        key_info = self.api_keys.get(api_key)
        
        if not key_info or not key_info.get('active'):
            return None
        
        # Update last used timestamp
        key_info['last_used'] = time.time()
        
        return {
            'user_id': key_info['user_id'],
            'permissions': key_info['permissions'],
            'auth_method': 'api_key'
        }
    
    def create_access_token(self, user_id: str, permissions: Optional[list] = None, expires_delta: Optional[timedelta] = None) -> str:
        """
        Create JWT access token
        
        Args:
            user_id: User identifier
            permissions: List of permissions
            expires_delta: Token expiration time
            
        Returns:
            JWT token
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=30)
        
        to_encode = {
            'sub': user_id,
            'permissions': permissions or ['verify:screenshot'],
            'exp': expire,
            'iat': datetime.utcnow(),
            'type': 'access_token'
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        logger.info(f"Created access token for user {user_id}")
        return encoded_jwt
    
    def validate_token(self, token: str) -> Optional[Dict]:
        """
        Validate JWT token or API key
        
        Args:
            token: Token to validate
            
        Returns:
            User information if valid, None otherwise
        """
        # Try API key first
        if token.startswith('sa_'):
            return self.validate_api_key(token)
        
        # Try JWT token
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is expired
            if payload.get('exp') and payload['exp'] < time.time():
                return None
            
            return {
                'user_id': payload.get('sub'),
                'permissions': payload.get('permissions', []),
                'auth_method': 'jwt_token'
            }
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke API key
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if revoked successfully
        """
        if api_key in self.api_keys:
            self.api_keys[api_key]['active'] = False
            self.api_keys[api_key]['revoked_at'] = time.time()
            logger.info(f"Revoked API key: {api_key[:10]}...")
            return True
        return False
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_session(self, user_id: str, metadata: Optional[Dict] = None) -> str:
        """
        Create user session
        
        Args:
            user_id: User identifier
            metadata: Additional session metadata
            
        Returns:
            Session ID
        """
        session_id = secrets.token_urlsafe(32)
        
        self.sessions[session_id] = {
            'user_id': user_id,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'metadata': metadata or {},
            'active': True
        }
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict]:
        """
        Validate session
        
        Args:
            session_id: Session ID to validate
            
        Returns:
            Session information if valid
        """
        session = self.sessions.get(session_id)
        
        if not session or not session.get('active'):
            return None
        
        # Check if session is expired (24 hours)
        if time.time() - session['created_at'] > 24 * 3600:
            session['active'] = False
            return None
        
        # Update last accessed
        session['last_accessed'] = time.time()
        
        return session
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session['created_at'] > 24 * 3600:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


class DataProtection:
    """
    Data protection utilities for sensitive information
    """
    
    @staticmethod
    def anonymize_ip(ip_address: str) -> str:
        """
        Anonymize IP address for privacy
        
        Args:
            ip_address: IP address to anonymize
            
        Returns:
            Anonymized IP address
        """
        if ':' in ip_address:  # IPv6
            parts = ip_address.split(':')
            return ':'.join(parts[:4] + ['0000'] * (len(parts) - 4))
        else:  # IPv4
            parts = ip_address.split('.')
            return '.'.join(parts[:3] + ['0'])
    
    @staticmethod
    def hash_sensitive_data(data: str, salt: Optional[str] = None) -> str:
        """
        Hash sensitive data with salt
        
        Args:
            data: Data to hash
            salt: Salt for hashing (generated if not provided)
            
        Returns:
            Hashed data with salt
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        combined = f"{salt}{data}"
        hash_obj = hashlib.sha256(combined.encode())
        return f"{salt}:{hash_obj.hexdigest()}"
    
    @staticmethod
    def verify_hashed_data(data: str, hashed_data: str) -> bool:
        """
        Verify data against hash
        
        Args:
            data: Original data
            hashed_data: Hashed data with salt
            
        Returns:
            True if data matches hash
        """
        try:
            salt, hash_value = hashed_data.split(':', 1)
            combined = f"{salt}{data}"
            hash_obj = hashlib.sha256(combined.encode())
            return hash_obj.hexdigest() == hash_value
        except ValueError:
            return False
    
    @staticmethod
    def generate_csrf_token() -> str:
        """Generate CSRF token"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def validate_csrf_token(token: str, expected_token: str) -> bool:
        """Validate CSRF token"""
        return hmac.compare_digest(token, expected_token)


class InputValidator:
    """
    Input validation utilities
    """
    
    @staticmethod
    def validate_file_type(filename: str, allowed_extensions: list) -> bool:
        """
        Validate file type by extension
        
        Args:
            filename: Name of the file
            allowed_extensions: List of allowed extensions
            
        Returns:
            True if file type is allowed
        """
        if not filename:
            return False
        
        extension = filename.lower().split('.')[-1]
        return extension in [ext.lower() for ext in allowed_extensions]
    
    @staticmethod
    def validate_file_size(file_size: int, max_size: int) -> bool:
        """
        Validate file size
        
        Args:
            file_size: Size of the file in bytes
            max_size: Maximum allowed size in bytes
            
        Returns:
            True if file size is within limits
        """
        return 0 < file_size <= max_size
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent path traversal
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        import re
        
        # Remove path separators and dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = re.sub(r'\.\.', '', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1)
            filename = name[:250] + '.' + ext
        
        return filename
    
    @staticmethod
    def validate_request_id(request_id: str) -> bool:
        """
        Validate request ID format
        
        Args:
            request_id: Request ID to validate
            
        Returns:
            True if valid format
        """
        import re
        
        # UUID4 format: 8-4-4-4-12 hex digits
        pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        return bool(re.match(pattern, request_id.lower()))


class SecurityAudit:
    """
    Security audit logging and monitoring
    """
    
    def __init__(self):
        self.audit_log = []
        self.security_events = []
    
    def log_auth_attempt(self, user_id: str, success: bool, ip_address: str, user_agent: str):
        """Log authentication attempt"""
        event = {
            'type': 'auth_attempt',
            'user_id': user_id,
            'success': success,
            'ip_address': DataProtection.anonymize_ip(ip_address),
            'user_agent': user_agent[:100],  # Truncate user agent
            'timestamp': time.time()
        }
        
        self.audit_log.append(event)
        
        if not success:
            self.security_events.append({
                'type': 'failed_auth',
                'user_id': user_id,
                'ip_address': DataProtection.anonymize_ip(ip_address),
                'timestamp': time.time()
            })
        
        logger.info(f"Auth attempt: user={user_id}, success={success}")
    
    def log_api_access(self, user_id: str, endpoint: str, ip_address: str, response_code: int):
        """Log API access"""
        event = {
            'type': 'api_access',
            'user_id': user_id,
            'endpoint': endpoint,
            'ip_address': DataProtection.anonymize_ip(ip_address),
            'response_code': response_code,
            'timestamp': time.time()
        }
        
        self.audit_log.append(event)
    
    def detect_suspicious_activity(self, user_id: str, ip_address: str) -> bool:
        """
        Detect suspicious activity patterns
        
        Args:
            user_id: User identifier
            ip_address: Client IP address
            
        Returns:
            True if suspicious activity detected
        """
        current_time = time.time()
        recent_events = [
            event for event in self.security_events
            if current_time - event['timestamp'] < 3600  # Last hour
        ]
        
        # Check for multiple failed auth attempts
        failed_auths = [
            event for event in recent_events
            if (event['type'] == 'failed_auth' and 
                (event['user_id'] == user_id or event['ip_address'] == DataProtection.anonymize_ip(ip_address)))
        ]
        
        if len(failed_auths) >= 5:
            logger.warning(f"Suspicious activity detected: multiple failed auths for user {user_id}")
            return True
        
        return False
    
    def get_audit_summary(self, hours: int = 24) -> Dict:
        """
        Get audit summary for the specified time period
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Audit summary
        """
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [
            event for event in self.audit_log
            if event['timestamp'] > cutoff_time
        ]
        
        summary = {
            'total_events': len(recent_events),
            'auth_attempts': len([e for e in recent_events if e['type'] == 'auth_attempt']),
            'failed_auths': len([e for e in recent_events if e['type'] == 'auth_attempt' and not e['success']]),
            'api_accesses': len([e for e in recent_events if e['type'] == 'api_access']),
            'unique_users': len(set(e['user_id'] for e in recent_events if e.get('user_id'))),
            'unique_ips': len(set(e['ip_address'] for e in recent_events if e.get('ip_address'))),
            'time_period_hours': hours
        }
        
        return summary