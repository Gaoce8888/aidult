"""
Rate limiting utilities for the Screenshot Authenticity AI API
"""
import time
from typing import Dict, Optional
from collections import defaultdict, deque
import threading


class RateLimiter:
    """
    Token bucket rate limiter implementation
    """
    
    def __init__(self, requests: int, window: int):
        """
        Initialize rate limiter
        
        Args:
            requests: Number of requests allowed per window
            window: Time window in seconds
        """
        self.requests = requests
        self.window = window
        self.clients: Dict[str, deque] = defaultdict(lambda: deque())
        self.lock = threading.Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client
        
        Args:
            client_id: Client identifier (e.g., IP address)
            
        Returns:
            True if request is allowed, False otherwise
        """
        current_time = time.time()
        
        with self.lock:
            # Get client request history
            client_requests = self.clients[client_id]
            
            # Remove old requests outside the window
            while client_requests and client_requests[0] <= current_time - self.window:
                client_requests.popleft()
            
            # Check if under limit
            if len(client_requests) < self.requests:
                client_requests.append(current_time)
                return True
            
            return False
    
    def get_client_info(self, client_id: str) -> Dict:
        """
        Get information about client's current rate limit status
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dictionary with client rate limit info
        """
        current_time = time.time()
        
        with self.lock:
            client_requests = self.clients[client_id]
            
            # Remove old requests
            while client_requests and client_requests[0] <= current_time - self.window:
                client_requests.popleft()
            
            remaining_requests = max(0, self.requests - len(client_requests))
            
            # Calculate reset time
            reset_time = None
            if client_requests:
                reset_time = client_requests[0] + self.window
            
            return {
                'limit': self.requests,
                'remaining': remaining_requests,
                'used': len(client_requests),
                'window_seconds': self.window,
                'reset_time': reset_time
            }
    
    def reset_client(self, client_id: str):
        """Reset rate limit for a specific client"""
        with self.lock:
            if client_id in self.clients:
                del self.clients[client_id]
    
    def cleanup_old_clients(self, max_age: int = 3600):
        """
        Cleanup old client records
        
        Args:
            max_age: Maximum age in seconds to keep client records
        """
        current_time = time.time()
        cutoff_time = current_time - max_age
        
        with self.lock:
            clients_to_remove = []
            
            for client_id, requests in self.clients.items():
                # Remove old requests from this client
                while requests and requests[0] <= current_time - self.window:
                    requests.popleft()
                
                # If no recent requests, mark for removal
                if not requests or requests[-1] <= cutoff_time:
                    clients_to_remove.append(client_id)
            
            # Remove old clients
            for client_id in clients_to_remove:
                del self.clients[client_id]


class AdvancedRateLimiter:
    """
    Advanced rate limiter with multiple tiers and burst handling
    """
    
    def __init__(self, tiers: Optional[Dict] = None):
        """
        Initialize advanced rate limiter
        
        Args:
            tiers: Dictionary of rate limit tiers
        """
        self.tiers = tiers or {
            'free': {'requests': 100, 'window': 3600, 'burst': 10},
            'premium': {'requests': 1000, 'window': 3600, 'burst': 50},
            'enterprise': {'requests': 10000, 'window': 3600, 'burst': 100}
        }
        
        self.limiters = {}
        self.client_tiers: Dict[str, str] = defaultdict(lambda: 'free')
        
        # Initialize limiters for each tier
        for tier, config in self.tiers.items():
            self.limiters[tier] = RateLimiter(
                config['requests'], 
                config['window']
            )
    
    def set_client_tier(self, client_id: str, tier: str):
        """Set tier for a client"""
        if tier in self.tiers:
            self.client_tiers[client_id] = tier
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client with tier consideration"""
        tier = self.client_tiers[client_id]
        limiter = self.limiters.get(tier)
        
        if limiter:
            return limiter.is_allowed(client_id)
        
        return False
    
    def get_client_info(self, client_id: str) -> Dict:
        """Get client rate limit info including tier"""
        tier = self.client_tiers[client_id]
        limiter = self.limiters.get(tier)
        
        if limiter:
            info = limiter.get_client_info(client_id)
            info['tier'] = tier
            info['tier_config'] = self.tiers[tier]
            return info
        
        return {'error': 'Invalid tier'}


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts based on system load
    """
    
    def __init__(self, base_requests: int, base_window: int):
        """
        Initialize adaptive rate limiter
        
        Args:
            base_requests: Base number of requests per window
            base_window: Base time window in seconds
        """
        self.base_requests = base_requests
        self.base_window = base_window
        self.current_multiplier = 1.0
        self.load_history = deque(maxlen=100)
        self.limiter = RateLimiter(base_requests, base_window)
        
    def update_system_load(self, load_percentage: float):
        """
        Update system load and adjust rate limits
        
        Args:
            load_percentage: Current system load as percentage (0-100)
        """
        self.load_history.append(load_percentage)
        
        # Calculate average load over recent history
        if len(self.load_history) >= 10:
            avg_load = sum(self.load_history) / len(self.load_history)
            
            # Adjust multiplier based on load
            if avg_load > 80:
                self.current_multiplier = 0.5  # Reduce by half under high load
            elif avg_load > 60:
                self.current_multiplier = 0.7
            elif avg_load < 30:
                self.current_multiplier = 1.5  # Increase under low load
            else:
                self.current_multiplier = 1.0
            
            # Update limiter with new limits
            new_requests = int(self.base_requests * self.current_multiplier)
            self.limiter = RateLimiter(new_requests, self.base_window)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed with adaptive limits"""
        return self.limiter.is_allowed(client_id)
    
    def get_current_limits(self) -> Dict:
        """Get current adaptive limits"""
        return {
            'base_requests': self.base_requests,
            'current_requests': int(self.base_requests * self.current_multiplier),
            'multiplier': self.current_multiplier,
            'window': self.base_window,
            'avg_load': sum(self.load_history) / len(self.load_history) if self.load_history else 0
        }