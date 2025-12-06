"""
API Key Management and Rotation

DEPRECATED: get_redis_url_with_auth() and validate_redis_auth() are kept only for legacy callers; 
will be removed after VCP 2.2 Wave 1. New code should import get_redis_url_with_auth from 
substrate.auth.redis_auth instead.

APIKeyManager remains here as it has no canonical equivalent yet.

Provides versioned API key management with rotation support.
"""

# Re-export get_redis_url_with_auth from canonical namespace for backward compatibility
from substrate.auth.redis_auth import get_redis_url_with_auth as _canonical_get_redis_url_with_auth

import os
import secrets
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path

from backend.security.runtime_env import get_required_env, MissingEnvironmentVariable

class APIKeyManager:
    """
    Manages API keys with versioning and rotation support.
    
    Keys are stored in format: mlvk-api-v{version}-{scope}-{random}
    """
    
    def __init__(self, keys_file: Optional[str] = None):
        """
        Initialize API key manager.
        
        Args:
            keys_file: Path to keys storage file (default: .api_keys.json)
        """
        self.keys_file = keys_file or os.path.join(
            os.path.dirname(__file__), 
            "../../.api_keys.json"
        )
        self.keys = self._load_keys()
    
    def _load_keys(self) -> Dict:
        """Load keys from storage file."""
        if not os.path.exists(self.keys_file):
            return {"keys": [], "active_key_id": None}
        
        try:
            with open(self.keys_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {"keys": [], "active_key_id": None}
    
    def _save_keys(self):
        """Save keys to storage file."""
        os.makedirs(os.path.dirname(self.keys_file), exist_ok=True)
        with open(self.keys_file, 'w') as f:
            json.dump(self.keys, f, indent=2)
    
    def generate_key(
        self, 
        scope: str = "readonly", 
        version: int = 1,
        expires_days: Optional[int] = None
    ) -> str:
        """
        Generate a new API key.
        
        Args:
            scope: Key scope (readonly, readwrite, admin)
            version: Key version number
            expires_days: Days until expiration (None = no expiry)
            
        Returns:
            Generated API key in format: mlvk-api-v{version}-{scope}-{random}
        """
        if scope not in ["readonly", "readwrite", "admin"]:
            raise ValueError(f"Invalid scope: {scope}")
        
        random_hex = secrets.token_hex(16)
        
        key = f"mlvk-api-v{version}-{scope}-{random_hex}"
        
        key_id = hashlib.sha256(key.encode()).hexdigest()[:16]
        
        created_at = datetime.utcnow().isoformat()
        expires_at = None
        if expires_days:
            expires_at = (datetime.utcnow() + timedelta(days=expires_days)).isoformat()
        
        key_data = {
            "key_id": key_id,
            "key_hash": hashlib.sha256(key.encode()).hexdigest(),
            "scope": scope,
            "version": version,
            "created_at": created_at,
            "expires_at": expires_at,
            "revoked": False
        }
        
        self.keys["keys"].append(key_data)
        
        if not self.keys["active_key_id"]:
            self.keys["active_key_id"] = key_id
        
        self._save_keys()
        
        return key
    
    def validate_key(self, key: str) -> tuple[bool, Optional[Dict]]:
        """
        Validate an API key.
        
        Args:
            key: API key to validate
            
        Returns:
            Tuple of (is_valid, key_metadata)
        """
        if not key or not key.startswith("mlvk-api-"):
            return False, None
        
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        for key_data in self.keys["keys"]:
            if key_data["key_hash"] == key_hash:
                if key_data["revoked"]:
                    return False, {"error": "Key revoked"}
                
                if key_data["expires_at"]:
                    expires = datetime.fromisoformat(key_data["expires_at"])
                    if datetime.utcnow() > expires:
                        return False, {"error": "Key expired"}
                
                return True, key_data
        
        return False, None
    
    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            key_id: Key ID to revoke
            
        Returns:
            True if key was revoked
        """
        for key_data in self.keys["keys"]:
            if key_data["key_id"] == key_id:
                key_data["revoked"] = True
                self._save_keys()
                return True
        return False
    
    def rotate_active_key(
        self, 
        scope: Optional[str] = None,
        expires_days: Optional[int] = 90
    ) -> str:
        """
        Rotate the active API key.
        
        Args:
            scope: Scope for new key (defaults to current active key scope)
            expires_days: Days until new key expires
            
        Returns:
            New API key
        """
        if not scope and self.keys["active_key_id"]:
            for key_data in self.keys["keys"]:
                if key_data["key_id"] == self.keys["active_key_id"]:
                    scope = key_data["scope"]
                    break
        
        scope = scope or "readonly"
        
        new_key = self.generate_key(scope=scope, expires_days=expires_days)
        
        new_key_hash = hashlib.sha256(new_key.encode()).hexdigest()
        new_key_id = None
        for key_data in self.keys["keys"]:
            if key_data["key_hash"] == new_key_hash:
                new_key_id = key_data["key_id"]
                break
        
        self.keys["active_key_id"] = new_key_id
        self._save_keys()
        
        return new_key
    
    def list_keys(self, include_revoked: bool = False) -> List[Dict]:
        """
        List all API keys.
        
        Args:
            include_revoked: Include revoked keys in list
            
        Returns:
            List of key metadata (without actual keys)
        """
        keys = []
        for key_data in self.keys["keys"]:
            if not include_revoked and key_data["revoked"]:
                continue
            
            safe_data = {k: v for k, v in key_data.items() if k != "key_hash"}
            keys.append(safe_data)
        
        return keys


# DEPRECATED: Use substrate.auth.redis_auth.get_redis_url_with_auth instead
def get_redis_url_with_auth() -> str:
    """
    Get Redis URL with authentication if configured.
    
    DEPRECATED: Import from substrate.auth.redis_auth instead.
    
    Returns:
        Redis URL with auth: redis://:<password>@host:port/db
        or rediss:// for TLS connections
    """
    return _canonical_get_redis_url_with_auth()


def validate_redis_auth() -> bool:
    """
    Validate Redis authentication is configured.
    
    Returns:
        True if Redis auth is properly configured
    """
    redis_url = os.getenv("REDIS_URL", "")
    redis_password = os.getenv("REDIS_PASSWORD")
    
    has_password = redis_password or ("@" in redis_url and ":" in redis_url.split("@")[0])
    
    uses_tls = redis_url.startswith("rediss://")
    
    return has_password or uses_tls
