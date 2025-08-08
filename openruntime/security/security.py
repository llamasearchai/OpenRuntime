#!/usr/bin/env python3
"""
OpenRuntime Enhanced Security Module
Comprehensive security features including authentication, authorization, and threat detection
"""

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import jwt
import redis.asyncio as redis
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logger = logging.getLogger("OpenRuntimeSecurity")

# =============================================================================
# Security Configuration and Models
# =============================================================================


class SecurityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(str, Enum):
    BRUTE_FORCE = "brute_force"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    COMMAND_INJECTION = "command_injection"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


@dataclass
class SecurityEvent:
    event_id: str
    timestamp: datetime
    source_ip: str
    user_id: Optional[str]
    threat_type: ThreatType
    severity: SecurityLevel
    description: str
    blocked: bool
    additional_data: Dict


@dataclass
class RateLimitRule:
    name: str
    max_requests: int
    time_window: int  # seconds
    block_duration: int  # seconds
    endpoints: List[str]
    user_specific: bool = True


class SecurityConfig(BaseModel):
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    allowed_hosts: List[str] = ["*"]
    blocked_ips: Set[str] = set()
    require_https: bool = True
    enable_rate_limiting: bool = True
    enable_threat_detection: bool = True


# =============================================================================
# Security Manager
# =============================================================================


class SecurityManager:
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.redis_client = None
        self.security_events: List[SecurityEvent] = []
        self.rate_limit_rules = self._initialize_rate_limits()
        self.threat_patterns = self._initialize_threat_patterns()
        self.blocked_tokens: Set[str] = set()
        self._rate_limit_cache: Dict = {}

    async def initialize(self):
        """Initialize async components"""
        try:
            self.redis_client = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Redis connection established for security features")
        except Exception as e:
            logger.warning(f"Redis unavailable, using in-memory security cache: {e}")
            self.redis_client = None

    def _initialize_rate_limits(self) -> List[RateLimitRule]:
        """Initialize rate limiting rules"""
        return [
            RateLimitRule(
                name="api_general",
                max_requests=100,
                time_window=60,
                block_duration=300,
                endpoints=["*"],
                user_specific=True,
            ),
            RateLimitRule(
                name="ai_tasks",
                max_requests=20,
                time_window=60,
                block_duration=600,
                endpoints=["/ai/tasks", "/ai/shell", "/ai/code"],
                user_specific=True,
            ),
            RateLimitRule(
                name="gpu_tasks",
                max_requests=50,
                time_window=60,
                block_duration=300,
                endpoints=["/tasks", "/benchmark"],
                user_specific=True,
            ),
            RateLimitRule(
                name="auth_endpoints",
                max_requests=5,
                time_window=300,
                block_duration=900,
                endpoints=["/auth/login", "/auth/register"],
                user_specific=False,
            ),
        ]

    def _initialize_threat_patterns(self) -> Dict[ThreatType, List[str]]:
        """Initialize threat detection patterns"""
        return {
            ThreatType.SQL_INJECTION: [
                r"(\bUNION\b.*\bSELECT\b)",
                r"(\bSELECT\b.*\bFROM\b.*\bWHERE\b)",
                r"(\bDROP\b.*\bTABLE\b)",
                r"(\bINSERT\b.*\bINTO\b)",
                r"(\bDELETE\b.*\bFROM\b)",
                r"(\bUPDATE\b.*\bSET\b)",
                r"('; *--)",
                r"(\b1=1\b|\b1 *= *1\b)",
            ],
            ThreatType.XSS: [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>",
                r"<object[^>]*>",
                r"<embed[^>]*>",
                r"eval\s*\(",
                r"expression\s*\(",
            ],
            ThreatType.COMMAND_INJECTION: [
                r";\s*(rm|del|format|fdisk)\s+",
                r"[;&|`]\s*(cat|type)\s+",
                r"[;&|`]\s*(ls|dir)\s+",
                r"\$\([^)]+\)",
                r"`[^`]+`",
                r"&&|\|\|",
                r">\s*/dev/",
                r"curl.*\|.*bash",
                r"wget.*\|.*sh",
            ],
        }

    # =============================================================================
    # Authentication and Authorization
    # =============================================================================

    def generate_token(self, user_id: str, permissions: List[str] = None) -> str:
        """Generate JWT token for user"""
        payload = {
            "user_id": user_id,
            "permissions": permissions or ["read"],
            "exp": datetime.utcnow() + timedelta(hours=self.config.jwt_expiration_hours),
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4()),  # JWT ID for revocation
        }

        token = jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
        return token

    def verify_token(self, token: str) -> Dict:
        """Verify and decode JWT token"""
        try:
            if token in self.blocked_tokens:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked"
                )

            payload = jwt.decode(
                token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    def revoke_token(self, token: str):
        """Add token to blacklist"""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm],
                options={"verify_exp": False},
            )
            self.blocked_tokens.add(payload.get("jti", token))
        except:
            self.blocked_tokens.add(token)

    def check_permissions(
        self, required_permissions: List[str], user_permissions: List[str]
    ) -> bool:
        """Check if user has required permissions"""
        if "admin" in user_permissions:
            return True
        return any(perm in user_permissions for perm in required_permissions)

    # =============================================================================
    # Rate Limiting
    # =============================================================================

    async def check_rate_limit(self, request: Request, user_id: Optional[str] = None) -> bool:
        """Check if request should be rate limited"""
        if not self.config.enable_rate_limiting:
            return True

        client_ip = self._get_client_ip(request)
        endpoint = request.url.path

        for rule in self.rate_limit_rules:
            if self._endpoint_matches_rule(endpoint, rule.endpoints):
                key = self._get_rate_limit_key(rule, client_ip, user_id)

                if self._is_rate_limited(key, rule):
                    await self._log_security_event(
                        source_ip=client_ip,
                        user_id=user_id,
                        threat_type=ThreatType.RATE_LIMIT_EXCEEDED,
                        severity=SecurityLevel.MEDIUM,
                        description=f"Rate limit exceeded for rule: {rule.name}",
                        blocked=True,
                        additional_data={"rule": rule.name, "endpoint": endpoint},
                    )
                    return False

                self._increment_rate_limit_counter(key, rule)

        return True

    def _is_rate_limited(self, key: str, rule: RateLimitRule) -> bool:
        """Check if key is currently rate limited"""
        cache_entry = self._rate_limit_cache.get(key)
        if cache_entry:
            if time.time() - cache_entry["timestamp"] > rule.time_window:
                # Reset counter if time window expired
                del self._rate_limit_cache[key]
                return False
            return cache_entry["count"] >= rule.max_requests
        return False

    def _increment_rate_limit_counter(self, key: str, rule: RateLimitRule):
        """Increment rate limit counter"""
        if key in self._rate_limit_cache:
            self._rate_limit_cache[key]["count"] += 1
        else:
            self._rate_limit_cache[key] = {"count": 1, "timestamp": time.time()}

    def _get_rate_limit_key(
        self, rule: RateLimitRule, client_ip: str, user_id: Optional[str]
    ) -> str:
        """Generate rate limit key"""
        if rule.user_specific and user_id:
            return f"rate_limit:{rule.name}:user:{user_id}"
        return f"rate_limit:{rule.name}:ip:{client_ip}"

    def _endpoint_matches_rule(self, endpoint: str, rule_endpoints: List[str]) -> bool:
        """Check if endpoint matches rate limit rule"""
        if "*" in rule_endpoints:
            return True
        return any(endpoint.startswith(pattern.rstrip("*")) for pattern in rule_endpoints)

    # =============================================================================
    # Threat Detection
    # =============================================================================

    async def scan_for_threats(self, request: Request, content: str = None) -> List[ThreatType]:
        """Scan request for security threats"""
        if not self.config.enable_threat_detection:
            return []

        threats = []

        # Scan URL parameters
        if request.query_params:
            query_string = str(request.query_params)
            threats.extend(self._scan_content(query_string))

        # Scan request body if provided
        if content:
            threats.extend(self._scan_content(content))

        # Scan headers for suspicious patterns
        for header_name, header_value in request.headers.items():
            if isinstance(header_value, str):
                threats.extend(self._scan_content(header_value))

        # Log threats if found
        if threats:
            client_ip = self._get_client_ip(request)
            for threat in threats:
                await self._log_security_event(
                    source_ip=client_ip,
                    user_id=None,
                    threat_type=threat,
                    severity=SecurityLevel.HIGH,
                    description=f"Potential {threat.value} detected in request",
                    blocked=True,
                    additional_data={"endpoint": request.url.path},
                )

        return threats

    def _scan_content(self, content: str) -> List[ThreatType]:
        """Scan content for threat patterns"""
        threats = []
        content_lower = content.lower()

        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    threats.append(threat_type)
                    break  # One match per threat type is enough

        return threats

    # =============================================================================
    # IP and Host Validation
    # =============================================================================

    def validate_request_source(self, request: Request) -> bool:
        """Validate request source (IP and host)"""
        client_ip = self._get_client_ip(request)

        # Check blocked IPs
        if client_ip in self.config.blocked_ips:
            return False

        # Check allowed hosts
        if self.config.allowed_hosts != ["*"]:
            host = request.headers.get("host", "")
            if not any(host.endswith(allowed) for allowed in self.config.allowed_hosts):
                return False

        # Check for private IP addresses in production
        try:
            ip_addr = ipaddress.ip_address(client_ip)
            if ip_addr.is_private and self.config.require_https:
                # Allow private IPs only for development
                logger.debug(f"Private IP {client_ip} allowed for development")
        except ValueError:
            # Invalid IP address
            return False

        return True

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check X-Forwarded-For header (from load balancer/proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to direct connection IP
        return request.client.host if request.client else "unknown"

    # =============================================================================
    # Security Event Logging
    # =============================================================================

    async def _log_security_event(
        self,
        source_ip: str,
        user_id: Optional[str],
        threat_type: ThreatType,
        severity: SecurityLevel,
        description: str,
        blocked: bool,
        additional_data: Dict = None,
    ):
        """Log security event"""
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            user_id=user_id,
            threat_type=threat_type,
            severity=severity,
            description=description,
            blocked=blocked,
            additional_data=additional_data or {},
        )

        self.security_events.append(event)

        # Log to structured logger
        logger.warning(
            f"Security Event: {threat_type.value} from {source_ip}",
            extra={
                "event_id": event.event_id,
                "threat_type": threat_type.value,
                "severity": severity.value,
                "blocked": blocked,
                "user_id": user_id,
                "additional_data": additional_data,
            },
        )

        # Store in Redis for centralized monitoring
        if self.redis_client:
            try:
                await self.redis_client.lpush(
                    "security_events",
                    json.dumps(
                        {
                            "event_id": event.event_id,
                            "timestamp": event.timestamp.isoformat(),
                            "source_ip": event.source_ip,
                            "user_id": event.user_id,
                            "threat_type": event.threat_type.value,
                            "severity": event.severity.value,
                            "description": event.description,
                            "blocked": event.blocked,
                            "additional_data": event.additional_data,
                        }
                    ),
                )
                # Keep only last 1000 events
                await self.redis_client.ltrim("security_events", 0, 999)
            except Exception as e:
                logger.error(f"Failed to store security event in Redis: {e}")

    def get_security_summary(self) -> Dict:
        """Get security events summary"""
        recent_events = [
            e for e in self.security_events if e.timestamp > datetime.utcnow() - timedelta(hours=24)
        ]

        summary = {
            "total_events_24h": len(recent_events),
            "blocked_events_24h": len([e for e in recent_events if e.blocked]),
            "threat_types": {},
            "severity_breakdown": {},
            "top_source_ips": {},
        }

        for event in recent_events:
            # Count threat types
            threat_type = event.threat_type.value
            summary["threat_types"][threat_type] = summary["threat_types"].get(threat_type, 0) + 1

            # Count severity levels
            severity = event.severity.value
            summary["severity_breakdown"][severity] = (
                summary["severity_breakdown"].get(severity, 0) + 1
            )

            # Count source IPs
            ip = event.source_ip
            summary["top_source_ips"][ip] = summary["top_source_ips"].get(ip, 0) + 1

        # Sort top IPs
        summary["top_source_ips"] = dict(
            sorted(summary["top_source_ips"].items(), key=lambda x: x[1], reverse=True)[:10]
        )

        return summary


# =============================================================================
# Security Middleware
# =============================================================================


class SecurityMiddleware:
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager

    async def __call__(self, request: Request, call_next):
        # Validate request source
        if not self.security_manager.validate_request_source(request):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Request blocked by security policy"
            )

        # Check rate limits
        if not await self.security_manager.check_rate_limit(request):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded"
            )

        # Scan for threats (basic scan without body)
        threats = await self.security_manager.scan_for_threats(request)
        if threats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request blocked due to security threat",
            )

        response = await call_next(request)
        return response


# =============================================================================
# Security Utilities
# =============================================================================


def hash_password(password: str, salt: str = None) -> Tuple[str, str]:
    """Hash password with salt"""
    if salt is None:
        salt = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]

    pwdhash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return pwdhash.hex(), salt


def verify_password(password: str, hashed: str, salt: str) -> bool:
    """Verify password against hash"""
    return hash_password(password, salt)[0] == hashed


def generate_api_key() -> str:
    """Generate secure API key"""
    return hashlib.sha256(f"{uuid.uuid4()}{time.time()}".encode()).hexdigest()


def secure_compare(a: str, b: str) -> bool:
    """Constant-time string comparison"""
    return hmac.compare_digest(a, b)
