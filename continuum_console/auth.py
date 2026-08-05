from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from http.cookies import SimpleCookie

COOKIE_NAME = "continuum_session"


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


@dataclass(frozen=True)
class AuthManager:
    username: str
    password: str
    session_secret: str
    required: bool = True
    ttl_seconds: int = 12 * 60 * 60
    secure_cookie: bool = False

    @classmethod
    def from_environment(cls, required: bool | None = None) -> "AuthManager":
        password = os.environ.get("CONTINUUM_ADMIN_PASSWORD", "")
        session_secret = os.environ.get("CONTINUUM_SESSION_SECRET", "")
        is_required = required if required is not None else os.environ.get("CONTINUUM_REQUIRE_AUTH", "0") == "1"
        if is_required and (len(password) < 12 or len(session_secret) < 32):
            raise RuntimeError(
                "Authentication requires CONTINUUM_ADMIN_PASSWORD (12+ characters) "
                "and CONTINUUM_SESSION_SECRET (32+ characters)."
            )
        return cls(
            username=os.environ.get("CONTINUUM_ADMIN_USER", "admin"),
            password=password,
            session_secret=session_secret or "local-development-session-secret-only",
            required=is_required,
            secure_cookie=os.environ.get("CONTINUUM_SECURE_COOKIE", "0") == "1",
        )

    def verify_credentials(self, username: str, password: str) -> bool:
        return hmac.compare_digest(username, self.username) and hmac.compare_digest(password, self.password)

    def issue(self) -> str:
        payload = _b64encode(json.dumps({"sub": self.username, "exp": int(time.time()) + self.ttl_seconds}, separators=(",", ":")).encode())
        signature = _b64encode(hmac.new(self.session_secret.encode(), payload.encode(), hashlib.sha256).digest())
        return payload + "." + signature

    def verify_token(self, token: str) -> bool:
        try:
            payload, signature = token.split(".", 1)
            expected = _b64encode(hmac.new(self.session_secret.encode(), payload.encode(), hashlib.sha256).digest())
            if not hmac.compare_digest(signature, expected):
                return False
            claims = json.loads(_b64decode(payload))
            return claims.get("sub") == self.username and int(claims.get("exp", 0)) > int(time.time())
        except (ValueError, TypeError, json.JSONDecodeError):
            return False

    def authenticated(self, cookie_header: str) -> bool:
        if not self.required:
            return True
        cookie = SimpleCookie()
        cookie.load(cookie_header or "")
        morsel = cookie.get(COOKIE_NAME)
        return bool(morsel and self.verify_token(morsel.value))

    def session_cookie(self, token: str) -> str:
        secure = "; Secure" if self.secure_cookie else ""
        return f"{COOKIE_NAME}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={self.ttl_seconds}{secure}"

    def clear_cookie(self) -> str:
        secure = "; Secure" if self.secure_cookie else ""
        return f"{COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0{secure}"
