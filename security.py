"""Optional password protection, same-origin check and response headers."""

from __future__ import annotations

import base64
import binascii
import os
import secrets
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _credentials_match(header: str, username: str, password: str) -> bool:
    scheme, _, encoded = header.partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False
    given_user, separator, given_password = decoded.partition(":")
    if not separator:
        return False
    # compare_digest avoids leaking information through response timing.
    user_ok = secrets.compare_digest(given_user.encode(), username.encode())
    password_ok = secrets.compare_digest(given_password.encode(), password.encode())
    return user_ok and password_ok


def install(app: FastAPI) -> None:
    username = os.getenv("APP_USERNAME", "")
    password = os.getenv("APP_PASSWORD", "")
    if os.getenv("REQUIRE_AUTH") == "1" and not (username and password):
        raise RuntimeError("REQUIRE_AUTH=1 but APP_USERNAME / APP_PASSWORD are not set.")

    @app.middleware("http")
    async def protect(request: Request, call_next) -> Response:
        path = request.url.path
        if username and password and path != "/health":
            if not _credentials_match(request.headers.get("authorization", ""), username, password):
                return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Student Risk"'})

        # Browsers send Origin on cross-site form posts; refuse any that is not this site.
        origin = request.headers.get("origin")
        if request.method in UNSAFE_METHODS and origin:
            if urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "Cross-origin request refused."}, status_code=403)

        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if path not in {"/docs", "/redoc"}:  # Swagger UI loads its assets from a CDN.
            response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        if path.startswith("/api/") or path.startswith("/students/"):
            response.headers["Cache-Control"] = "no-store"
        return response
