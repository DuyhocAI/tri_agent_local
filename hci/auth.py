"""HCI authentication helpers — password gate + CSRF double-submit cookie."""
from __future__ import annotations

import hashlib
import secrets
from functools import wraps

from flask import session, request, jsonify, redirect, url_for

from config import HCI_CONFIG


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def check_password(password: str) -> bool:
    stored = HCI_CONFIG.get("password_hash", "")
    if not stored:
        return False  # HCI disabled if no password set
    candidate = _hash_password(password)
    return secrets.compare_digest(candidate, stored)


def hci_login_required(fn):
    """Decorator: redirect to /hci/login if not authenticated."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("hci_authenticated"):
            if request.is_json:
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("hci.login_page"))
        return fn(*args, **kwargs)
    return wrapper


def csrf_required(fn):
    """Decorator: require X-CSRF-Token header on mutating requests."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            token      = request.headers.get("X-CSRF-Token", "")
            sess_token = session.get("csrf_token", "")
            if not sess_token or not secrets.compare_digest(token, sess_token):
                return jsonify({"error": "Invalid CSRF token"}), 403
        return fn(*args, **kwargs)
    return wrapper


def generate_csrf_token() -> str:
    token = secrets.token_hex(32)
    session["csrf_token"] = token
    return token


def hci_enabled() -> bool:
    return bool(HCI_CONFIG.get("password_hash", ""))
