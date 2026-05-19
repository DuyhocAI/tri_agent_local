"""
HCI Blueprint — hci/routes.py  (P5)

Flask Blueprint mounted at /hci/.
Serves the dashboard HTML and handles login/logout.
"""
from __future__ import annotations

from flask import (
    Blueprint, render_template, request, session,
    redirect, url_for, jsonify,
)

from hci.auth import check_password, generate_csrf_token, hci_login_required, hci_enabled
from hci.api import api_bp

hci_bp = Blueprint("hci", __name__, url_prefix="/hci",
                   template_folder="../templates/hci")

# Register the API sub-blueprint
hci_bp.register_blueprint(api_bp)


@hci_bp.get("/")
@hci_bp.get("/dashboard")
def dashboard():
    if not hci_enabled():
        return (
            "<h2>HCI Dashboard disabled</h2>"
            "<p>Set <code>HERMES_HCI_PASSWORD_HASH</code> environment variable to enable.</p>"
            "<pre>$env:HERMES_HCI_PASSWORD_HASH = python -c \"import hashlib; "
            "print(hashlib.sha256(b'yourpassword').hexdigest())\"</pre>",
            503,
        )
    if not session.get("hci_authenticated"):
        return redirect(url_for("hci.login_page"))
    csrf = generate_csrf_token()
    return render_template("dashboard.html", csrf_token=csrf)


@hci_bp.get("/login")
def login_page():
    if session.get("hci_authenticated"):
        return redirect(url_for("hci.dashboard"))
    return render_template("login.html")


@hci_bp.post("/login")
def do_login():
    password = request.form.get("password", "")
    if check_password(password):
        session["hci_authenticated"] = True
        session.permanent = True
        return redirect(url_for("hci.dashboard"))
    return render_template("login.html", error="Incorrect password")


@hci_bp.post("/logout")
def do_logout():
    session.pop("hci_authenticated", None)
    session.pop("csrf_token", None)
    return redirect(url_for("hci.login_page"))


@hci_bp.get("/status")
def status():
    """Quick health endpoint — returns 200 if HCI is reachable."""
    return jsonify({"ok": True, "hci_enabled": hci_enabled()})
