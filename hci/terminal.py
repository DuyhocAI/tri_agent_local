"""
HCI Terminal — hci/terminal.py  (P5)

Provides a real Windows PTY terminal via SocketIO namespace /hci-terminal.
Uses pywinpty for proper PTY support on Windows. Degrades gracefully if
pywinpty is not installed.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger("hermes.hci.terminal")

_WINPTY_AVAILABLE = False
try:
    import winpty  # type: ignore
    _WINPTY_AVAILABLE = True
except ImportError:
    pass


class TerminalSession:
    """One PTY process per connected WebSocket client."""

    def __init__(self, sid: str, socketio, shell: str = "powershell.exe"):
        self._sid      = sid
        self._socketio = socketio
        self._proc     = None
        self._thread   = None
        if _WINPTY_AVAILABLE:
            self._start(shell)

    def _start(self, shell: str) -> None:
        try:
            self._proc = winpty.PtyProcess.spawn(shell)
            self._thread = threading.Thread(
                target=self._read_loop, daemon=True,
                name=f"pty-{self._sid[:8]}"
            )
            self._thread.start()
        except Exception as e:
            logger.warning(f"PTY start failed for {self._sid}: {e}")

    def _read_loop(self) -> None:
        while self._proc and self._proc.isalive():
            try:
                data = self._proc.read(4096)
                if data:
                    self._socketio.emit(
                        "terminal_output", {"data": data},
                        room=self._sid, namespace="/hci-terminal"
                    )
            except Exception:
                break

    def write(self, data: str) -> None:
        if self._proc and self._proc.isalive():
            try:
                self._proc.write(data)
            except Exception as e:
                logger.debug(f"PTY write error: {e}")

    def resize(self, rows: int, cols: int) -> None:
        if self._proc and self._proc.isalive():
            try:
                self._proc.setwinsize(rows, cols)
            except Exception:
                pass

    def close(self) -> None:
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass


def register_terminal_namespace(socketio) -> None:
    """Register SocketIO event handlers for /hci-terminal namespace."""
    sessions: dict[str, TerminalSession] = {}

    @socketio.on("connect", namespace="/hci-terminal")
    def on_connect():
        from flask import request
        sid = request.sid
        if not _WINPTY_AVAILABLE:
            socketio.emit(
                "terminal_output",
                {"data": "\r\nTerminal unavailable. Run: pip install pywinpty\r\n"},
                room=sid, namespace="/hci-terminal"
            )
            return
        sessions[sid] = TerminalSession(sid, socketio)

    @socketio.on("terminal_input", namespace="/hci-terminal")
    def on_input(data):
        from flask import request
        sess = sessions.get(request.sid)
        if sess:
            sess.write(data.get("data", ""))

    @socketio.on("terminal_resize", namespace="/hci-terminal")
    def on_resize(data):
        from flask import request
        sess = sessions.get(request.sid)
        if sess:
            sess.resize(data.get("rows", 24), data.get("cols", 80))

    @socketio.on("disconnect", namespace="/hci-terminal")
    def on_disconnect():
        from flask import request
        sess = sessions.pop(request.sid, None)
        if sess:
            sess.close()
