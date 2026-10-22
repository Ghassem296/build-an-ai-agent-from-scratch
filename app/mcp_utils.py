from __future__ import annotations

import os


_ALLOWED_ENV_NAMES = (
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "HOME",
    "LANG",
    "LC_ALL",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)


def build_mcp_environment() -> dict[str, str]:
    """Return a safe subset of environment variables for the MCP server."""
    env = {
        name: os.environ[name]
        for name in _ALLOWED_ENV_NAMES
        if name in os.environ
    }

    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    return env