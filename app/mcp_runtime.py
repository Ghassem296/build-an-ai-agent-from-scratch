"""Shared MCP runtime: subprocess environment and client session setup.

Both entry points (``app.main`` and ``mcp_agent.py``) used to carry their own
copy of this logic. Keeping it in one place means the allowlist of forwarded
environment variables and the stdio wiring can only be wrong in one file.
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# Only these variables are forwarded to the MCP subprocess. Anything else in
# the parent environment (API keys, tokens, cloud credentials, ...) stays out
# of the child process on purpose.
ALLOWED_ENV_VARS: tuple[str, ...] = (
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


# Startup is bounded so a broken server cannot hang the CLI forever.
INIT_TIMEOUT_SECONDS = 60.0


def build_mcp_environment() -> dict[str, str]:
    """Return a minimal, UTF-8 friendly environment for the MCP server."""
    env = {
        name: os.environ[name]
        for name in ALLOWED_ENV_VARS
        if name in os.environ
    }

    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    return env


def build_server_params(project_root: Path) -> StdioServerParameters:
    """Build the stdio parameters used to launch ``tools.mcp_server``."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "tools.mcp_server"],
        cwd=str(project_root),
        env=build_mcp_environment(),
    )


@asynccontextmanager
async def mcp_session(project_root: Path) -> AsyncIterator[ClientSession]:
    """Open an initialized MCP session and close it on exit."""
    server_params = build_server_params(project_root)

    async with AsyncExitStack() as stack:
        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(server_params)
        )

        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        async with asyncio.timeout(INIT_TIMEOUT_SECONDS):
            await session.initialize()

        yield session
